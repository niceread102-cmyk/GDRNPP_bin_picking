"""Full pipeline on dataset frames: detect -> GDRN pose -> ICP, with GT evaluation and visualisation.

    .venv\\Scripts\\python infer.py --detector runs/detector/best.pt --pose runs/pose/best.pt
    .venv\\Scripts\\python infer.py --pose runs/pose/best.pt --gt-masks     # skip detector (GT masks)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch

from bin_dataset import BinPoseDataset
from bin6d.config import Cfg, parse_args
from bin6d.detector import build_detector, set_native_scale
from bin6d.geometry import bbox_to_crop, crop_pixel_grid, pixel_rays, sample_crops_batched
from bin6d.pose_data import ObjectModel, build_network_input, frame_planes, input_channels
from bin6d.pose_model import GDRNet, decode_pose, pose_errors
from bin6d.refine import flag_confident, icp_refine, observed_points
from bin6d.viz import gt_overlay_sets, render_2d


def extra_args(ap):
    ap.add_argument("--detector", default="runs/detector/best.pt")
    ap.add_argument("--pose", default="runs/pose/best.pt")
    ap.add_argument("--gt-masks", action="store_true", help="use GT visible masks instead of the detector")
    ap.add_argument("--scenes", nargs="*", default=None, help="default: data.val_scenes")
    ap.add_argument("--view-stride", type=int, default=10)
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--no-viz", action="store_true")


def masks_to_xywh(masks: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import masks_to_boxes
    b = masks_to_boxes(masks)
    return torch.cat([b[:, :2], b[:, 2:] - b[:, :2] + 1], dim=1)


@torch.no_grad()
def predict_poses(net, pc, model, rgb_t, xyz_t, valid_t, masks, K, device, batch):
    """masks (N,H,W) bool on device -> R (N,3,3), t (N,3) numpy."""
    bbox = masks_to_xywh(masks).float()
    center, size = bbox_to_crop(bbox, pc.pad_scale, pc.min_crop_px)
    Rs, ts = [], []
    for s in range(0, len(masks), batch):
        c, z = center[s:s + batch], size[s:s + batch]
        uv_in = crop_pixel_grid(c, z, pc.input_res)
        uv_out = crop_pixel_grid(c, z, pc.output_res)
        m = sample_crops_batched(masks[s:s + batch, None].float(), uv_in, "nearest")
        x = build_network_input(rgb_t, xyz_t, valid_t, m, uv_in, model.diameter)
        Kb = K.expand(len(c), 3, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, pixel_rays(uv_out, Kb))
        R, t = decode_pose(out, Kb, c, z, model.diameter)
        Rs.append(R)
        ts.append(t)
    return torch.cat(Rs).cpu().numpy(), torch.cat(ts).cpu().numpy()


@torch.no_grad()
def main():
    args, cfg = parse_args(__doc__, extra_args)
    device = torch.device(cfg.device)
    ck_pose = torch.load(args.pose, map_location=device, weights_only=False)
    pc = Cfg(ck_pose["cfg"]).pose                  # use the architecture the checkpoint was trained with
    pc.pretrained = False                          # weights come from the checkpoint
    rc, ic = cfg.refine, cfg.infer

    frames = BinPoseDataset(cfg.data.root, scenes=args.scenes or cfg.data.val_scenes,
                            bin_margin_mm=cfg.data.bin_margin_mm)
    model = ObjectModel(frames, cfg.data.obj_id, pc.num_regions, pc.num_model_points)
    net = GDRNet(pc, input_channels()).to(device).eval()
    net.load_state_dict(ck_pose["model"])

    det = None
    if not args.gt_masks:
        ck_det = torch.load(args.detector, map_location=device, weights_only=False)
        det = build_detector(Cfg(ck_det["cfg"]).detector).to(device).eval()
        det.load_state_dict(ck_det["model"])
        det.roi_heads.score_thresh = ic.score_thresh

    pts_mm = torch.from_numpy(model.points).to(device)
    sym_R = torch.from_numpy(model.sym_R).to(device)
    sym_t = torch.from_numpy(model.sym_t).to(device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = {"n_frames": 0, "n_dets": 0, "n_matched": 0, "add_s": [], "add_s_icp": [], "refined": 0,
           "confident": 0, "confident_add_s": [], "confident_unmatched": 0}
    for idx in range(0, len(frames), args.view_stride):
        meta = frames.frame_meta(idx)
        rgb, depth, inst = frames.load_images(idx, meta["depth_scale"])
        K_np = meta["K"]
        rgb_t, xyz_t, valid_t = (x.to(device) for x in frame_planes(rgb, depth, K_np))
        K = torch.from_numpy(K_np).to(device)
        inst_t = torch.from_numpy(inst).to(device)
        objs = meta["objects_all"]

        if det is None:
            keep = objs["visib_px"] >= pc.min_visib_px
            ids = torch.from_numpy(objs["inst_id"][keep]).to(device)
            masks = inst_t[None] == ids[:, None, None]
            scores = np.ones(len(ids))
        else:
            set_native_scale(det, *rgb.shape[:2])
            o = det([rgb_t])[0]
            masks = o["masks"][:, 0] >= 0.5
            scores = o["scores"].cpu().numpy()
            big = masks.flatten(1).sum(1) >= ic.min_mask_px
            masks, scores = masks[big], scores[big.cpu().numpy()]
        if len(masks) == 0:
            continue

        R, t = predict_poses(net, pc, model, rgb_t, xyz_t, valid_t, masks, K, device, ic.pose_batch)

        # match detections to GT by majority instance id inside the mask (IoU >= 0.5)
        gt_index = {int(i): k for k, i in enumerate(objs["inst_id"])}
        xyz_np, valid_np = xyz_t.permute(1, 2, 0).cpu().numpy(), depth > 0
        results = []
        for n in range(len(masks)):
            m = masks[n]
            vals = inst_t[m]
            vals = vals[vals > 0]
            gt_k = None
            if len(vals):
                gid = int(torch.mode(vals).values)
                gm = inst_t == gid
                iou = float((gm & m).sum() / (gm | m).sum())
                if iou >= 0.5 and gid in gt_index:
                    gt_k = gt_index[gid]
            m_np = m.cpu().numpy()
            ref = icp_refine(R[n], t[n], observed_points(xyz_np, valid_np, m_np, rc.mask_erode_px),
                             model.dense_points, model.dense_normals, rc)
            r = {"score": float(scores[n]), "R_rgbd": R[n].tolist(), "t_rgbd": t[n].tolist(),
                 "R": np.asarray(ref["R"]).tolist(), "t": np.asarray(ref["t"]).tolist(),
                 "icp_refined": ref["refined"], "icp_fitness": ref["fitness"], "icp_rmse": ref["rmse"],
                 "n_depth_points": ref["n_points"], "gt_inst_id": None}
            if gt_k is not None:
                Rg = torch.from_numpy(objs["R"][gt_k][None]).to(device)
                tg = torch.from_numpy(objs["t"][gt_k][None]).to(device)
                e0 = pose_errors(torch.from_numpy(R[n][None]).float().to(device),
                                 torch.from_numpy(t[n][None]).float().to(device), Rg, tg, pts_mm, sym_R, sym_t)
                e1 = pose_errors(torch.from_numpy(np.asarray(ref["R"])[None]).float().to(device),
                                 torch.from_numpy(np.asarray(ref["t"])[None]).float().to(device), Rg, tg,
                                 pts_mm, sym_R, sym_t)
                r.update(gt_inst_id=int(objs["inst_id"][gt_k]), add_s=float(e0["add_s"]),
                         add_s_refined=float(e1["add_s"]), rot_deg_refined=float(e1["rot_deg"]),
                         add_s_refined_ok=bool(e1["add_s"] < 0.1 * model.diameter))
                agg["add_s"].append(r["add_s"])
                agg["add_s_icp"].append(r["add_s_refined"])
                agg["n_matched"] += 1
            agg["refined"] += ref["refined"]
            results.append(r)

        flag_confident(results, rc)
        for r in results:
            if r["confident"]:
                agg["confident"] += 1
                if r["gt_inst_id"] is None:
                    agg["confident_unmatched"] += 1
                else:
                    agg["confident_add_s"].append(r["add_s_refined"])
        agg["n_frames"] += 1
        agg["n_dets"] += len(results)
        tag = f"{meta['scene']}_{meta['view']:06d}"
        with open(out_dir / f"{tag}.json", "w") as f:
            json.dump(results, f)
        if not args.no_viz:
            gt = gt_overlay_sets(objs, results, pc.min_visib_px)
            render_2d(rgb, K_np, results, model, gt, title=tag).save(out_dir / f"{tag}.png")
        a0, a1 = np.array([r.get("add_s", np.nan) for r in results]), np.array(
            [r.get("add_s_refined", np.nan) for r in results])
        d01 = 0.1 * model.diameter
        conf = np.array([r["confident"] for r in results])
        conf_err = a1[conf & np.isfinite(a1)]
        print(f"{tag}: {len(results)} dets, matched {np.isfinite(a0).sum()}, "
              f"ADD-S<0.1d rgbd {np.nanmean(a0 < d01) if np.isfinite(a0).any() else float('nan'):.3f} "
              f"-> icp {np.nanmean(a1 < d01) if np.isfinite(a1).any() else float('nan'):.3f} | "
              f"confident {conf.sum()} (ADD-S<1mm {(conf_err < 1).sum()}/{len(conf_err)} matched)")

    d01 = 0.1 * model.diameter
    a0, a1 = np.array(agg["add_s"]), np.array(agg["add_s_icp"])
    summary = {"frames": agg["n_frames"], "detections": agg["n_dets"], "matched_to_gt": agg["n_matched"],
               "icp_applied": agg["refined"]}
    if len(a0):
        summary.update({"add_s_0.1d_rgbd": float((a0 < d01).mean()), "add_s_0.1d_icp": float((a1 < d01).mean()),
                        "add_s_median_rgbd_mm": float(np.median(a0)), "add_s_median_icp_mm": float(np.median(a1))})
    c = np.array(agg["confident_add_s"])
    summary.update({"confident": agg["confident"], "confident_per_frame": agg["confident"] / max(agg["n_frames"], 1),
                    "confident_unmatched_to_gt": agg["confident_unmatched"]})
    if len(c):
        summary.update({"confident_matched": len(c), "confident_add_s_0.5mm": float((c < 0.5).mean()),
                        "confident_add_s_1mm": float((c < 1).mean()), "confident_add_s_2mm": float((c < 2).mean()),
                        "confident_wrong_ge_1mm": int((c >= 1).sum())})
    print(json.dumps(summary, indent=1))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
