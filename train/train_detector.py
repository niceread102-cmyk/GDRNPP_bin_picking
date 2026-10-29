"""Train the Mask R-CNN part detector (stage 1).

    .venv\\Scripts\\python train/train_detector.py --config configs/default.yaml [--set detector.train.epochs=20] [--resume]
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root: bin6d, data_conversion, train
from data_conversion.bin_dataset import BinPoseDataset
from bin6d.config import parse_args, save_config
from bin6d.detector import DetectionDataset, build_detector, det_collate, set_native_scale
from train.train_pose import split_scenes, warmup_cosine


@torch.no_grad()
def evaluate(model, loader, ec, device, img_hw) -> dict:
    """Greedy mask-IoU matching. Recall over GT with >= ec.min_visib_px visible px; precision over all GT."""
    model.eval()
    set_native_scale(model, *img_hw)
    tp_all = n_pred = tp_big = n_big = 0
    ious = []
    for imgs, targets in loader:
        outs = model([i.to(device) for i in imgs])
        for out, tgt in zip(outs, targets):
            keep = out["scores"] >= ec.score_thresh
            pm = (out["masks"][keep, 0] >= ec.mask_thresh)[:, ::2, ::2].flatten(1).half()
            inst = tgt["inst_full"].to(device)
            ids, counts = torch.unique(inst, return_counts=True)
            ids, counts = ids[ids > 0], counts[ids > 0]
            gm = (inst[None] == ids[:, None, None])[:, ::2, ::2].flatten(1).half()
            n_pred += len(pm)
            big = counts >= ec.min_visib_px
            n_big += int(big.sum())
            if len(pm) == 0 or len(gm) == 0:
                continue
            inter = pm @ gm.T
            iou = inter / (pm.sum(1)[:, None] + gm.sum(1)[None] - inter).clamp(min=1)
            matched_gt = torch.zeros(len(gm), dtype=torch.bool, device=device)
            for p in torch.argsort(out["scores"][keep], descending=True):
                row = iou[p].masked_fill(matched_gt, -1)
                j = int(row.argmax())
                if row[j] >= ec.iou_thresh:
                    matched_gt[j] = True
                    tp_all += 1
                    ious.append(float(row[j]))
            tp_big += int((matched_gt & big).sum())
    model.train()
    prec = tp_all / max(n_pred, 1)
    rec = tp_big / max(n_big, 1)
    return {"precision": prec, "recall_big": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "mean_matched_iou": float(np.mean(ious)) if ious else 0.0, "n_pred": n_pred, "n_gt_big": n_big}


def main():
    args, cfg = parse_args(__doc__, lambda ap: ap.add_argument("--resume", action="store_true"))
    dc, tc = cfg.detector, cfg.detector.train
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if tc.get("gpu_mem_fraction"):                # hard cap for this process (caching allocator frees first)
        torch.cuda.set_per_process_memory_fraction(tc.gpu_mem_fraction, device.index or 0)
    out_dir = Path(dc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    all_frames = BinPoseDataset(cfg.data.root, bin_margin_mm=cfg.data.bin_margin_mm)
    train_scenes, val_scenes = split_scenes(all_frames, cfg.data.val_scenes)
    train_frames = BinPoseDataset(cfg.data.root, scenes=train_scenes, bin_margin_mm=cfg.data.bin_margin_mm)
    val_frames = BinPoseDataset(cfg.data.root, scenes=val_scenes, bin_margin_mm=cfg.data.bin_margin_mm)
    img_hw = all_frames.load_images(0, 1.0)[0].shape[:2]

    nw = cfg.data.num_workers
    # spawn (the Windows default): forked workers deadlock on Linux
    kw = dict(collate_fn=det_collate, num_workers=nw, persistent_workers=nw > 0, pin_memory=True,
              multiprocessing_context="spawn" if nw > 0 else None)
    train_loader = DataLoader(DetectionDataset(train_frames, dc, train=True), batch_size=tc.batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_set = Subset(DetectionDataset(val_frames, dc, train=False),
                     list(range(0, len(val_frames), dc.eval.view_stride)))
    val_loader = DataLoader(val_set, batch_size=1, **kw)
    print(f"train {len(train_frames)} frames {train_scenes} | val {len(val_set)} frames {val_scenes}")

    model = build_detector(dc).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tc.lr, momentum=tc.momentum, weight_decay=tc.weight_decay)
    total = tc.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, tc.warmup_iters, total))
    scaler = torch.amp.GradScaler("cuda", enabled=tc.amp)

    start_epoch, best, step = 0, -1.0, 0
    ckpt_path = out_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best, step = ck["epoch"] + 1, ck["best"], ck["step"]
        print(f"resumed from epoch {ck['epoch']}")

    writer = SummaryWriter(out_dir / "tb")
    for epoch in range(start_epoch, tc.epochs):
        model.train()
        set_native_scale(model, dc.train_crop, dc.train_crop)
        t0, run = time.time(), {}
        for it, (imgs, targets) in enumerate(train_loader):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items() if k != "inst_id"}
                       for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=tc.amp):
                losses = model(imgs, targets)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses["total"] = loss
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            if (it + 1) % tc.log_every == 0:
                msg = " ".join(f"{k}={v / tc.log_every:.4f}" for k, v in run.items())
                print(f"ep {epoch} it {it + 1}/{len(train_loader)} lr={sched.get_last_lr()[0]:.2e} {msg} "
                      f"({(time.time() - t0) / (it + 1):.2f}s/it)")
                for k, v in run.items():
                    writer.add_scalar(f"train/{k}", v / tc.log_every, step)
                run = {}

        if (epoch + 1) % dc.eval.every_epochs == 0 or epoch == tc.epochs - 1:
            m = evaluate(model, val_loader, dc.eval, device, img_hw)
            print(f"== val ep {epoch}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in m.items()))
            for k, v in m.items():
                writer.add_scalar(f"val/{k}", v, step)
            if m["f1"] > best:
                best = m["f1"]
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch, "metrics": m},
                           out_dir / "best.pt")
                print(f"   new best F1 = {best:.4f} -> {out_dir / 'best.pt'}")

        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "step": step, "cfg": cfg.to_dict()}, ckpt_path)
    writer.close()


if __name__ == "__main__":
    main()
