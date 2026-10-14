"""Fine-tune the sim-trained GDRN pose network on sim + real captures (stage 2).

Starts from pose.finetune.init (the sim-trained checkpoint; its architecture and crop / augmentation settings are
kept) and mixes real frames into every batch (oversampled to pose.finetune.real_fraction). Real targets come from
the pseudo-GT poses (FoundationPose) raycast on the CAD, exactly like the sim targets. Real frames are split by VIEW,
by default with the held-out views of the detector fine-tuning (pose.finetune.real_split_file) so both stages are
judged on the same unseen captures.

The final pose comes from depth anchoring + ICP; the network only has to land inside the ICP basin. best.pt is
therefore chosen on rotation error (pose.finetune.select_metric, mean of the real held-out views and the sim val
scene; lower is better), not on ADD-S, which on real data is dominated by the depth bias the anchor removes.
The median is the default: the p95 of a few hundred sim crops rests on ~16 crops and jumps between epochs (the
first run picked epoch 1 on it, while the real metrics kept improving to epoch 11). The share of crops under
pose.finetune.select_rot_deg is logged too; on real GT-mask crops it starts at ~100 %, so it cannot rank.

    .venv/bin/python finetuning_pose.py [--set pose.finetune.epochs=20] [--set pose.finetune.real_fraction=0.3]
    .venv/bin/python infer.py --pose runs/pose_ft/best.pt --set data.root=data/pseudo_gt_70_bop --scenes 000000 --view-stride 1 --out runs/photoneo70_poseft
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from bin_dataset import BinPoseDataset
from bin6d.config import Cfg, parse_args, save_config
from bin6d.pose_data import ObjectModel, PoseCropDataset, cat_collate, input_channels
from bin6d.pose_model import GDRNet, decode_pose, pose_errors, pose_losses
from train_pose import autocast_ctx, split_scenes, warmup_cosine


def split_real_views(frames: BinPoseDataset, fc) -> tuple[list[int], list[int]]:
    """Frame indices (train, val) of the real root. Held-out views: fc.real_split_file (the detector fine-tuning
    split) if it exists, else every fc.real_val_every-th view (offset 1)."""
    views = [f.view for f in frames.frames]
    split_file = Path(fc.real_split_file) if fc.get("real_split_file") else None
    if split_file is not None and split_file.exists():
        val_views = set(json.load(open(split_file))["real_val_views"])
    else:
        val_views = set(sorted(set(views))[1::fc.real_val_every])
    train = [i for i, v in enumerate(views) if v not in val_views]
    val = [i for i, v in enumerate(views) if v in val_views]
    if not train or not val:
        raise ValueError(f"bad real split: train={len(train)} val={len(val)} frames")
    return train, val


@torch.no_grad()
def evaluate(net, loader, model: ObjectModel, pts_mm, sym_R, sym_t, device, amp: str, rot_deg: float) -> dict:
    """Network-only pose errors on GT crops; adds the rotation tail that decides whether ICP converges."""
    net.eval()
    errs = {"add_s": [], "rot_deg": [], "trans_mm": [], "z_mm": []}
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with autocast_ctx(amp):
            out = net(b["input"], b["rays"])
        R, t = decode_pose(out, b["K"], b["center"], b["size"], model.diameter)
        for k, v in pose_errors(R, t, b["R"], b["t"], pts_mm, sym_R, sym_t).items():
            errs[k].append(v.float().cpu())
    net.train()
    e = {k: torch.cat(v) for k, v in errs.items()}
    return {
        f"rot<{rot_deg:g}deg": (e["rot_deg"] < rot_deg).float().mean().item(),
        "rot_deg_median": e["rot_deg"].median().item(),
        "rot_deg_p95": e["rot_deg"].quantile(0.95).item(),
        "rot_deg_max": e["rot_deg"].max().item(),
        "add_s_0.1d": (e["add_s"] < 0.1 * model.diameter).float().mean().item(),
        "trans_mm_median": e["trans_mm"].median().item(),
        "z_mm_median": e["z_mm"].median().item(),
        "n": len(e["add_s"]),
    }


def fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())


def main():
    args, cfg = parse_args(__doc__)
    fc = cfg.pose.finetune
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    tc = cfg.pose.train
    if tc.get("gpu_mem_fraction"):
        torch.cuda.set_per_process_memory_fraction(tc.gpu_mem_fraction, device.index or 0)
    out_dir = Path(fc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(fc.init, map_location=device, weights_only=False)
    pc = Cfg(ck["cfg"]).pose                       # architecture, crops and augmentation of the checkpoint
    pc.pretrained = False                          # weights come from the checkpoint
    pc.use_pretrained_model = False
    out_cfg = cfg.to_dict()
    out_cfg["pose"] = {**pc.to_dict(), "finetune": fc.to_dict()}
    save_config(Cfg(out_cfg), out_dir / "config.yaml")

    # sim: same scene split as train_pose.py
    sim_all = BinPoseDataset(cfg.data.root, bin_margin_mm=cfg.data.bin_margin_mm)
    sim_train_scenes, sim_val_scenes = split_scenes(sim_all, cfg.data.val_scenes)
    sim_train = BinPoseDataset(cfg.data.root, scenes=sim_train_scenes, bin_margin_mm=cfg.data.bin_margin_mm)
    sim_val = BinPoseDataset(cfg.data.root, scenes=sim_val_scenes, bin_margin_mm=cfg.data.bin_margin_mm)
    # real: split by view
    real = BinPoseDataset(fc.real_root)
    real_train_idx, real_val_idx = split_real_views(real, fc)
    split = {"real_root": fc.real_root,
             "real_train_views": [real.frames[i].view for i in real_train_idx],
             "real_val_views": [real.frames[i].view for i in real_val_idx]}
    with open(out_dir / "real_split.json", "w") as f:
        json.dump(split, f, indent=1)
    model = ObjectModel(sim_train, cfg.data.obj_id, pc.num_regions, pc.num_model_points)

    obj = cfg.data.obj_id
    train_set = ConcatDataset([PoseCropDataset(sim_train, model, pc, train=True, obj_id=obj),
                               Subset(PoseCropDataset(real, model, pc, train=True, obj_id=obj), real_train_idx)])
    n_sim, n_real = len(sim_train), len(real_train_idx)
    weights = torch.cat([torch.full((n_sim,), (1 - fc.real_fraction) / n_sim),
                         torch.full((n_real,), fc.real_fraction / n_real)]).double()
    sampler = WeightedRandomSampler(weights, num_samples=fc.iters_per_epoch * fc.batch_frames, replacement=True)

    nw = cfg.data.num_workers
    # spawn (the Windows default): forked workers deadlock on Linux
    kw = dict(num_workers=nw, collate_fn=cat_collate, pin_memory=True, persistent_workers=nw > 0,
              multiprocessing_context="spawn" if nw > 0 else None)
    train_loader = DataLoader(train_set, batch_size=fc.batch_frames, sampler=sampler, drop_last=True, **kw)
    val_loaders = {
        "real": DataLoader(Subset(PoseCropDataset(real, model, pc, train=False, obj_id=obj), real_val_idx),
                           batch_size=fc.batch_frames, **kw),
        "sim": DataLoader(Subset(PoseCropDataset(sim_val, model, pc, train=False, obj_id=obj),
                                 list(range(0, len(sim_val), fc.sim_val_view_stride))),
                          batch_size=fc.batch_frames, **kw),
    }
    print(f"sim train {n_sim} frames {sim_train_scenes} | real train {n_real} frames, "
          f"real val {len(real_val_idx)} frames (views {split['real_val_views']}) | "
          f"sim val {len(val_loaders['sim'].dataset)} frames {sim_val_scenes} | real share per batch {fc.real_fraction} | "
          f"{fc.batch_frames} frames x {pc.crops_per_frame} crops per step")

    net = GDRNet(pc, input_channels())
    net.load_state_dict(ck["model"])
    net = net.to(device)
    if fc.freeze_backbone:
        for p in net.backbone.parameters():
            p.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=fc.lr, weight_decay=tc.weight_decay)
    total = fc.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, fc.warmup_iters, total))
    scaler = torch.amp.GradScaler("cuda", enabled=tc.amp == "fp16")
    writer = SummaryWriter(out_dir / "tb")

    pts_mm = torch.from_numpy(model.points).to(device)
    pts_norm = pts_mm / model.diameter
    sym_R = torch.from_numpy(model.sym_R).to(device)
    sym_t = torch.from_numpy(model.sym_t).to(device)
    key = f"rot<{fc.select_rot_deg:g}deg"
    sel = fc.select_metric
    if sel not in ("rot_deg_median", "rot_deg_p95"):
        raise ValueError(f"pose.finetune.select_metric must be rot_deg_median or rot_deg_p95, got {sel!r}")

    def validate(epoch: int, step: int) -> dict:
        ms = {name: evaluate(net, loader, model, pts_mm, sym_R, sym_t, device, tc.amp, fc.select_rot_deg)
              for name, loader in val_loaders.items()}
        for name, m in ms.items():
            print(f"== val {name} ep {epoch}: {fmt(m)}")
            for k, v in m.items():
                writer.add_scalar(f"val_{name}/{k}", v, step)
        ms["score"] = 0.5 * (ms["real"][sel] + ms["sim"][sel])                         # lower is better
        return ms

    def save(path: Path, epoch: int, metrics: dict | None):
        torch.save({"model": net.state_dict(), "cfg": out_cfg, "epoch": epoch, "metrics": metrics, "split": split},
                   path)

    m0 = validate(-1, 0)                               # the sim-trained starting point
    best, step = m0["score"], 0
    save(out_dir / "best.pt", -1, m0)
    print(f"   start score (mean {sel} real/sim) = {best:.3f}")

    for epoch in range(fc.epochs):
        net.train()
        t0, run = time.time(), {}
        for it, b in enumerate(train_loader):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with autocast_ctx(tc.amp):
                out = net(b["input"], b["rays"])
            losses = pose_losses(out, b, pts_norm, sym_R, pc.loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            if (it + 1) % fc.log_every == 0:
                msg = " ".join(f"{k}={v / fc.log_every:.4f}" for k, v in run.items())
                print(f"ep {epoch} it {it + 1}/{len(train_loader)} lr={sched.get_last_lr()[0]:.2e} {msg} "
                      f"({(time.time() - t0) / (it + 1):.2f}s/it)")
                for k, v in run.items():
                    writer.add_scalar(f"train/{k}", v / fc.log_every, step)
                run = {}

        if (epoch + 1) % fc.eval_every_epochs == 0 or epoch == fc.epochs - 1:
            ms = validate(epoch, step)
            if ms["score"] < best:
                best = ms["score"]
                save(out_dir / "best.pt", epoch, ms)
                print(f"   new best score (mean {sel} real/sim) = {best:.3f} -> {out_dir / 'best.pt'}")
        save(out_dir / "last.pt", epoch, None)
    writer.close()
    print(f"start: {sel} real {m0['real'][sel]:.2f} / sim {m0['sim'][sel]:.2f}, "
          f"{key} real {m0['real'][key]:.4f} / sim {m0['sim'][key]:.4f} | best score {best:.3f}")


if __name__ == "__main__":
    main()
