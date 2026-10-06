"""Train the GDRN-style RGB-D pose network on GT instance crops.

    .venv\\Scripts\\python train_pose.py --config configs/default.yaml [--set pose.train.epochs=50] [--resume]
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bin_dataset import BinPoseDataset
from bin6d.config import parse_args, save_config
from bin6d.pose_data import ObjectModel, PoseCropDataset, cat_collate, input_channels
from bin6d.pose_model import GDRNet, decode_pose, load_gdrnpp_backbone, pose_errors, pose_losses


def split_scenes(frames: BinPoseDataset, val_scenes: list[str]) -> tuple[list[str], list[str]]:
    scenes = sorted({f.scene for f in frames.frames})
    val = [s for s in scenes if s in set(val_scenes)]
    train = [s for s in scenes if s not in set(val_scenes)]
    if not train or not val:
        raise ValueError(f"bad split: train={train} val={val}")
    return train, val


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def autocast_ctx(mode: str):
    if mode == "none":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


@torch.no_grad()
def evaluate(net, loader, model: ObjectModel, pts_mm, sym_R, sym_t, device, amp: str) -> dict:
    net.eval()
    errs = {"add_s": [], "rot_deg": [], "trans_mm": [], "z_mm": []}
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with autocast_ctx(amp):
            out = net(b["input"], b["rays"])
        R, t = decode_pose(out, b["K"], b["center"], b["size"], model.diameter)
        for k, v in pose_errors(R, t, b["R"], b["t"], pts_mm, sym_R, sym_t).items():
            errs[k].append(v.cpu())
    net.train()
    e = {k: torch.cat(v) for k, v in errs.items()}
    d = model.diameter
    return {
        "add_s_mean_mm": e["add_s"].mean().item(),
        "add_s_0.1d": (e["add_s"] < 0.1 * d).float().mean().item(),   # standard BOP-style accuracy
        "add_s_0.05d": (e["add_s"] < 0.05 * d).float().mean().item(),
        "rot_deg_median": e["rot_deg"].median().item(),
        "trans_mm_median": e["trans_mm"].median().item(),
        "z_mm_median": e["z_mm"].median().item(),
        "n": len(e["add_s"]),
    }


def main():
    args, cfg = parse_args(__doc__, lambda ap: ap.add_argument("--resume", action="store_true"))
    pc, tc = cfg.pose, cfg.pose.train
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if tc.get("gpu_mem_fraction"):                # hard cap for this process (caching allocator frees first)
        torch.cuda.set_per_process_memory_fraction(tc.gpu_mem_fraction, device.index or 0)
    out_dir = Path(pc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    all_frames = BinPoseDataset(cfg.data.root, bin_margin_mm=cfg.data.bin_margin_mm)
    train_scenes, val_scenes = split_scenes(all_frames, cfg.data.val_scenes)
    train_frames = BinPoseDataset(cfg.data.root, scenes=train_scenes, bin_margin_mm=cfg.data.bin_margin_mm)
    val_frames = BinPoseDataset(cfg.data.root, scenes=val_scenes, bin_margin_mm=cfg.data.bin_margin_mm)
    model = ObjectModel(train_frames, cfg.data.obj_id, pc.num_regions, pc.num_model_points)
    print(f"train scenes {train_scenes} ({len(train_frames)} frames) | val scenes {val_scenes} ({len(val_frames)})")

    nw = cfg.data.num_workers
    loader_kw = dict(num_workers=nw, collate_fn=cat_collate, pin_memory=True, persistent_workers=nw > 0)
    train_loader = DataLoader(PoseCropDataset(train_frames, model, pc, train=True, obj_id=cfg.data.obj_id),
                              batch_size=tc.batch_frames, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(PoseCropDataset(val_frames, model, pc, train=False, obj_id=cfg.data.obj_id),
                            batch_size=tc.batch_frames, shuffle=False, **loader_kw)

    net = GDRNet(pc, input_channels())
    if pc.use_pretrained_model and not args.resume:
        load_gdrnpp_backbone(net, pc.pretrained_model)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    total_steps = tc.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, tc.warmup_iters, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=tc.amp == "fp16")

    pts_mm = torch.from_numpy(model.points).to(device)
    pts_norm = pts_mm / model.diameter
    sym_R = torch.from_numpy(model.sym_R).to(device)
    sym_t = torch.from_numpy(model.sym_t).to(device)

    start_epoch, best, step = 0, -1.0, 0
    ckpt_path = out_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best, step = ck["epoch"] + 1, ck["best"], ck["step"]
        print(f"resumed from epoch {ck['epoch']}")

    writer = SummaryWriter(out_dir / "tb")
    for epoch in range(start_epoch, tc.epochs):
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            if (it + 1) % tc.log_every == 0:
                msg = " ".join(f"{k}={v / tc.log_every:.4f}" for k, v in run.items())
                print(f"ep {epoch} it {it + 1}/{len(train_loader)} lr={sched.get_last_lr()[0]:.2e} {msg} "
                      f"({(time.time() - t0) / (it + 1):.2f}s/it)")
                for k, v in run.items():
                    writer.add_scalar(f"train/{k}", v / tc.log_every, step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                run = {}

        is_last = epoch == tc.epochs - 1
        if (epoch + 1) % pc.eval.every_epochs == 0 or is_last:
            metrics = evaluate(net, val_loader, model, pts_mm, sym_R, sym_t, device, tc.amp)
            print(f"== val ep {epoch}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in metrics.items()))
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, step)
            if metrics["add_s_0.1d"] > best:
                best = metrics["add_s_0.1d"]
                torch.save({"model": net.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch, "metrics": metrics},
                           out_dir / "best.pt")
                print(f"   new best ADD-S<0.1d = {best:.4f} -> {out_dir / 'best.pt'}")

        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "step": step, "cfg": cfg.to_dict()}, ckpt_path)
    writer.close()


if __name__ == "__main__":
    main()
