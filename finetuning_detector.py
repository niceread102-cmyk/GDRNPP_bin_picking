"""Fine-tune the sim-trained Mask R-CNN detector on sim + real captures (stage 1).

Starts from detector.finetune.init (the sim-trained checkpoint) and mixes real frames into every batch
(oversampled to detector.finetune.real_fraction) so the detector learns the real background / appearance
without forgetting full sim bins. Real frames are split by VIEW (a real root has a single scene); the held-out
real views and the sim val scene are both evaluated, and best.pt is chosen on their mean F1.

    .venv/bin/python finetuning_detector.py [--set detector.finetune.epochs=20] [--set detector.finetune.real_fraction=0.3]
    .venv/bin/python infer.py --detector runs/detector_ft/best.pt --set data.root=data/pseudo_gt_70_bop --scenes 000000 --view-stride 1 --out runs/photoneo70_ft
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from bin_dataset import BinPoseDataset
from bin6d.config import Cfg, parse_args, save_config
from bin6d.detector import DetectionDataset, build_detector, det_collate, set_native_scale
from train_detector import evaluate
from train_pose import split_scenes, warmup_cosine


def split_real_views(frames: BinPoseDataset, fc) -> tuple[list[int], list[int]]:
    """Frame indices (train, val) of the real root: explicit fc.real_val_views, else every fc.real_val_every-th
    view (offset 1), which spreads the held-out views over separated / overlapping / dense captures."""
    views = [f.view for f in frames.frames]
    if fc.get("real_val_views"):
        val_views = set(fc.real_val_views)
    else:
        val_views = set(sorted(set(views))[1::fc.real_val_every])
    train = [i for i, v in enumerate(views) if v not in val_views]
    val = [i for i, v in enumerate(views) if v in val_views]
    if not train or not val:
        raise ValueError(f"bad real split: train={len(train)} val={len(val)} frames")
    return train, val


def fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())


def main():
    args, cfg = parse_args(__doc__)
    dc, fc, ec = cfg.detector, cfg.detector.finetune, cfg.detector.eval
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if dc.train.get("gpu_mem_fraction"):
        torch.cuda.set_per_process_memory_fraction(dc.train.gpu_mem_fraction, device.index or 0)
    out_dir = Path(fc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    # sim: same scene split as train_detector.py
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
    img_hw = real.load_images(0, 1.0)[0].shape[:2]

    train_set = ConcatDataset([DetectionDataset(sim_train, dc, train=True),
                               Subset(DetectionDataset(real, dc, train=True), real_train_idx)])
    n_sim, n_real = len(sim_train), len(real_train_idx)
    weights = torch.cat([torch.full((n_sim,), (1 - fc.real_fraction) / n_sim),
                         torch.full((n_real,), fc.real_fraction / n_real)]).double()
    sampler = WeightedRandomSampler(weights, num_samples=fc.iters_per_epoch * fc.batch_size, replacement=True)

    nw = cfg.data.num_workers
    # spawn (the Windows default): forked workers deadlock on Linux
    kw = dict(collate_fn=det_collate, num_workers=nw, persistent_workers=nw > 0, pin_memory=True,
              multiprocessing_context="spawn" if nw > 0 else None)
    train_loader = DataLoader(train_set, batch_size=fc.batch_size, sampler=sampler, drop_last=True, **kw)
    val_loaders = {
        "real": DataLoader(Subset(DetectionDataset(real, dc, train=False), real_val_idx), batch_size=1, **kw),
        "sim": DataLoader(Subset(DetectionDataset(sim_val, dc, train=False),
                                 list(range(0, len(sim_val), fc.sim_val_view_stride))), batch_size=1, **kw),
    }
    print(f"sim train {n_sim} frames {sim_train_scenes} | real train {n_real} frames, "
          f"real val {len(real_val_idx)} frames (views {split['real_val_views']}) | "
          f"sim val {len(val_loaders['sim'].dataset)} frames {sim_val_scenes} | real share per batch {fc.real_fraction}")

    ck = torch.load(fc.init, map_location=device, weights_only=False)
    init_dc = Cfg(ck["cfg"]).detector
    init_dc.pretrained = False                         # weights come from the checkpoint
    model = build_detector(init_dc).to(device)
    model.load_state_dict(ck["model"])
    if fc.freeze_backbone:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=fc.lr, momentum=dc.train.momentum, weight_decay=dc.train.weight_decay)
    total = fc.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, fc.warmup_iters, total))
    scaler = torch.amp.GradScaler("cuda", enabled=dc.train.amp)
    writer = SummaryWriter(out_dir / "tb")

    def validate(epoch: int, step: int) -> dict:
        ms = {name: evaluate(model, loader, ec, device, img_hw) for name, loader in val_loaders.items()}
        for name, m in ms.items():
            print(f"== val {name} ep {epoch}: {fmt(m)}")
            for k, v in m.items():
                writer.add_scalar(f"val_{name}/{k}", v, step)
        ms["score"] = 0.5 * (ms["real"]["f1"] + ms["sim"]["f1"])
        return ms

    m0 = validate(-1, 0)                               # the sim-trained starting point
    best, step = m0["score"], 0
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": -1, "metrics": m0, "split": split},
               out_dir / "best.pt")
    print(f"   start score (mean F1 real/sim) = {best:.4f}")

    for epoch in range(fc.epochs):
        model.train()
        set_native_scale(model, dc.train_crop, dc.train_crop)
        t0, run = time.time(), {}
        for it, (imgs, targets) in enumerate(train_loader):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items() if k != "inst_id"}
                       for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=dc.train.amp):
                losses = model(imgs, targets)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, dc.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses["total"] = loss
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            if (it + 1) % dc.train.log_every == 0:
                msg = " ".join(f"{k}={v / dc.train.log_every:.4f}" for k, v in run.items())
                print(f"ep {epoch} it {it + 1}/{len(train_loader)} lr={sched.get_last_lr()[0]:.2e} {msg} "
                      f"({(time.time() - t0) / (it + 1):.2f}s/it)")
                for k, v in run.items():
                    writer.add_scalar(f"train/{k}", v / dc.train.log_every, step)
                run = {}

        if (epoch + 1) % fc.eval_every_epochs == 0 or epoch == fc.epochs - 1:
            ms = validate(epoch, step)
            if ms["score"] > best:
                best = ms["score"]
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch, "metrics": ms,
                            "split": split}, out_dir / "best.pt")
                print(f"   new best score (mean F1 real/sim) = {best:.4f} -> {out_dir / 'best.pt'}")
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch, "split": split},
                   out_dir / "last.pt")
    writer.close()
    print(f"start: real F1 {m0['real']['f1']:.4f}, sim F1 {m0['sim']['f1']:.4f} | best score {best:.4f}")


if __name__ == "__main__":
    main()
