"""Resample a lens-distorted BOP dataset onto its pinhole K (rgb, depth, visible instance masks).

The pipeline back-projects depth with K only, so images rendered with lens distortion (e.g. FFB6D
side_plate_10k: PhoXi colour model, k1 = -0.3594 at 1952x1460, rendered at half resolution) put edge parts in the
wrong place by up to ~15 % of their size. The distortion is the OpenCV radial model, ideal -> distorted:
x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) on normalised coordinates (checked against the GT: projecting the CAD with it
puts every visible GT mask fully inside the silhouette, 1.00 at the edges vs 0.88 without).

GT poses and K are pinhole quantities and stay unchanged; rgb is sampled bilinearly, depth and masks nearest;
px_count_visib / bbox_visib in scene_gt_info.json are recomputed from the resampled masks.

    .venv/bin/python undistort_bop.py --src /home/joon/FFB6D/data/side_plate_10k --dst data/side_plate_10k_undist \
        --scenes 000000 000001 000002 000003 000004
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def source_pixels(K: np.ndarray, dist: list[float], h: int, w: int) -> np.ndarray:
    """(2, h, w) distorted-image (row, col) to sample for every pinhole pixel."""
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    if p1 or p2:
        raise NotImplementedError("tangential distortion (p1, p2) is not supported")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = (u - cx) / fx, (v - cy) / fy
    r2 = x * x + y * y
    s = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return np.stack([fy * y * s + cy, fx * x * s + cx])


def remap(img: np.ndarray, coords: np.ndarray, order: int) -> np.ndarray:
    if img.ndim == 3:
        return np.stack([remap(img[..., c], coords, order) for c in range(img.shape[2])], -1)
    out = map_coordinates(img.astype(np.float32), coords, order=order, mode="constant", cval=0)
    return np.round(out).astype(img.dtype) if order else out.astype(img.dtype)


def convert_scene(src: Path, dst: Path, K: np.ndarray, dist: list[float], coords_cache: dict) -> int:
    for sub in ("rgb", "depth", "mask_visib_inst"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for name in ("scene_gt.json", "scene_camera.json"):
        shutil.copyfile(src / name, dst / name)
    info = json.load(open(src / "scene_gt_info.json"))
    views = sorted(int(p.stem) for p in (src / "rgb").glob("*.png"))
    for view in views:
        name = f"{view:06d}.png"
        rgb = np.asarray(Image.open(src / "rgb" / name).convert("RGB"))
        h, w = rgb.shape[:2]
        if (h, w) not in coords_cache:
            coords_cache[(h, w)] = source_pixels(K, dist, h, w)
        coords = coords_cache[(h, w)]
        Image.fromarray(remap(rgb, coords, 1)).save(dst / "rgb" / name)
        depth = np.asarray(Image.open(src / "depth" / name))
        Image.fromarray(remap(depth, coords, 0)).save(dst / "depth" / name)
        mask = remap(np.asarray(Image.open(src / "mask_visib_inst" / name)), coords, 0)
        Image.fromarray(mask).save(dst / "mask_visib_inst" / name)
        # mask value = GT index + 1: recompute the visible pixel count and box per instance
        ids, counts = np.unique(mask, return_counts=True)
        count = dict(zip(ids.tolist(), counts.tolist()))
        for k, entry in enumerate(info[str(view)]):
            n = count.get(k + 1, 0)
            entry["px_count_visib"] = int(n)
            if n:
                vv, uu = np.nonzero(mask == k + 1)
                entry["bbox_visib"] = [int(uu.min()), int(vv.min()), int(uu.max() - uu.min() + 1),
                                       int(vv.max() - vv.min() + 1)]
            else:
                entry["bbox_visib"] = [-1, -1, -1, -1]
            entry["visib_fract"] = None
    with open(dst / "scene_gt_info.json", "w") as f:
        json.dump(info, f)
    return len(views)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="dataset root with shard_* dirs")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--scenes", nargs="*", default=None, help="scene ids to convert (default: all)")
    args = ap.parse_args()
    shards = sorted(args.src.glob("shard_*")) or [args.src]
    total = 0
    for shard in shards:
        split = "train_real" if (shard / "train_real").is_dir() else "train_pbr"
        scenes = [d for d in sorted((shard / split).iterdir()) if d.is_dir()
                  and (args.scenes is None or d.name in args.scenes)]
        if not scenes:
            continue
        cam = json.load(open(shard / "camera.json"))
        dist = cam.get("dist", [0.0])
        K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float64)
        out = args.dst / shard.name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("config_resolved.yaml", "run_info.json"):
            if (shard / f).exists():
                shutil.copyfile(shard / f, out / f)
        if not (out / "models").exists():
            shutil.copytree(shard / "models", out / "models")
        with open(out / "camera.json", "w") as f:
            json.dump({**cam, "dist": [0.0] * 5, "undistorted_from": {"src": str(shard), "dist": dist}}, f, indent=1)
        cache: dict = {}
        for sc in scenes:
            n = convert_scene(sc, out / split / sc.name, K, dist, cache)
            total += n
            print(f"{shard.name}/{sc.name}: {n} views (dist {dist})", flush=True)
    print(f"done: {total} views -> {args.dst}")


if __name__ == "__main__":
    main()
