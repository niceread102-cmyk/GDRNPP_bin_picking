"""Loader for the Isaac Sim bin-picking dataset (BOP layout).

Layout (per shard):
    shard_XX/
        camera.json, config_resolved.yaml
        models/obj_000001.ply, models_info.json          # CAD model in mm
        train_pbr/<scene>/
            rgb/<view>.png                                # uint8 RGB, 1236x1032
            depth/<view>.png                              # uint16, * depth_scale -> mm (0 = invalid)
            mask_visib_inst/<view>.png                    # uint16, value = gt_idx + 1 (0 = background)
            poses/<view>.csv                              # same poses as scene_gt + euler/bin-frame extras
            scene_gt.json, scene_gt_info.json, scene_camera.json

Every part in the bin has a GT pose, including fully occluded ones (px_count_visib == 0).
Parts that the sim "picked" are NOT deleted from scene_gt: they are parked on the floor 1.2-1.8 m
outside the bin (world z = 10 mm). The loader drops them using the bin footprint from the config.
There is no stored point cloud; it is back-projected from depth + K (camera frame, mm).

Real captures (e.g. data/pseudo_gt_20_bop) use the same layout with `train_real/` instead of `train_pbr/`, no
config_resolved.yaml (no bin filter) and no cam_*_w2c (t_world = camera frame).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Frame:
    shard: Path
    scene: str
    view: int


_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
              "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
              "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


def read_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian PLY -> (vertices (V,3) float32, triangles (F,3) int64)."""
    with open(path, "rb") as f:
        elements, cur = [], None
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            tok = line.split()
            if tok[0] == "format" and tok[1] != "binary_little_endian":
                raise ValueError(f"unsupported PLY format {tok[1]}")
            if tok[0] == "element":
                cur = {"name": tok[1], "count": int(tok[2]), "props": []}
                elements.append(cur)
            elif tok[0] == "property":
                if tok[1] == "list":
                    # assume fixed-size triangle lists (count, 3 indices)
                    cur["props"] += [("_n", _PLY_TYPES[tok[2]]), (tok[4], _PLY_TYPES[tok[3]], (3,))]
                else:
                    cur["props"].append((tok[2], _PLY_TYPES[tok[1]]))
        data = {e["name"]: np.frombuffer(f.read(e["count"] * np.dtype(e["props"]).itemsize),
                                         dtype=e["props"], count=e["count"]) for e in elements}
    v = data["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    face = data["face"]
    if not np.all(face["_n"] == 3):
        raise ValueError("only triangle meshes are supported")
    name = [n for n in face.dtype.names if n != "_n" and face.dtype[n].shape == (3,)][0]
    return verts, face[name].astype(np.int64)


def read_ply_vertices(path: str | Path) -> np.ndarray:
    return read_ply_mesh(path)[0]


def read_bin_inner_size_mm(config_path: Path) -> np.ndarray:
    """bin.inner_size_m from config_resolved.yaml (x, y, z) in mm, parsed without PyYAML."""
    text = config_path.read_text()
    m = re.search(r"^bin:\n(?:[ ].*\n)*?  inner_size_m:\n((?:  - .*\n){3})", text, re.M)
    if m is None:
        raise ValueError(f"bin.inner_size_m not found in {config_path}")
    return np.array([float(x.split("-", 1)[1]) for x in m.group(1).splitlines()]) * 1000.0


def backproject(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Organized point cloud (H, W, 3) in camera frame, mm. Invalid pixels are 0."""
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_mm.astype(np.float32)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


class BinPoseDataset(Dataset):
    """One item = one (scene, view) frame.

    Returns a dict with:
        rgb        (H, W, 3) uint8
        depth      (H, W)    float32, mm, 0 = invalid
        xyz        (H, W, 3) float32, camera-frame points in mm (organized cloud)
        valid      (H, W)    bool, depth > 0
        inst_mask  (H, W)    int32, gt_idx + 1 for visible pixels, 0 = background
        K          (3, 3)    float32
        cam_R_w2c, cam_t_w2c world(bin) -> camera extrinsics (t in mm)
        objects_all  dict of per-part arrays over every part currently IN the bin (N parts),
                     visible or fully occluded; picked/parked parts are excluded:
            obj_id (N,), R (N,3,3), t (N,3) camera frame mm, t_world (N,3) bin frame mm,
            visib_px (N,), valid_px (N,) visible px with valid depth, visible (N,) bool,
            bbox_visib (N,4) xywh (-1 if hidden), bbox_obj (N,4),
            center_2d (N,2) projected model origin in pixels, inst_id (N,) value in inst_mask
        objects      same keys, filtered to visib_px >= min_visib_px (== objects_all if 0)
        scene, view, index

    Args:
        root: dataset root containing shard_* dirs (or a single shard dir).
        scenes: optional subset of scene ids (e.g. ["000000", ...]) for train/val splits.
        min_visib_px: `objects` keeps parts with at least this many visible pixels. 0 (default)
            keeps hidden parts too; 1 = visible only; ~200 = reasonably visible training targets.
            `objects_all` always has every in-bin part.
        bin_margin_mm: slack around the bin inner footprint when deciding a part is in the bin.
        n_points: if set, also return `points` (n_points, 6: xyz mm + rgb 0..1) and
            `point_inst` (n_points,) sampled from valid depth pixels.
        to_tensor: convert arrays to torch tensors (rgb -> float CHW in 0..1).
    """

    def __init__(
        self,
        root: str | Path,
        scenes: list[str] | None = None,
        min_visib_px: int = 0,
        n_points: int | None = None,
        to_tensor: bool = False,
        seed: int = 0,
        bin_margin_mm: float = 50.0,
    ):
        root = Path(root)
        self.shards = sorted(root.glob("shard_*")) or [root]
        self.min_visib_px = min_visib_px
        self.n_points = n_points
        self.to_tensor = to_tensor
        self.rng = np.random.default_rng(seed)
        bin_cfg = self.shards[0] / "config_resolved.yaml"
        self.bin_half_xy = read_bin_inner_size_mm(bin_cfg)[:2] / 2 + bin_margin_mm if bin_cfg.exists() else None
        self.split = "train_real" if (self.shards[0] / "train_real").is_dir() else "train_pbr"

        self.frames: list[Frame] = []
        for shard in self.shards:
            for scene_dir in sorted((shard / self.split).iterdir()):
                if not scene_dir.is_dir() or (scenes is not None and scene_dir.name not in scenes):
                    continue
                for p in sorted((scene_dir / "rgb").glob("*.png")):
                    self.frames.append(Frame(shard, scene_dir.name, int(p.stem)))

        # models / symmetries (assumes all shards share one model set)
        models_dir = self.shards[0] / "models"
        self.models_info = {int(k): v for k, v in json.load(open(models_dir / "models_info.json")).items()}
        self._model_points: dict[int, np.ndarray] = {}
        self._scene_cache: dict[tuple[Path, str], dict] = {}

    # ---------------------------------------------------------------- helpers
    def model_points(self, obj_id: int = 1) -> np.ndarray:
        """CAD model vertices (M, 3) in mm, model frame."""
        if obj_id not in self._model_points:
            path = self.shards[0] / "models" / f"obj_{obj_id:06d}.ply"
            self._model_points[obj_id] = read_ply_vertices(path)
        return self._model_points[obj_id]

    def symmetries(self, obj_id: int = 1) -> list[np.ndarray]:
        """Discrete symmetry transforms (4x4, translation in mm), identity included."""
        syms = [np.eye(4)]
        for s in self.models_info[obj_id].get("symmetries_discrete", []):
            syms.append(np.asarray(s, dtype=np.float64).reshape(4, 4))
        return syms

    def _scene(self, shard: Path, scene: str) -> dict:
        key = (shard, scene)
        if key not in self._scene_cache:
            d = shard / self.split / scene
            self._scene_cache[key] = {
                "gt": json.load(open(d / "scene_gt.json")),
                "info": json.load(open(d / "scene_gt_info.json")),
                "cam": json.load(open(d / "scene_camera.json")),
            }
        return self._scene_cache[key]

    # ---------------------------------------------------------------- dataset
    def __len__(self) -> int:
        return len(self.frames)

    def frame_meta(self, index: int) -> dict:
        """Camera + labels for a frame without touching the images (cheap)."""
        fr = self.frames[index]
        meta = self._scene(fr.shard, fr.scene)
        v = str(fr.view)
        cam = meta["cam"][v]
        K = np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3)

        gts, infos = meta["gt"][v], meta["info"][v]
        R = np.asarray([g["cam_R_m2c"] for g in gts], dtype=np.float32).reshape(-1, 3, 3)
        t = np.asarray([g["cam_t_m2c"] for g in gts], dtype=np.float32).reshape(-1, 3)
        proj = t @ K.T
        R_w2c = np.asarray(cam.get("cam_R_w2c", np.eye(3).ravel()), dtype=np.float32).reshape(3, 3)
        t_w2c = np.asarray(cam.get("cam_t_w2c", np.zeros(3)), dtype=np.float32)
        t_world = (t - t_w2c) @ R_w2c  # R_w2c^T (t - t_w2c) per row
        objects_all = {
            "obj_id": np.asarray([g["obj_id"] for g in gts], dtype=np.int64),
            "R": R,
            "t": t,
            "t_world": t_world,
            "visib_px": np.asarray([i["px_count_visib"] for i in infos], dtype=np.int64),
            "valid_px": np.asarray([i["px_count_valid"] for i in infos], dtype=np.int64),
            "bbox_visib": np.asarray([i["bbox_visib"] for i in infos], dtype=np.float32).reshape(-1, 4),
            "bbox_obj": np.asarray([i["bbox_obj"] for i in infos], dtype=np.float32).reshape(-1, 4),
            "center_2d": proj[:, :2] / proj[:, 2:3],
            "inst_id": np.arange(1, len(gts) + 1, dtype=np.int64),  # value in inst_mask
        }
        objects_all["visible"] = objects_all["visib_px"] > 0

        in_bin = (np.all(np.abs(t_world[:, :2]) <= self.bin_half_xy, axis=1) if self.bin_half_xy is not None
                  else np.ones(len(t), dtype=bool))
        objects_all = {k: a[in_bin] for k, a in objects_all.items()}
        keep = objects_all["visib_px"] >= self.min_visib_px
        return {
            "K": K,
            "depth_scale": float(cam["depth_scale"]),
            "cam_R_w2c": R_w2c,
            "cam_t_w2c": t_w2c,
            "objects": {k: a[keep] for k, a in objects_all.items()},
            "objects_all": objects_all,
            "scene": fr.scene,
            "view": fr.view,
            "index": index,
        }

    def load_images(self, index: int, depth_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rgb (H,W,3) uint8, depth (H,W) float32 mm, inst_mask (H,W) int32)."""
        fr = self.frames[index]
        d = fr.shard / self.split / fr.scene
        name = f"{fr.view:06d}.png"
        rgb = np.array(Image.open(d / "rgb" / name).convert("RGB"))
        depth = np.asarray(Image.open(d / "depth" / name), dtype=np.float32) * depth_scale
        inst_mask = np.array(Image.open(d / "mask_visib_inst" / name), dtype=np.int32)
        return rgb, depth, inst_mask

    def __getitem__(self, index: int) -> dict:
        meta = self.frame_meta(index)
        rgb, depth, inst_mask = self.load_images(index, meta.pop("depth_scale"))
        K = meta["K"]
        xyz = backproject(depth, K)
        valid = depth > 0
        sample = {"rgb": rgb, "depth": depth, "xyz": xyz, "valid": valid, "inst_mask": inst_mask, **meta}

        if self.n_points:
            idx = np.flatnonzero(valid)
            idx = self.rng.choice(idx, self.n_points, replace=len(idx) < self.n_points)
            sample["points"] = np.concatenate(
                [xyz.reshape(-1, 3)[idx], rgb.reshape(-1, 3)[idx].astype(np.float32) / 255.0], axis=1
            )
            sample["point_inst"] = inst_mask.reshape(-1)[idx]

        return self._tensorize(sample) if self.to_tensor else sample

    @staticmethod
    def _tensorize(s: dict) -> dict:
        out = {}
        for k, a in s.items():
            if isinstance(a, dict):
                out[k] = {kk: torch.from_numpy(np.ascontiguousarray(aa)) for kk, aa in a.items()}
            elif isinstance(a, np.ndarray):
                out[k] = torch.from_numpy(np.ascontiguousarray(a))
            else:
                out[k] = a
        out["rgb"] = out["rgb"].permute(2, 0, 1).float().div_(255.0)
        return out


def collate(batch: list[dict]) -> dict:
    """Stack fixed-size fields; keep per-frame object dicts as a list (variable part count)."""
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k in ("objects", "objects_all") or not isinstance(vals[0], torch.Tensor):
            out[k] = vals
        else:
            out[k] = torch.stack(vals)
    return out


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "bin_ds_side_plate_bp_sample"
    ds = BinPoseDataset(root, n_points=20000)
    print(f"frames: {len(ds)}  shards: {[s.name for s in ds.shards]}")
    s = ds[0]
    o = s["objects_all"]
    print(f"scene {s['scene']} view {s['view']}: rgb {s['rgb'].shape}, depth valid {s['valid'].mean():.2%}")
    print(f"parts in bin: {len(o['t'])}, visible: {o['visible'].sum()}, "
          f"mask ids: {len(np.unique(s['inst_mask'])) - 1}")
    print(f"points: {s['points'].shape}, model verts: {ds.model_points().shape}, symmetries: {len(ds.symmetries())}")

    # sanity: model points transformed by GT should land on observed depth for a well-visible part
    i = int(np.argmax(o["valid_px"]))
    pts = ds.model_points() @ o["R"][i].T + o["t"][i]
    uv = pts @ s["K"].T
    uv = (uv[:, :2] / uv[:, 2:3]).round().astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < s["depth"].shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < s["depth"].shape[0])
    obs = s["depth"][uv[ok, 1], uv[ok, 0]]
    on_part = (s["inst_mask"][uv[ok, 1], uv[ok, 0]] == o["inst_id"][i]) & (obs > 0)
    front = pts[ok][on_part, 2] - obs[on_part]
    print(f"part {i}: model-vs-depth z residual on visible pixels: min {front.min():.1f} mm, "
          f"median of front-facing |res| {np.median(np.abs(front[front < 5])):.2f} mm")
