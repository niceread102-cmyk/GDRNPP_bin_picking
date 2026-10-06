"""Load infer.py outputs and training logs into tables (used by notebooks/results_summary.ipynb)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bin_dataset import BinPoseDataset, backproject
from bin6d.pose_model import pose_errors
from bin6d.refine import flag_confident


def frame_index(frames: BinPoseDataset) -> dict:
    return {(f.scene, f.view): i for i, f in enumerate(frames.frames)}


def load_frame(frames: BinPoseDataset, index: int) -> dict:
    meta = frames.frame_meta(index)
    rgb, depth, inst = frames.load_images(index, meta["depth_scale"])
    return {**meta, "rgb": rgb, "depth": depth, "inst": inst, "xyz": backproject(depth, meta["K"]),
            "valid": depth > 0}


def load_run(run_dir: str | Path, rc=None) -> dict:
    """{(scene, view): [detections]} from infer.py JSONs; re-flags confidence with `rc` if given."""
    out = {}
    for f in sorted(Path(run_dir).glob("*_*.json")):
        scene, view = f.stem.split("_")
        dets = json.load(open(f))
        if rc is not None:
            flag_confident(dets, rc)
        out[(scene, int(view))] = dets
    return out


def _tilt_deg(Ra, Rb) -> float:
    """Angle between the parts' z axes (plate normal), invariant to the 180-degree z symmetry."""
    return float(np.degrees(np.arccos(np.clip(abs(np.asarray(Ra)[:, 2] @ np.asarray(Rb)[:, 2]), -1, 1))))


def detections_table(run: dict, frames: BinPoseDataset, model) -> pd.DataFrame:
    """One row per detection with network-only and final (after ICP) errors vs GT."""
    idx = frame_index(frames)
    P = torch.from_numpy(model.points)
    SR, ST = torch.from_numpy(model.sym_R), torch.from_numpy(model.sym_t)
    rows = []
    for (scene, view), dets in run.items():
        objs = frames.frame_meta(idx[(scene, view)])["objects_all"]
        gid = {int(i): k for k, i in enumerate(objs["inst_id"])}
        for n, d in enumerate(dets):
            row = {"scene": scene, "view": view, "det": n, "score": d["score"], "confident": d["confident"],
                   "icp_refined": d["icp_refined"], "icp_fitness": d["icp_fitness"], "icp_rmse": d["icp_rmse"],
                   "n_depth_points": d["n_depth_points"], "matched": d["gt_inst_id"] is not None}
            if row["matched"]:
                k = gid[d["gt_inst_id"]]
                Rg, tg = torch.from_numpy(objs["R"][k][None]).float(), torch.from_numpy(objs["t"][k][None]).float()
                row["visib_px"] = int(objs["visib_px"][k])
                for tag, R, t in (("net", d["R_rgbd"], d["t_rgbd"]), ("final", d["R"], d["t"])):
                    e = pose_errors(torch.tensor([R]).float(), torch.tensor([t]).float(), Rg, tg, P, SR, ST)
                    row[f"{tag}_add_s"] = float(e["add_s"])
                    row[f"{tag}_centre_mm"] = float(e["trans_mm"])
                    row[f"{tag}_tilt_deg"] = _tilt_deg(R, objs["R"][k])
            rows.append(row)
    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame, thresholds_mm=(0.5, 1.0, 2.0), diameter: float | None = None) -> pd.DataFrame:
    """Network only / + ICP (all) / + ICP (confident only), over detections matched to GT."""
    m = df[df.matched]
    ths = list(thresholds_mm) + ([0.1 * diameter] if diameter else [])
    rows = []
    for name, tag, sub in (("Network only", "net", m), ("+ ICP, all detections", "final", m),
                           ("+ ICP, confident only", "final", m[m.confident])):
        r = {"": name, "n": len(sub)}
        for th in ths:
            r[f"ADD-S < {th:.1f} mm"] = f"{(sub[f'{tag}_add_s'] < th).mean():.1%}"
        c, tl = sub[f"{tag}_centre_mm"], sub[f"{tag}_tilt_deg"]
        r["centre err median / p95 (mm)"] = f"{c.median():.2f} / {c.quantile(0.95):.1f}"
        r["tilt median / p95 (deg)"] = f"{tl.median():.2f} / {tl.quantile(0.95):.1f}"
        r["wrong (>= 1 mm)"] = int((sub[f"{tag}_add_s"] >= 1).sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("")


def per_frame_table(df: pd.DataFrame, frames: BinPoseDataset) -> pd.DataFrame:
    idx = frame_index(frames)
    rows = []
    for (scene, view), g in df.groupby(["scene", "view"]):
        c = g[g.confident & g.matched]
        rows.append({"view": view, "parts in bin": len(frames.frame_meta(idx[(scene, view)])["objects_all"]["t"]),
                     "detections": len(g), "confident": int(g.confident.sum()),
                     "confident correct (<1 mm)": int((c.final_add_s < 1).sum()),
                     "confident wrong": int((c.final_add_s >= 1).sum()),
                     "median fit error (mm)": round(float(g[g.icp_refined].icp_rmse.median()), 3)})
    return pd.DataFrame(rows).set_index("view")


_VAL_RE = re.compile(r"== val ep (\d+): (.*)")


def parse_val_log(path: str | Path) -> pd.DataFrame:
    """Validation lines of train_detector.py / train_pose.py logs -> one row per evaluated epoch."""
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = _VAL_RE.search(line)
        if m:
            row = {"epoch": int(m.group(1)) + 1}
            for kv in m.group(2).split():
                k, v = kv.split("=")
                row[k] = float(v)
            rows.append(row)
    return pd.DataFrame(rows).set_index("epoch")
