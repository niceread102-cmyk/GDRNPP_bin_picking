"""Load infer.py outputs and training logs into tables (used by notebooks/results_summary.ipynb)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_conversion.bin_dataset import BinPoseDataset, backproject
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
                stages = [("net", d["R_rgbd"], d["t_rgbd"]), ("final", d["R"], d["t"])]
                if "t_anchored" in d:                    # runs written before depth anchoring lack it
                    stages.insert(1, ("anchor", d["R_rgbd"], d["t_anchored"]))
                for tag, R, t in stages:
                    e = pose_errors(torch.tensor([R]).float(), torch.tensor([t]).float(), Rg, tg, P, SR, ST)
                    row[f"{tag}_add_s"] = float(e["add_s"])
                    row[f"{tag}_centre_mm"] = float(e["trans_mm"])
                    row[f"{tag}_tilt_deg"] = _tilt_deg(R, objs["R"][k])
            rows.append(row)
    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame, thresholds_mm=(0.5, 1.0, 2.0), diameter: float | None = None) -> pd.DataFrame:
    """Network only / + depth anchor (if run with it) / + ICP (all) / + ICP (confident only), over matched
    detections."""
    m = df[df.matched]
    ths = list(thresholds_mm) + ([0.1 * diameter] if diameter else [])
    rows = []
    stages = [("Network only", "net", m), ("+ ICP, all detections", "final", m),
              ("+ ICP, confident only", "final", m[m.confident])]
    if "anchor_add_s" in m and m.anchor_add_s.notna().all():
        stages.insert(1, ("+ depth anchor (before ICP)", "anchor", m))
    for name, tag, sub in stages:
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


# ---------------------------------------------------------------- label-free checks (captures without GT)
def surface_check(R, t, depth: np.ndarray, K: np.ndarray, model, tol_mm: float = 3.0) -> dict:
    """Compare the CAD's camera-facing surface at pose (R, t) with the measured depth at the projected pixels.

    support    measured depth within tol of the CAD surface
    violation  measured depth BEHIND the CAD surface by > tol: the sensor sees through where the part should be
               (free space) -> the pose cannot be right there. Self-occlusion only ever reads as `occluded`.
    occluded   measured depth in FRONT by > tol (another part / own geometry in front): neutral
    Fractions are over projected points with valid depth; `coverage` = share of projected points with depth.
    """
    R, t = np.asarray(R, np.float64), np.asarray(t, np.float64)
    pts = model.dense_points @ R.T + t
    pts = pts[((model.dense_normals @ R.T) * pts).sum(1) < 0]
    uv = pts @ np.asarray(K, np.float64).T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    h, w = depth.shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    dm, z = depth[v[ok], u[ok]], pts[ok, 2]
    has = dm > 0
    diff = dm[has] - z[has]
    n = max(int(has.sum()), 1)
    return {"support": float((np.abs(diff) <= tol_mm).sum() / n), "violation": float((diff > tol_mm).sum() / n),
            "occluded": float((diff < -tol_mm).sum() / n), "coverage": float(has.mean()) if len(has) else 0.0}


def _rot_change_deg(Ra, Rb) -> float:
    Rr = np.asarray(Ra).T @ np.asarray(Rb)
    return float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))


def gt_free_table(run: dict, frames: BinPoseDataset, model, tol_mm: float = 3.0) -> pd.DataFrame:
    """One row per detection with label-free quality signals (no GT needed):
    ICP fit (fitness, rmse), how far depth anchoring / ICP moved the network pose, and the CAD-vs-depth
    surface check at the final pose. If the frames have GT, `matched` and `final_add_s` are added for calibration."""
    idx = frame_index(frames)
    has_gt = any(len(frames.frame_meta(i)["objects_all"]["t"]) for i in range(len(frames)))
    gt = detections_table(run, frames, model).set_index(["scene", "view", "det"]) if has_gt else None
    rows = []
    for (scene, view), dets in run.items():
        f = load_frame(frames, idx[(scene, view)])
        for n, d in enumerate(dets):
            t_net, t_anc = np.asarray(d["t_rgbd"]), np.asarray(d.get("t_anchored", d["t_rgbd"]))
            row = {"scene": scene, "view": view, "det": n, "score": d["score"], "confident": d["confident"],
                   "icp_refined": d["icp_refined"], "icp_fitness": d["icp_fitness"], "icp_rmse": d["icp_rmse"],
                   "n_depth_points": d["n_depth_points"],
                   "anchor_dz_mm": float(t_anc[2] - t_net[2]),                  # network depth error proxy
                   "icp_shift_mm": float(np.linalg.norm(np.asarray(d["t"]) - t_anc)),
                   "icp_rot_deg": _rot_change_deg(d["R_rgbd"], d["R"]),
                   "distance_mm": float(np.asarray(d["t"])[2])}
            row.update({f"surf_{k}": v for k, v in surface_check(d["R"], d["t"], f["depth"], f["K"], model,
                                                                 tol_mm).items()})
            if gt is not None:
                g = gt.loc[(scene, view, n)]
                row["matched"] = bool(g.matched)
                row["final_add_s"] = float(g.final_add_s) if g.matched else np.nan
            rows.append(row)
    cols = ["scene", "view", "det", "score", "confident", "icp_refined", "icp_fitness", "icp_rmse", "n_depth_points",
            "anchor_dz_mm", "icp_shift_mm", "icp_rot_deg", "distance_mm", "surf_support", "surf_violation",
            "surf_occluded", "surf_coverage"] + (["matched", "final_add_s"] if gt is not None else [])
    return pd.DataFrame(rows, columns=cols)


def static_groups(frames: BinPoseDataset, tol_mm: float = 3.0, max_changed: float = 0.005) -> list[list[int]]:
    """Consecutive views of an unchanged scene (repeated captures): at most max_changed of the pixels valid in both
    neighbouring depth maps differ by > tol_mm. Photoneo, same scene: ~0.01 %; one part moved: >= 2.5 % (parts
    cover only a few % of the image, so a share-that-agrees test cannot tell them apart).
    Returns lists of views; singletons are scenes captured once."""
    groups, prev = [], None
    for i, fr in enumerate(frames.frames):
        meta = frames.frame_meta(i)
        depth = frames.load_images(i, meta["depth_scale"])[1]
        same = False
        if prev is not None and prev.shape == depth.shape:
            both = (prev > 0) & (depth > 0)
            same = both.any() and float((np.abs(prev[both] - depth[both]) > tol_mm).mean()) <= max_changed
        if same:
            groups[-1].append(fr.view)
        else:
            groups.append([fr.view])
        prev = depth
    return groups


def repeatability_table(run: dict, scene: str, groups: list[list[int]], match_mm: float = 10.0) -> pd.DataFrame:
    """Pose spread of the same part over repeated captures of an unchanged scene (precision, not accuracy).
    Confident detections of the group's first view are tracked through the other views by centre distance;
    rotation spread is symmetry-aware for the 2-fold z symmetry (tilt of the z axis + in-plane angle mod 180)."""
    rows = []
    for g in (g for g in groups if len(g) >= 2):
        ref = [d for d in run.get((scene, g[0]), []) if d["confident"]]
        for k, d0 in enumerate(ref):
            t0, R0 = np.asarray(d0["t"]), np.asarray(d0["R"])
            ts, tilts, rolls = [t0], [], []
            for v in g[1:]:
                cand = [d for d in run.get((scene, v), []) if d["confident"]]
                if not cand:
                    continue
                d = min(cand, key=lambda c: np.linalg.norm(np.asarray(c["t"]) - t0))
                if np.linalg.norm(np.asarray(d["t"]) - t0) > match_mm:
                    continue
                R = np.asarray(d["R"])
                ts.append(np.asarray(d["t"]))
                tilts.append(_tilt_deg(R, R0))
                Rr = R0.T @ R
                a = np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])) % 180
                rolls.append(min(a, 180 - a))
            if len(ts) < 2:
                continue
            ts = np.stack(ts)
            dev = np.linalg.norm(ts - ts.mean(0), axis=1)
            rows.append({"group": f"{g[0]}-{g[-1]}", "part": k, "seen in": f"{len(ts)}/{len(g)}",
                         "centre spread max (mm)": float(dev.max()), "z spread (mm)": float(np.ptp(ts[:, 2])),
                         "tilt max (deg)": float(max(tilts)), "in-plane rot max (deg)": float(max(rolls))})
    return pd.DataFrame(rows)


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


def explained_depth(dets: list[dict], depth: np.ndarray, K: np.ndarray, model, tol_mm: float = 3.0,
                    n_points: int = 200000, roi_pad_px: int = 0) -> dict:
    """Label-free coverage of a pile: render every detection's CAD at its final pose into one z-buffer and mark the
    measured depth pixels it explains (|depth - CAD| <= tol). Depth inside the ROI (box spanned by the projected
    detection centres, padded by roi_pad_px; keeps the bin walls out) that no pose explains is a part the pipeline
    missed, or a gap between parts. Returns {"explained": (H,W) bool, "roi": (H,W) bool, "share": float}."""
    from bin6d.geometry import sample_surface
    if getattr(model, "_render_pts", None) is None:
        model._render_pts = sample_surface(model.verts, model.faces, n_points, seed=4, return_normals=True)
    pts0, nrm0 = model._render_pts
    h, w = depth.shape
    K = np.asarray(K, np.float64)
    zbuf = np.full(h * w, np.inf)
    centres = []
    for d in dets:
        R, t = np.asarray(d["R"], np.float64), np.asarray(d["t"], np.float64)
        pts = pts0 @ R.T + t
        pts = pts[((nrm0 @ R.T) * pts).sum(1) < 0]
        uv = pts @ K.T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        np.minimum.at(zbuf, v[ok] * w + u[ok], pts[ok, 2])
        c = K @ t
        centres.append(c[:2] / c[2])
    zbuf = zbuf.reshape(h, w)
    explained = (depth > 0) & np.isfinite(zbuf) & (np.abs(depth - zbuf) <= tol_mm)
    roi = np.zeros((h, w), bool)
    if centres:
        c = np.array(centres)
        x0, y0 = np.maximum(c.min(0).astype(int) - roi_pad_px, 0)
        x1, y1 = c.max(0).astype(int) + roi_pad_px
        roi[y0:y1, x0:x1] = True
    roi &= depth > 0
    return {"explained": explained, "roi": roi, "share": float((explained & roi).sum() / max(roi.sum(), 1))}
