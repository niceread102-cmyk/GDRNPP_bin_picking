"""Stage 3: depth refinement of a predicted pose by ICP against the CAD surface."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, binary_erosion


def observed_points(xyz: np.ndarray, valid: np.ndarray, mask: np.ndarray, erode_px: int) -> np.ndarray:
    m = mask & valid
    if erode_px > 0:
        m = binary_erosion(m, iterations=erode_px) & valid
    return xyz[m]


def depth_anchor(R: np.ndarray, t: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray,
                 model_pts: np.ndarray, model_normals: np.ndarray, min_px: int, iters: int = 3) -> np.ndarray:
    """Slide t along its viewing ray until the CAD's camera-facing surface matches the measured depth in the mask.

    The network never sees absolute depth (its depth input is relative to the part surface), so it infers t_z from
    apparent size; on real captures that reads parts ~1 % small (+7 mm). Scaling t keeps the 2D projection.
    Returns t unchanged if fewer than min_px projected surface points land on valid mask pixels.
    """
    h, w = depth.shape
    m = mask & (depth > 0)
    t = np.asarray(t, dtype=np.float64).copy()
    nrm_c = model_normals @ R.T
    for _ in range(iters):
        pts = model_pts @ R.T + t
        pts = pts[(nrm_c * pts).sum(1) < 0]                     # camera-facing surface
        uv = pts @ K.T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[ok], v[ok], pts[ok, 2]
        on = m[v, u]
        if on.sum() < min_px:
            break
        dz = float(np.median(depth[v[on], u[on]] - z[on]))
        t *= (t[2] + dz) / t[2]
    return t


def dedup_detections(results: list[dict], dist_mm: float) -> list[dict]:
    """Detections whose final centres are closer than dist_mm are the same part (duplicate or partial masks):
    keep the one with the best ICP fit (accepted first, then lowest fit error, then detector score)."""
    if dist_mm <= 0:
        return results
    order = sorted(range(len(results)), key=lambda i: (not results[i]["icp_refined"],
                                                       results[i]["icp_rmse"] if results[i]["icp_refined"] else 0.0,
                                                       -results[i]["score"]))
    kept: list[int] = []
    for i in order:
        t = np.asarray(results[i]["t"])
        if all(np.linalg.norm(t - np.asarray(results[k]["t"])) >= dist_mm for k in kept):
            kept.append(i)
    return [results[i] for i in sorted(kept)]


def mask_cover(R: np.ndarray, t: np.ndarray, mask: np.ndarray, K: np.ndarray, model_pts: np.ndarray,
               model_normals: np.ndarray, dilate_px: int = 2) -> float:
    """Share of the detection mask inside the CAD silhouette projected at the final pose. A real part explains its
    whole visible mask (real 0.91-1.0); a partial mask fitted to a fragment of a part, or a mask that also covers
    floor / a neighbour, leaves much of the mask outside the silhouette (0.2-0.6)."""
    h, w = mask.shape
    nrm_c = model_normals @ R.T
    pts = model_pts @ R.T + t
    uv = pts[(nrm_c * pts).sum(1) < 0] @ K.T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sil = np.zeros_like(mask)
    sil[v[ok], u[ok]] = True
    if dilate_px > 0:
        sil = binary_dilation(sil, iterations=dilate_px)
    return float((mask & sil).sum() / max(mask.sum(), 1))


def _overlap_px(a: np.ndarray, oa, b: np.ndarray, ob) -> int:
    """Pixels set in both masks; a, b are crops whose top-left corners sit at oa, ob (row, col) in the image."""
    y0, x0 = max(oa[0], ob[0]), max(oa[1], ob[1])
    y1, x1 = min(oa[0] + a.shape[0], ob[0] + b.shape[0]), min(oa[1] + a.shape[1], ob[1] + b.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    return int((a[y0 - oa[0]:y1 - oa[0], x0 - oa[1]:x1 - oa[1]] & b[y0 - ob[0]:y1 - ob[0], x0 - ob[1]:x1 - ob[1]]).sum())


def reject_unverified(results: list[dict], masks, rc, offsets=None) -> tuple[list[dict], list[dict]]:
    """Keep only detections whose pose is verified against the depth, so every output is a real part:
    ICP accepted with fitness >= verify_min_fitness and fit error <= verify_max_rmse_ratio x this frame's median
    (capped at verify_max_rmse_mm; the cap alone with < 3 accepted fits), the CAD silhouette explains
    >= verify_min_mask_cover of the mask, and the mask is not mostly (>= verify_max_mask_overlap) inside a larger
    kept mask (partial / duplicate). Looser than flag_confident, which then grades the kept ones.
    results[i]["mask_idx"] indexes masks (N,H,W) bool, or N mask crops with offsets (N,2) = (row, col) of each
    crop's top-left corner in the image. Returns (kept, rejected); rejected get a "reject" reason."""
    off = np.zeros((len(masks), 2), int) if offsets is None else np.asarray(offsets, int)
    rmse = [r["icp_rmse"] for r in results if r["icp_refined"]]
    limit = rc.verify_max_rmse_mm
    if len(rmse) >= 3:
        limit = min(limit, rc.verify_max_rmse_ratio * float(np.median(rmse)))
    kept, rejected = [], []
    for r in sorted(results, key=lambda r: -r["mask_px"]):                    # big masks first
        i = r["mask_idx"]
        if not r["icp_refined"]:
            r["reject"] = "icp_rejected" if r["n_depth_points"] >= rc.min_points else "no_depth"
        elif r["icp_fitness"] < rc.verify_min_fitness or r["icp_rmse"] > limit:
            r["reject"] = "poor_fit"
        elif r["mask_cover"] < rc.verify_min_mask_cover:
            r["reject"] = "mask_not_explained"
        elif any(_overlap_px(masks[i], off[i], masks[k["mask_idx"]], off[k["mask_idx"]])
                 >= rc.verify_max_mask_overlap * r["mask_px"] for k in kept):
            r["reject"] = "inside_other_mask"
        (rejected if "reject" in r else kept).append(r)
    order = {id(r): i for i, r in enumerate(results)}
    return sorted(kept, key=lambda r: order[id(r)]), sorted(rejected, key=lambda r: order[id(r)])


def icp_refine(R: np.ndarray, t: np.ndarray, scene_pts: np.ndarray, model_pts: np.ndarray,
               model_normals: np.ndarray, rc) -> dict:
    """Align observed (partial) surface points to the CAD surface placed at the initial pose.

    Target = model points facing the camera at the initial pose (the only ones a depth sensor can see),
    point-to-plane ICP, coarse to fine. ICP finds T with T @ scene ~= model(R, t); refined pose = T^-1 @ [R|t].
    Corrections larger than rc.max_rot_change_deg / rc.max_trans_change_mm are rejected.
    """
    T0 = np.eye(4)
    T0[:3, :3], T0[:3, 3] = R, t
    result = {"R": R, "t": t, "refined": False, "fitness": 0.0, "rmse": float("nan"), "n_points": len(scene_pts)}
    if len(scene_pts) < rc.min_points:
        return result
    nrm_c = model_normals @ R.T
    pts_c = model_pts @ R.T + t
    facing = (nrm_c * pts_c).sum(1) < 0                     # normal points back towards the camera
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_c[facing].astype(np.float64)))
    tgt.normals = o3d.utility.Vector3dVector(nrm_c[facing].astype(np.float64))
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts.astype(np.float64)))
    T = np.eye(4)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=rc.icp_iters)
    for dist in rc.icp_max_corr_mm:
        reg = o3d.pipelines.registration.registration_icp(src, tgt, dist, T, est, crit)
        T = reg.transformation
    T_new = np.linalg.inv(T) @ T0
    d_rot = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    d_trans = np.linalg.norm(T_new[:3, 3] - t)
    result.update(fitness=float(reg.fitness), rmse=float(reg.inlier_rmse))
    if reg.fitness > 0 and d_rot <= rc.max_rot_change_deg and d_trans <= rc.max_trans_change_mm:
        result.update(R=T_new[:3, :3], t=T_new[:3, 3], refined=True)
    elif rc.get("icp_retry_turns_deg"):
        _icp_retry(R, t, src, model_pts, model_normals, rc, result)
    return result


def _icp_retry(R, t, src, model_pts, model_normals, rc, result):
    """Second try when point-to-plane ICP failed: on a part seen mostly as its flat disc, point-to-plane can slide
    along the disc and lose every correspondence (fitness 0) although the start fits. Point-to-point ICP from the pose
    turned about the part axis by each of rc.icp_retry_turns_deg; the best fit is kept if the part axis moved at most
    rc.max_rot_change_deg and the centre at most rc.max_trans_change_mm (the turn about the axis itself is free: the
    disc is near-symmetric and the gripper goes into the round centre hole). Updates result in place."""
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=rc.icp_iters)
    best = None
    for turn in rc.icp_retry_turns_deg:
        a = np.radians(turn)
        Rs = R @ np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        nrm_c = model_normals @ Rs.T
        pts_c = model_pts @ Rs.T + t
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_c[(nrm_c * pts_c).sum(1) < 0].astype(np.float64)))
        T = np.eye(4)
        for dist in rc.icp_max_corr_mm:
            reg = o3d.pipelines.registration.registration_icp(src, tgt, dist, T, est, crit)
            T = reg.transformation
        Ts = np.eye(4)
        Ts[:3, :3], Ts[:3, 3] = Rs, t
        T_new = np.linalg.inv(T) @ Ts
        axis_change = np.degrees(np.arccos(np.clip(R[:, 2] @ T_new[:3, 2], -1, 1)))
        if (reg.fitness > 0 and axis_change <= rc.max_rot_change_deg
                and np.linalg.norm(T_new[:3, 3] - t) <= rc.max_trans_change_mm
                and (best is None or (reg.fitness, -reg.inlier_rmse) > (best[0], -best[1]))):
            best = (float(reg.fitness), float(reg.inlier_rmse), T_new, turn)
    if best is not None:
        result.update(R=best[2][:3, :3], t=best[2][:3, 3], refined=True, fitness=best[0], rmse=best[1],
                      icp_retry_turn=best[3])


def flag_confident(results: list[dict], rc) -> None:
    """Label-free confidence per detection: ICP accepted, fitness >= min, and fit error (inlier RMSE) small
    RELATIVE to this frame's typical fit error, so the test adapts to the sensor's noise level.
    An absolute cap guards against frames where every fit is bad; with < 3 accepted fits only the cap applies."""
    rmse = [r["icp_rmse"] for r in results if r["icp_refined"]]
    limit = rc.confidence_max_rmse_mm
    if len(rmse) >= 3:
        limit = min(limit, rc.confidence_rmse_ratio * float(np.median(rmse)))
    for r in results:
        r["confident"] = bool(r["icp_refined"] and r["icp_fitness"] >= rc.confidence_min_fitness
                              and r["icp_rmse"] <= limit)
        r["confidence_rmse_limit_mm"] = limit
