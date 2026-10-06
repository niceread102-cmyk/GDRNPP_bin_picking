"""Stage 3: depth refinement of a predicted pose by ICP against the CAD surface."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion


def observed_points(xyz: np.ndarray, valid: np.ndarray, mask: np.ndarray, erode_px: int) -> np.ndarray:
    m = mask & valid
    if erode_px > 0:
        m = binary_erosion(m, iterations=erode_px) & valid
    return xyz[m]


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
    return result


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
