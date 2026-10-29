#!/usr/bin/env python3
"""Real bin picking loop with a window: Photoneo 3D scan -> side plate 6D pose -> picks -> Lite6, over and over.

Script form of notebooks/results_6dpose_real_optimize.ipynb: same models, pick rules, TCP targets, 2D correction and
robot hand-off (auto_pick_ee6d.py -> ee_6d.py -> /target_pose -> sequence_real_action). After every pick a 2D image
corrects / drops the remaining picks; when none is left a new 3D scan follows, and a scan that finds nothing graspable
is repeated every --idle seconds, so the loop keeps picking while parts are put into the bin.

Start terminal 1 (MoveIt + robot) and terminal 2 (sequence_real_action) first (gdrnpp_binpicking_real.txt) and shut
down the notebook kernel: only one process can hold the scanner.

    python binpick/binpick_real.py                     # camera + robot, press Start in the window
    python binpick/binpick_real.py --no-robot          # camera only: scan, pose and picks, nothing is sent to the robot
    python binpick/binpick_real.py --offline data/pseudo_gt_70_bop 59 --no-robot     # saved frame, no camera, no robot
    python binpick/binpick_real.py --no-ui --autostart # console only
"""
import argparse
import csv
import json
import os
import queue as queue_mod
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # project root (this file is in binpick/)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation
from torchvision.ops import masks_to_boxes

from data_conversion.bin_dataset import BinPoseDataset, backproject
from bin6d.config import Cfg, load_config
from bin6d.detector import build_detector, set_native_scale
from bin6d.pose_data import ObjectModel, frame_planes, input_channels
from bin6d.pose_model import GDRNet
from bin6d.refine import (dedup_detections, depth_anchor, flag_confident, icp_refine, mask_cover, observed_points,
                          reject_unverified)
from bin6d.report import load_frame, frame_index, surface_check
from bin6d.viz import render_2d
from data_conversion.camera_convert_data import undistort, undistort_lookup
from infer import predict_poses

# camera
HARVESTERS_PY = Path.home() / "Projects/photoneo-python-examples/GigE-V/harvesters/.venv/bin/python"
SERIAL = None                                 # Photoneo serial number; None = first device found
CAPTURE_DIR = Path("runs/realtime/capture")   # raw captures (camera_convert_data.py layout)
UNDISTORT = True                              # resample rgb/depth onto the pinhole K (same as --to-bop)
OFFLINE = None                                # (bop_root, view) = use a saved frame instead of the camera (--offline)

# models: best setting of results_photoneo_summary.ipynb (§6-§9)
CONFIG = "configs/default.yaml"
DETECTOR = "runs/detector_ft/best.pt"         # sim + real fine-tuned detector
POSE = "runs/pose/best.pt"                    # sim-trained pose (fine-tuned pose: same final result, §8)
MODEL_ROOT = "data/pseudo_gt_70_bop"          # CAD model + symmetries (same CAD as training)
DET_SCALE = 1.0                               # detector input scale; 1.0 for the 1236x1032 primary camera
ROI_PAD = 8                                   # px around each mask for depth refinement (> mask_erode_px, mask_cover dilation)
# ~35 % of the detector's masks lose ICP (point-to-plane slides along the flat disc, fitness 0) although the network
# pose fits: retry point-to-point from these turns about the part axis (refine.icp_retry_turns_deg; () = off)
ICP_RETRY_TURNS = (0, 60, 120, 180, 240, 300)

# pick selection (summary §10)
TOP_K = 15              # picks chosen per 3D scan (the notebook: 10); auto_pick.py --per-scan overrides it
R_FINGER = 5.0          # mm, radius of the closed fingers (hole: 11.2-13.8 mm diameter)
INSERT_DEPTH = 8.0      # mm, finger insertion depth
R_BODY = 15.0           # mm, radius of the gripper body / tip that must clear the surroundings
RING_MARGIN = 3.0       # mm, surroundings may stick up this much above the hole entrance (noise, calibration)
APPROACH_LEN = 60.0     # mm, height above the entrance checked for obstacles
HOLE_BLOCK_MAX = 0.10   # allowed blocked share of the finger disc
RING_BLOCK_MAX = 0.10   # allowed blocked share of the body ring
GRASP_MAX_OCCLUDED = 0.5
# share of a pick's image silhouette that may overlap a pick chosen before it in the same scan (the notebook: 0.02).
# Overlapping picks are fine: after each pick the 2D check corrects or drops the ones that moved. At 0.20 every
# graspable part of today's captures is chosen (0.02 dropped up to 2 per scan); auto_pick.py --max-overlap
MAX_OVERLAP_SELECTED = 0.20
OUT_DIR = Path("runs/realtime")

# --- robot: hand-eye + TCP target ---
# hand-eye calibration, camera -> robot base: p_base = T_CAM_BASE @ p_cam (metres)
# ~/Projects/xarm_lite6/cam2robot_pos_calibration/result/T_base_cam.yaml (touch/board_model, 2026-10-27 16:45,
# fit RMS 1.1 mm); copy the new matrix here after every recalibration
T_CAM_BASE = np.array([
    [-0.00169203, -0.99945118,  0.03308287,  0.25562417],
    [-0.97936149,  0.00834258,  0.20194425, -0.25696641],
    [-0.20210942, -0.03205840, -0.97883811,  0.71477754],
    [ 0.0,         0.0,         0.0,         1.0],
])
T_BASE_CAM = T_CAM_BASE.copy()
T_BASE_CAM[:3, 3] *= 1000.0                                          # the pipeline works in mm
FRAME, T_OUT = "base", T_BASE_CAM
UP_CAM = T_OUT[2, :3]                                                # base +Z in the camera frame (tilt reference)
R_TOOL_DOWN = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()   # tool z straight down
# robot safety box for /target_pose (lite6_control config/default.yaml `safety`): TCP outside it is rejected there
SAFE_X, SAFE_Y, SAFE_Z, SAFE_REACH = (0.05, 0.45), (-0.45, 0.45), (0.01, 0.60), 0.44
# tilted parts are picked too: the tool goes in along the part axis, pointing down, whichever end is up
MAX_TILT = 60.0         # deg, part axis vs base +Z; parts tilted more are skipped ("tilted"); 90 = no limit
# turn of the tool about its own axis: "down" = the tool-down pose R_TOOL_DOWN turned as little as possible to put the
# tool z on the part axis (no spin: flat parts always get rpy 180 0 0; the fingers go into the round centre hole, so
# the part's own x does not matter) | "part" = tool x follows the part x (of +-x the one closer to tool-down)
GRIP_YAW = "down"
# blue bin: parts closer than BIN_WALL_MARGIN (horizontal, robot base) to the bin wall are skipped ("near bin wall").
# Wall = blue pixels (PIL HSV 0..255) higher than the part centre - BIN_WALL_BELOW: the rim and the inner wall faces
# that stick up around the part (blue at floor level outside the bin does not count). Checked on top of the gripper's
# real shape (TOOL_*, "gripper hits ..."); 0 = off (only the shape check, ~half of a full bin's parts more)
BIN_WALL_MARGIN = 0.08  # m (the notebook's rule was 0.10); auto_pick.py --wall-margin
# gripper + wrist camera vs the measured cloud (bin walls and other parts): the collision meshes of lite6_complete.urdf
# (gripper shell, D435 + stand, link6) as the largest radius around the tool axis per 5 mm of height and 10 deg of
# direction, placed at the grasp pose and at every 10 mm of the approach up to the pre-grasp pose
TOOL_MESH_DIR = Path("lite6_control/xarm_ros2/install/xarm_description/share/xarm_description/meshes")
TCP_OFFSET_M = 0.101    # robot.tcp_offset_m (lite6_control config/robots/lite6.yaml): link_eef -> TCP target, tool z
APPROACH_M = 0.04       # sequence.approach_dist_m: pre-grasp = grasp pose backed off this far along the tool z
TOOL_MARGIN = 5.0       # mm added to the tool radius (pose error, calibration)
TOOL_HIT_MAX = 30       # measured points inside the tool allowed (depth noise)
PART_PUSH_MM = 10.0     # the gripper shell may press this far into neighbouring parts (loose parts give way); the bin
                        # wall and, for parts too, the wrist camera + link6 keep TOOL_MARGIN
# RealSense stand on the flange and its cable, as directions around the tool axis: deg about the tool z (link_eef,
# right-handed about the tool z that points to the tip). Seen from above with the tool pointing straight down
# (rpy 180 0 0): 0 = robot base +x (away from the robot), 180 = base -x (towards it), 90 = base -y, -90 = base +y;
# +90 from a direction = 90 deg clockwise from it. The fingers go into the round centre hole, so the tool may turn
# freely about its axis: a tilted tool is turned so that the stand and the cable are both on its upper side (their
# middle direction on the side the tool axis rises to), nothing of them hangs down towards the parts or the bin wall
MOUNT_DEG = 180.0       # the real robot: stand towards the robot base (as the URDF stand mesh; its camera_link is at +x)
CABLE_DEG = MOUNT_DEG - 90.0   # cable 90 deg counter-clockwise from the stand seen from above (base -y, tool-down pose)
UP_KEEP_MIN_TILT = 5.0  # deg: a tool tilted less than this has no upper side and is not turned
# if the gripper + camera hit something there, these turns (deg) from that pose are tried, the first clear one is used;
# within +-35 both stay within 80 deg of straight up
TOOL_YAWS = (0, 20, -20, 35, -35)
BIN_HUE, BIN_MIN_SAT, BIN_MIN_VAL = (130, 185), 120, 35   # bluish glints on the parts stay below saturation ~110
BIN_WALL_BELOW = 0.01   # m
# pick order: "top" = parts nothing lies on first (covered by others <= FREE_COVER), each group from the highest
# down; the rest by how little they are covered | "confidence" = conf score (the old rule)
PICK_ORDER = "top"
FREE_COVER = 0.02       # share of the part's silhouette under a higher part, up to this = free
ABOVE_MM = 5.0          # another part counts as lying on this one when its centre is this much higher (base z)
BIN_MIN_PX = 3000       # blue regions smaller than this are dropped; none left = no bin, the wall rule is off
SPECK_PX = 150          # depth islands smaller than this, cut off by invalid pixels or jumps > SPECK_JUMP_MM, are
SPECK_JUMP_MM = 20.0    # measurement outliers (solid_mask): not obstacles for the gripper checks

# --- auto pick with 2D correction ---
TRACK_SEARCH_PX = 40        # px, search radius of the 2D match (~17 mm at 0.6 m)
TRACK_MIN_NCC = 0.75        # match score below this = the part changed (taken, covered, flipped) -> not graspable
TRACK_MAX_SHIFT_MM = 15.0   # moved more than this -> not graspable (2D does not see height or tilt changes)
LITE6 = ROOT / "lite6_control"              # lite6_control repo inside this project (xarm_ros2 + lite6_ws)
EE_6D = LITE6 / "lite6_ws/src/my_lite6_control/my_lite6_control/ee_6d.py"
ROS_SETUP = (f"source /opt/ros/jazzy/setup.bash && source {LITE6}/xarm_ros2/install/setup.bash "
             f"&& source {LITE6}/lite6_ws/install/setup.bash")

UI = None   # queue.Queue of (kind, args) for the window; None = console only


def post(kind, *a):
    if UI is not None:
        UI.put((kind, a))


def log(msg=""):
    print(msg, flush=True)
    post("log", msg)


def state(key, text=""):
    post("state", key, text)


# ---------------------------------------------------------------------------------------------------------- models
cfg = rc = ic = device = net = pc = det = model = None


def load_models():
    global cfg, rc, ic, device, net, pc, det, model
    cfg = load_config(CONFIG)
    rc, ic = cfg.refine, cfg.infer
    rc.icp_retry_turns_deg = list(ICP_RETRY_TURNS)                   # failed ICP: second try (bin6d/refine.py)
    device = torch.device(cfg.device)
    ck_pose = torch.load(POSE, map_location=device, weights_only=False)
    pc = Cfg(ck_pose["cfg"]).pose                  # architecture the checkpoint was trained with
    pc.pretrained = False
    net = GDRNet(pc, input_channels()).to(device).eval()
    net.load_state_dict(ck_pose["model"])
    ck_det = torch.load(DETECTOR, map_location=device, weights_only=False)
    det = build_detector(Cfg(ck_det["cfg"]).detector).to(device).eval()
    det.load_state_dict(ck_det["model"])
    det.roi_heads.score_thresh = ic.score_thresh
    model = ObjectModel(BinPoseDataset(MODEL_ROOT), cfg.data.obj_id, pc.num_regions, pc.num_model_points)
    log(f"detector {DETECTOR} | pose {POSE} | part diameter {model.diameter:.1f} mm | device {device}")
    log(f"refine: depth_anchor={rc.depth_anchor}, dedup_mm={rc.dedup_mm}, reject_unverified={rc.reject_unverified}")


# ---------------------------------------------------------------------------------------------------------- camera
_lookups = {}


def _lookup(cam_dir, cam):
    key = (cam["width"], cam["height"], round(cam["fx"], 3), round(cam["fy"], 3), round(cam["cx"], 3), round(cam["cy"], 3))
    if key not in _lookups:
        ab = np.load(Path(cam_dir) / cam["undistort_map"])
        if ab.shape[:2] != (cam["height"], cam["width"]):
            raise ValueError(f"undistort map {ab.shape[:2]} does not match camera.json {cam['height']}x{cam['width']}")
        _lookups[key] = undistort_lookup(ab)
    return _lookups[key]


def _undistort_frame(rgb, depth, cam_dir, cam):
    lk = _lookup(cam_dir, cam)
    return undistort(rgb, *lk, order=1), undistort(depth, *lk, order=0)


def _undistort_img(img, cam, order):
    return undistort(img, *_lookup(CAPTURE_DIR, cam), order=order) if UNDISTORT else img


CONNECT_TRIES, CONNECT_RETRY_S = 5, 3.0     # the scanner may refuse for a few s after another process let it go


def _alive(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"   # zombie = already exited
    except OSError:
        return False


def release_scanner():
    """Stop photoneo_server.py of other programs (the notebook kernel keeps one running): only one process can hold
    the scanner. SIGTERM ends the server through its finally: cam.close(), so the device is released cleanly."""
    others = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"photoneo_server.py") for a in args) and _alive(pid):
            others.append(pid)
    for pid in others:
        log(f"photoneo_server.py PID {pid} (notebook kernel?) holds the scanner: stopping it")
        os.kill(int(pid), signal.SIGTERM)
    t_end = time.monotonic() + 8.0
    while others and time.monotonic() < t_end:
        others = [pid for pid in others if _alive(pid)]
        time.sleep(0.2)
    for pid in others:
        log(f"photoneo_server.py PID {pid} did not stop on SIGTERM: SIGKILL")
        os.kill(int(pid), signal.SIGKILL)


class PhotoneoServer:
    """photoneo_server.py kept running in the harvesters venv: the scanner stays connected, so a capture costs only the
    shot. request("3d") = scan + point cloud + 2D reference, request("2d") = 2D image only."""

    def __init__(self):
        self.p = None
        self.lock = threading.Lock()

    def start(self):
        if self.p is not None and self.p.poll() is None:
            return
        release_scanner()
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [str(HARVESTERS_PY), "photoneo_server.py", "--out", str(CAPTURE_DIR)] + ([SERIAL] if SERIAL else [])
        for attempt in range(1, CONNECT_TRIES + 1):
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                      stderr=open(CAPTURE_DIR / "server.log", "a"))
            r = self._read()
            if r.get("ready"):
                break
            self.p.wait()
            if "AccessDenied" not in str(r.get("error")) or attempt == CONNECT_TRIES:
                raise RuntimeError(f"photoneo_server.py: {r.get('error')} (log: {CAPTURE_DIR / 'server.log'}); "
                                   "another program (PhoXi Control, a notebook) may hold the scanner")
            log(f"scanner busy ({attempt}/{CONNECT_TRIES}), retry in {CONNECT_RETRY_S:.0f} s")
            time.sleep(CONNECT_RETRY_S)
        log(f"camera server: {r['model']} {r['serial']} | connect {r['timing']['connect']:.1f} s + setup "
            f"{r['timing']['setup']:.1f} s (once, not per capture)")

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"photoneo_server.py exited (log: {CAPTURE_DIR / 'server.log'})")
            if line.startswith("@@ "):
                return json.loads(line[3:])

    def request(self, cmd):
        with self.lock:
            self.start()
            self.p.stdin.write(json.dumps({"cmd": cmd}) + "\n")
            self.p.stdin.flush()
            r = self._read()
        if not r.get("ok"):
            raise RuntimeError(f"photoneo_server.py {cmd}: {r.get('error')}")
        return r

    def close(self):
        if self.p is not None and self.p.poll() is None:
            try:
                self.p.stdin.write('{"cmd": "quit"}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.p.kill()


CAM = PhotoneoServer()


def fmt_timing(tm):
    """One line of capture times (s) from photoneo_server.py + this process."""
    parts = []
    if "scan_3d" in tm:
        parts.append(f"3D scan {tm['scan_3d']:.2f}")
        parts.append(f"point cloud {tm['pointcloud']:.3f} (+ ply write {tm['save_ply']:.3f})")
    if "shot_2d" in tm:
        parts.append(f"2D shot {tm['shot_2d']:.2f}")
    sw = tm.get("mode_3d", 0) + tm.get("mode_2d", 0)
    if sw:
        parts.append(f"3D/2D mode switch {sw:.2f}")
    parts.append(f"save {tm.get('save_rgbd', 0) + tm.get('save_2d', 0):.2f}")
    if "load_undistort" in tm:
        parts.append(f"load + undistort {tm['load_undistort']:.2f}")
    if "request_total" in tm:
        parts.append(f"total {tm['request_total'] + tm.get('load_undistort', 0):.2f}")
    return " | ".join(parts) + " s"


def capture_frame():
    """One Photoneo shot -> {rgb, depth (mm), K, xyz, valid, gray2d (2D reference), timing (s), source}."""
    if OFFLINE is not None:
        root, view = OFFLINE
        ds = BinPoseDataset(root)
        f = load_frame(ds, frame_index(ds)[(ds.frames[0].scene, view)])
        return {k: f[k] for k in ("rgb", "depth", "K", "xyz", "valid")} | {"source": f"{root} view {view}"}
    t0 = time.perf_counter()
    r = CAM.request("3d")
    name, timing = r["name"], r["timing"]
    t1 = time.perf_counter()
    cam = json.load(open(CAPTURE_DIR / "camera.json"))
    if (cam["width"], cam["height"]) != (1236, 1032):
        log(f"WARNING: capture is {cam['width']}x{cam['height']}, models were trained on 1236x1032 (set DET_SCALE)")
    rgb = np.asarray(Image.open(CAPTURE_DIR / "rgb" / name).convert("RGB"))
    depth = np.asarray(Image.open(CAPTURE_DIR / "depth" / name), dtype=np.float32) * cam["depth_scale"]
    if UNDISTORT:
        rgb, depth = _undistort_frame(rgb, depth, CAPTURE_DIR, cam)
    gray_file = CAPTURE_DIR / "gray2d" / name.replace(".png", ".npy")
    gray = _undistort_img(np.load(gray_file).astype(np.float32), cam, 1) if gray_file.exists() else None
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], np.float32)
    timing = timing | {"request_total": t1 - t0, "load_undistort": time.perf_counter() - t1}
    return {"rgb": rgb, "depth": depth, "K": K, "xyz": backproject(depth, K), "valid": depth > 0, "gray2d": gray,
            "timing": timing, "source": str(CAPTURE_DIR / "rgb" / name)}


def capture_2d():
    """2D image only (CameraOnlyMode, same pixels as the 3D scan, undistorted like it) and its timing (s)."""
    t0 = time.perf_counter()
    r = CAM.request("2d")
    t1 = time.perf_counter()
    cam = json.load(open(CAPTURE_DIR / "camera.json"))
    gray = _undistort_img(np.load(CAPTURE_DIR / "gray2d" / r["name"]).astype(np.float32), cam, 1)
    return gray, r["timing"] | {"request_total": t1 - t0, "load_undistort": time.perf_counter() - t1}


# ------------------------------------------------------------------------------------------------------------ pose
@torch.no_grad()
def estimate_poses(f):
    """infer.py on one frame without GT: detect -> GDRN -> depth anchor -> ICP -> dedup -> verify -> confidence."""
    rgb, depth, K_np = f["rgb"], f["depth"], np.asarray(f["K"], np.float32)
    rgb_t, xyz_t, valid_t = (x.to(device) for x in frame_planes(rgb, depth, K_np))
    K = torch.from_numpy(K_np).to(device)
    h, w = depth.shape
    set_native_scale(det, round(h * DET_SCALE), round(w * DET_SCALE))
    o = det([rgb_t])[0]
    masks = o["masks"][:, 0] >= 0.5
    scores = o["scores"].cpu().numpy()
    big = masks.flatten(1).sum(1) >= ic.min_mask_px
    masks, scores = masks[big], scores[big.cpu().numpy()]
    if len(masks) == 0:
        return [], [], 0
    R, t = predict_poses(net, pc, model, rgb_t, xyz_t, valid_t, masks, K, device, ic.pose_batch)
    # depth refinement on each mask's bbox (+ ROI_PAD px) with K shifted to the crop: same result as the full image
    # (everything outside the bbox is off the mask), but no full-resolution mask copies (~4x faster on full bins)
    boxes = masks_to_boxes(masks).int().cpu().numpy()
    crops, offsets, results = [], [], []
    for n, (x0, y0, x1, y1) in enumerate(boxes):
        y0, x0 = max(y0 - ROI_PAD, 0), max(x0 - ROI_PAD, 0)
        y1, x1 = min(y1 + ROI_PAD + 1, h), min(x1 + ROI_PAD + 1, w)
        m = masks[n, y0:y1, x0:x1].cpu().numpy()
        crops.append(m)
        offsets.append((y0, x0))
        Kc = K_np.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        t0 = (depth_anchor(R[n], t[n], depth[y0:y1, x0:x1], m, Kc, model.dense_points, model.dense_normals,
                           rc.depth_anchor_min_px) if rc.get("depth_anchor") else t[n])
        ref = icp_refine(R[n], t0, observed_points(f["xyz"][y0:y1, x0:x1], f["valid"][y0:y1, x0:x1], m, rc.mask_erode_px),
                         model.dense_points, model.dense_normals, rc)
        results.append({"score": float(scores[n]), "R_rgbd": R[n].tolist(), "t_rgbd": t[n].tolist(),
                        "t_anchored": np.asarray(t0).tolist(),
                        "R": np.asarray(ref["R"]).tolist(), "t": np.asarray(ref["t"]).tolist(),
                        "icp_refined": ref["refined"], "icp_fitness": ref["fitness"], "icp_rmse": ref["rmse"],
                        "n_depth_points": ref["n_points"], "mask_idx": n, "mask_px": int(m.sum()),
                        "mask_cover": mask_cover(np.asarray(ref["R"]), np.asarray(ref["t"]), m, Kc,
                                                 model.dense_points, model.dense_normals)})
    results = dedup_detections(results, rc.get("dedup_mm", 0.0))
    rejected = []
    if rc.get("reject_unverified"):
        results, rejected = reject_unverified(results, crops, rc, offsets)
    flag_confident(results, rc)
    return results, rejected, len(crops)


# ------------------------------------------------------------------------------------------------- pick selection
def pose_rows(dets, frame):
    """Per output detection: depth check (support / violation / occluded) and the confidence score (summary §10)."""
    acc = [d["icp_rmse"] for d in dets if d["icp_refined"]]
    med_rmse = float(np.median(acc)) if acc else np.nan                # frame's typical fit error
    rows = []
    for n, d in enumerate(dets):
        sc = surface_check(d["R"], d["t"], frame["depth"], frame["K"], model)
        rows.append({"det": n, "confident": d["confident"], "fit error (mm)": d["icp_rmse"], "fitness": d["icp_fitness"],
                     "support": sc["support"], "violation": sc["violation"], "occluded": sc["occluded"],
                     "conf score": (float(sc["support"] * d["icp_fitness"] * min(1.0, med_rmse / max(d["icp_rmse"], 1e-6)))
                                    if d["confident"] else 0.0)})
    return pd.DataFrame(rows).set_index("det")


def silhouette(det, frame):
    # projected CAD silhouette (all CAD points, closed and filled) in the image
    p = (model.dense_points @ np.asarray(det["R"]).T + np.asarray(det["t"])) @ np.asarray(frame["K"]).T
    u, v = np.round(p[:, 0] / p[:, 2]).astype(int), np.round(p[:, 1] / p[:, 2]).astype(int)
    h, w = frame["depth"].shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), bool)
    m[v[ok], u[ok]] = True
    return binary_fill_holes(binary_closing(m, iterations=3))


def top_picks(df, dets, frame, k=None):
    # graspable parts (conditions 1-3) in pick_order, skipping any that overlaps a part already selected
    k = TOP_K if k is None else k                                    # read at call time: auto_pick.py --per-scan
    if not df.confident.any():
        return df.iloc[:0].assign(rank=[]), df.assign(**{"grasp ok": False})
    g = pd.DataFrame({n: grasp_check(dets[n], frame) for n in df.index[df.confident]}).T
    num = [c for c in g.columns if c not in ("approach side", "approach (cam)")]
    g[num] = g[num].astype(float)
    full = df.join(g)
    full["grasp ok"] = full.apply(lambda r: bool(r["confident"]) and why_not(r) == "", axis=1)
    order = pick_order(full[full["grasp ok"]], dets, frame)          # which graspable part first
    for c in order.columns.difference(full.columns):
        full[c] = order[c]
    sil, chosen = {}, []
    for n in order.index:
        sil[n] = silhouette(dets[n], frame)
        a = max(sil[n].sum(), 1)
        if any((sil[n] & sil[m]).sum() / a > MAX_OVERLAP_SELECTED for m in chosen):
            continue
        chosen.append(n)
        if len(chosen) == k:
            break
    best = full.loc[chosen].copy()
    best.insert(0, "rank", range(1, len(best) + 1))
    return best, full


def bin_wall(frame):
    """Blue bin pixels: image mask and their points in the robot base (m); None when no bin is seen."""
    hsv = np.asarray(Image.fromarray(frame["rgb"]).convert("HSV")).astype(int)
    blue = ((hsv[..., 0] >= BIN_HUE[0]) & (hsv[..., 0] <= BIN_HUE[1]) & (hsv[..., 1] >= BIN_MIN_SAT)
            & (hsv[..., 2] >= BIN_MIN_VAL) & frame["valid"])
    lab, _ = label(binary_opening(blue))                             # keep the large blue regions (rim, walls)
    size = np.bincount(lab.ravel())
    size[0] = 0
    blue = np.isin(lab, np.nonzero(size >= BIN_MIN_PX)[0])
    if not blue.any():
        return None
    p = (frame["xyz"][blue].astype(float) @ T_OUT[:3, :3].T + T_OUT[:3, 3]) / 1000.0   # camera mm -> base m
    return {"mask": blue, "points": p, "top_z": float(np.percentile(p[:, 2], 95))}


def solid_mask(frame):
    """Valid depth pixels without measurement outliers: islands of fewer than SPECK_PX pixels, cut off from the rest by
    invalid pixels or depth jumps > SPECK_JUMP_MM (the nearer side of a jump is cut), are left out. Flying pixels and
    reflections in the centre holes would otherwise stand in the way of the gripper in tool_hits / grasp_check."""
    d, v = frame["depth"], frame["valid"]
    cut = np.zeros_like(v)
    for ax in (0, 1):
        jump = np.diff(d, axis=ax)
        both = (v[1:] & v[:-1]) if ax == 0 else (v[:, 1:] & v[:, :-1])
        near_first, near_second = both & (jump > SPECK_JUMP_MM), both & (jump < -SPECK_JUMP_MM)
        if ax == 0:
            cut[:-1] |= near_first
            cut[1:] |= near_second
        else:
            cut[:, :-1] |= near_first
            cut[:, 1:] |= near_second
    lab, _ = label(v & ~cut)
    big = np.bincount(lab.ravel()) >= SPECK_PX
    big[0] = False
    return big[lab]


def wall_dist(det, frame):
    """Horizontal distance (m, robot base) from the part centre to the nearest bin wall point above it
    (inf = no wall that high, nan = no bin in the frame)."""
    wall = frame.get("wall")
    if wall is None:
        return float("nan")
    c = (T_OUT[:3, :3] @ np.asarray(det["t"], float) + T_OUT[:3, 3]) / 1000.0
    p = wall["points"]
    p = p[p[:, 2] > c[2] - BIN_WALL_BELOW]
    return float(np.sqrt(((p[:, :2] - c[:2]) ** 2).sum(1).min())) if len(p) else float("inf")


_tool_table = None


def _stl(path, steps=6):
    """Points on the surface of an STL mesh (each triangle sampled on a barycentric grid, not only its corners)."""
    b = open(path, "rb").read()
    if b[:5] == b"solid" and b"facet" in b[:300]:
        v = np.array([l.split()[1:4] for l in b.decode(errors="ignore").splitlines()
                      if l.strip().startswith("vertex")], float)
    else:
        n = int.from_bytes(b[80:84], "little")
        v = np.frombuffer(b[84:84 + 50 * n], np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"]
    tri = np.asarray(v, float).reshape(-1, 3, 3)
    w = np.array([(i / steps, j / steps) for i in range(steps + 1) for j in range(steps + 1 - i)])
    return (tri[:, None, 0] + w[None, :, :1] * (tri[:, None, 1] - tri[:, None, 0])
            + w[None, :, 1:] * (tri[:, None, 2] - tri[:, None, 0])).reshape(-1, 3)


def tool_table():
    """(z edges, angle edges, {"shell": radius table, "rigid": radius table}) in the link_eef frame (m): the largest
    distance from the tool axis per 5 mm of z and 10 deg of direction (0 = nothing there), for the gripper shell and
    for the wrist camera + link6."""
    global _tool_table
    if _tool_table is None:
        flip = np.diag([-1.0, -1.0, 1.0])                            # d435 stand: collision origin rpy 0 0 pi
        z_edges, a_edges = np.arange(-0.030, 0.065, 0.005), np.radians(np.arange(-180, 181, 10))

        def radii(p):
            p = p / 1000.0 if np.abs(p).max() > 5 else p
            zi = np.digitize(p[:, 2], z_edges) - 1
            ai = np.clip(np.digitize(np.arctan2(p[:, 1], p[:, 0]), a_edges) - 1, 0, len(a_edges) - 2)
            r = np.zeros((len(z_edges) - 1, len(a_edges) - 1))
            ok = (zi >= 0) & (zi < len(z_edges) - 1)
            np.maximum.at(r, (zi[ok], ai[ok]), np.hypot(p[ok, 0], p[ok, 1]))
            r = np.maximum(r, np.roll(r, 1, 1))                      # neighbouring directions: no gaps between bins
            return np.maximum(r, np.roll(r, -1, 1))

        metres = lambda p: p / 1000.0 if np.abs(p).max() > 5 else p
        cam = metres(_stl(TOOL_MESH_DIR / "camera/realsense/collision/d435_with_cam_stand.stl") @ flip)
        far = cam[np.hypot(cam[:, 0], cam[:, 1]) > 0.06]                # the camera body, away from the flange
        turn = np.radians(MOUNT_DEG) - np.arctan2(far[:, 1].mean(), far[:, 0].mean())
        cam = cam @ np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]]).T
        _tool_table = z_edges, a_edges, {                            # stand + camera turned to MOUNT_DEG
            "shell": radii(_stl(TOOL_MESH_DIR / "gripper/lite/visual/shell.stl")),
            "rigid": radii(np.vstack([cam, metres(_stl(TOOL_MESH_DIR / "lite6/collision/link6.stl"))]))}
    return _tool_table


def up_keeping_yaw(T_tcp):
    """Turn (deg about the tool z) that puts the middle of the RealSense stand (MOUNT_DEG) and its cable (CABLE_DEG)
    on the upper side of the tilted tool: the side its axis rises to, robot base +Z seen in the tool frame.
    0 for a tool within UP_KEEP_MIN_TILT of straight down (no upper side)."""
    up = T_tcp[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    if np.hypot(up[0], up[1]) < np.sin(np.radians(UP_KEEP_MIN_TILT)):
        return 0.0
    m, c = np.radians(MOUNT_DEG), np.radians(CABLE_DEG)
    middle = np.arctan2(np.sin(m) + np.sin(c), np.cos(m) + np.cos(c))
    return float((np.degrees(np.arctan2(up[1], up[0]) - middle) + 180.0) % 360.0 - 180.0)


def tool_hits(T_tcp, frame, base_yaw=0.0):
    """Measured points (camera cloud: bin walls, other parts) inside the gripper + wrist camera at the grasp pose
    or anywhere on the approach from the pre-grasp pose, for the tool turned by base_yaw + each of TOOL_YAWS.
    Returns (yaw deg: the first with at most TOOL_HIT_MAX points, else the one with the fewest, its count,
    share of its points on the blue bin). Bin points count inside the whole tool + TOOL_MARGIN; part points inside the
    wrist camera / link6 + TOOL_MARGIN, or deeper than PART_PUSH_MM into the gripper shell (sideways or from below)."""
    z_edges, a_edges, tables = tool_table()
    r = np.maximum(tables["shell"], tables["rigid"])
    R, eef = T_tcp[:3, :3], T_tcp[:3, 3] - T_tcp[:3, :3][:, 2] * TCP_OFFSET_M
    K, xyz, valid = frame["K"], frame["xyz"], frame.get("solid", frame["valid"])   # without depth outliers
    c = np.linalg.inv(T_OUT) @ np.append(T_tcp[:3, 3] * 1000.0, 1.0)          # TCP in the camera frame (mm)
    u0, v0 = K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]
    pad_px = int(120.0 * K[0, 0] / max(c[2] - 200.0, 200.0))       # 12 cm around, tool top ~20 cm nearer the camera
    hh, ww = valid.shape
    ys = slice(max(int(v0) - pad_px, 0), min(int(v0) + pad_px, hh))
    xs = slice(max(int(u0) - pad_px, 0), min(int(u0) + pad_px, ww))
    m = valid[ys, xs]
    p = (xyz[ys, xs][m].astype(float) @ T_OUT[:3, :3].T + T_OUT[:3, 3]) / 1000.0
    q = (p - eef) @ R                                                # link_eef frame (m), z towards the tip
    near = np.hypot(q[:, 0], q[:, 1]) < r.max() + TOOL_MARGIN / 1000.0
    q, on_blue = q[near], (frame["wall"]["mask"][ys, xs][m][near] if frame.get("wall") else np.zeros(near.sum(), bool))
    rho, ang = np.hypot(q[:, 0], q[:, 1]), np.arctan2(q[:, 1], q[:, 0])
    best = None
    for yaw in ((base_yaw + off + 180.0) % 360.0 - 180.0 for off in TOOL_YAWS):
        # tool turned by +yaw = the points turned by -yaw in its frame
        ai = np.clip(np.digitize((ang - np.radians(yaw) + np.pi) % (2 * np.pi) - np.pi, a_edges) - 1, 0,
                     len(a_edges) - 2)
        hit = np.zeros(len(q), bool)
        lookup = lambda table, z: (lambda zi: np.where((zi >= 0) & (zi < len(z_edges) - 1),
                                                       table[np.clip(zi, 0, len(z_edges) - 2), ai], 0.0))(
            np.digitize(z, z_edges) - 1)
        margin, push = TOOL_MARGIN / 1000.0, PART_PUSH_MM / 1000.0
        for s in np.arange(0.0, APPROACH_M + 1e-9, 0.01):            # tool backed off s along its axis
            whole = lookup(r, q[:, 2] + s)
            rigid = lookup(tables["rigid"], q[:, 2] + s)
            shell_in = lookup(tables["shell"], q[:, 2] + s + push)   # shell shrunk by push from below ...
            hit |= np.where(on_blue, (whole > 0) & (rho < whole + margin),
                            ((rigid > 0) & (rho < rigid + margin)) | ((shell_in > 0) & (rho < shell_in - push)))
        n = int(hit.sum())                                           # ... and sideways, for the parts only
        if best is None or n < best[1]:
            best = (yaw, n, float(on_blue[hit].mean()) if n else 0.0)
        if n <= TOOL_HIT_MAX:
            break
    return best


def grasp_check(det, frame):
    # entrance of the centre hole on the upper side (base frame), part axis, and what the measured cloud has around it.
    # The upper side is the end the part axis points to in the base frame, the same rule as tcp_pose, so the approach
    # (-up) is the tool z of the TCP target for upright, upside-down and tilted parts alike.
    R, t = np.asarray(det["R"], float), np.asarray(det["t"], float)
    hub = R[:, 2]
    side = 1.0 if hub @ UP_CAM >= 0 else -1.0                        # +1: hub end points up (base +Z)
    z_top = model.xyz_min[2] + model.xyz_size[2] if side > 0 else model.xyz_min[2]
    entry = R @ np.array([0.0, 0.0, z_top]) + t
    up = side * hub                                                  # out of the part, upwards
    tilt = float(np.degrees(np.arccos(np.clip(up @ UP_CAM, -1, 1))))
    K, xyz, valid = frame["K"], frame["xyz"], frame.get("solid", frame["valid"])   # without depth outliers
    c = K @ entry
    u0, v0 = c[0] / c[2], c[1] / c[2]
    half = int(1.5 * (R_BODY + 5) * K[0, 0] / entry[2]) + 2
    hh, ww = valid.shape
    ys = slice(max(int(v0) - half, 0), min(int(v0) + half, hh))
    xs = slice(max(int(u0) - half, 0), min(int(u0) + half, ww))
    d = xyz[ys, xs][valid[ys, xs]].astype(float) - entry
    h = d @ up                                                       # height above the entrance plane
    rho = np.linalg.norm(d - h[:, None] * up, axis=1)                # distance from the hole axis
    # blocked AREA share: blocking points / pixels the disc or ring covers in the image
    px_per_mm2 = (K[0, 0] / entry[2]) * (K[1, 1] / entry[2]) * max(abs(up[2]), 0.2)
    hole_px = np.pi * R_FINGER ** 2 * px_per_mm2
    ring_px = np.pi * (R_BODY ** 2 - R_FINGER ** 2) * px_per_mm2
    hole_blocked = min(float(((rho < R_FINGER) & (h > -INSERT_DEPTH)).sum()) / hole_px, 1.0)
    ring_blocked = min(float(((rho >= R_FINGER) & (rho < R_BODY) & (h > RING_MARGIN) & (h < APPROACH_LEN)).sum())
                       / ring_px, 1.0)
    T_tcp = tcp_pose(det | {"tool_yaw": 0.0})[1]
    yaw, hits, on_bin = tool_hits(T_tcp, frame, base_yaw=up_keeping_yaw(T_tcp))   # stand + cable kept up
    det["tool_yaw"] = float(yaw)                                     # tcp_pose of this part uses it from now on
    return {"approach side": "hub" if side > 0 else "plate", "approach tilt (deg)": tilt,
            "hole blocked": hole_blocked, "ring blocked": ring_blocked, "approach (cam)": (-up).tolist(),
            "reach (m)": float(np.linalg.norm(T_tcp[:3, 3])),           # |TCP target| from the robot base
            "tool hits": hits, "tool hits on bin": on_bin, "tool yaw (deg)": float(yaw),
            "wall dist (m)": wall_dist(det, frame),
            "height (m)": float((T_OUT[2, :3] @ t + T_OUT[2, 3]) / 1000.0)}  # part centre, base z


def tcp_pose(det):
    """Part pose and robot TCP target in the robot base frame (metres).
    TCP = finger tip (the robot config's tcp_offset_m, 0.106 m, is the flange -> finger tip length): INSERT_DEPTH mm
    past the centre-hole entrance of the upper end, along the tool z. Tool z runs along the part axis (CAD z) and
    points DOWN in the base frame:
    part z up -> tool z = -part z, part z down -> tool z = +part z, and the same rule when the part is tilted.
    Rotation about the tool z: GRIP_YAW "down" = the shortest rotation taking the tool-down pose (rpy 180, 0, 0) onto
    that tool z, so the wrist only tilts and never spins with the part; "part" = tool x = part x (the part is 180 deg
    symmetric about its axis, so of +-x the one closer to tool-down)."""
    T_obj = np.eye(4)
    T_obj[:3, :3], T_obj[:3, 3] = np.asarray(det["R"]), np.asarray(det["t"])
    T_obj = T_OUT @ T_obj
    T_obj[:3, 3] /= 1000.0                                           # mm -> m
    part_z_up = bool(T_obj[2, 2] >= 0)
    z = -T_obj[:3, 2] if part_z_up else T_obj[:3, 2]
    if GRIP_YAW == "part":
        x = T_obj[:3, 0]
        R_tcp = min((np.column_stack([s * x, np.cross(z, s * x), z]) for s in (1.0, -1.0)),
                    key=lambda R: Rotation.from_matrix(R_TOOL_DOWN.T @ R).magnitude())
    else:
        z0 = R_TOOL_DOWN[:, 2]                                       # (0, 0, -1); z always points down, never opposite
        axis = np.cross(z0, z)
        angle = np.arctan2(np.linalg.norm(axis), z0 @ z)
        R_align = (Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
                   if np.linalg.norm(axis) > 1e-9 else np.eye(3))
        R_tcp = R_align @ R_TOOL_DOWN
    if det.get("tool_yaw"):                                          # turned about the tool z (grasp_check, TOOL_YAWS)
        R_tcp = R_tcp @ Rotation.from_euler("z", det["tool_yaw"], degrees=True).as_matrix()
    z_top = model.xyz_min[2] + model.xyz_size[2] if part_z_up else model.xyz_min[2]   # upper end face (CAD z, mm)
    entry = T_obj[:3, 3] + T_obj[:3, 2] * z_top / 1000.0                               # hole entrance
    T_tcp = np.eye(4)
    T_tcp[:3, :3], T_tcp[:3, 3] = R_tcp, entry + z * INSERT_DEPTH / 1000.0
    return T_obj, T_tcp, part_z_up


def safety_issue(p):
    """Why the robot's safety box would reject TCP position p ("" = inside)."""
    for name, v, (lo, hi) in (("x", p[0], SAFE_X), ("y", p[1], SAFE_Y), ("z", p[2], SAFE_Z)):
        if not lo <= v <= hi:
            return f"{name} {v:.3f} outside [{lo}, {hi}]"
    if np.linalg.norm(p) > SAFE_REACH:
        return f"reach {np.linalg.norm(p):.3f} > {SAFE_REACH}"
    return ""


def why_not(r, detail=True):
    # first failed condition ("" = graspable); detail=False gives just the category
    if not r["confident"]:
        return "not confident"
    for bad, name, val in ((r["reach (m)"] > SAFE_REACH, "out of reach", f"{r['reach (m)']:.3f} m"),
                           (r["wall dist (m)"] < BIN_WALL_MARGIN, "near bin wall", f"{r['wall dist (m)']:.3f} m"),
                           (r["occluded"] > GRASP_MAX_OCCLUDED, "covered", f"occluded {r['occluded']:.2f}"),
                           (r["approach tilt (deg)"] > MAX_TILT, "tilted", f"{r['approach tilt (deg)']:.0f} deg"),
                           (r["tool hits"] > TOOL_HIT_MAX,
                            "gripper hits bin wall" if r["tool hits on bin"] >= 0.5 else "gripper hits parts",
                            f"{r['tool hits']:.0f} points"),
                           (r["hole blocked"] > HOLE_BLOCK_MAX, "hole blocked", f"{r['hole blocked']:.0%}"),
                           (r["ring blocked"] > RING_BLOCK_MAX, "gripper ring blocked", f"{r['ring blocked']:.0%}")):
        if bad:
            return f"{name} {val}" if detail else name
    return ""


def _silhouette_crop(det, frame):
    """CAD silhouette in the image as (y0, x0, mask) of its bounding box (silhouette() without the full frame)."""
    p = (model.dense_points @ np.asarray(det["R"]).T + np.asarray(det["t"])) @ np.asarray(frame["K"]).T
    u, v = np.round(p[:, 0] / p[:, 2]).astype(int), np.round(p[:, 1] / p[:, 2]).astype(int)
    y0, x0 = v.min() - 4, u.min() - 4
    m = np.zeros((v.max() - y0 + 5, u.max() - x0 + 5), bool)
    m[v - y0, u - x0] = True
    return y0, x0, binary_fill_holes(binary_closing(m, iterations=3))


def covered_by_others(n, dets, frame, sils, heights):
    """Share of part n's silhouette that lies under the silhouette of a part whose centre is ABOVE_MM higher."""
    y0, x0, m = sils[n]
    under = np.zeros_like(m)
    for k, (yk, xk, mk) in sils.items():
        if k == n or heights[k] < heights[n] + ABOVE_MM / 1000.0:
            continue
        ya, yb = max(y0, yk), min(y0 + m.shape[0], yk + mk.shape[0])
        xa, xb = max(x0, xk), min(x0 + m.shape[1], xk + mk.shape[1])
        if ya < yb and xa < xb:
            under[ya - y0:yb - y0, xa - x0:xb - x0] |= mk[ya - yk:yb - yk, xa - xk:xb - xk]
    return float((under & m).sum() / max(m.sum(), 1))


def pick_order(ok, dets, frame):
    """Graspable rows in the order they should be picked (PICK_ORDER), with the column "covered by others"."""
    if ok.empty or PICK_ORDER == "confidence":
        return ok.sort_values("conf score", ascending=False).assign(**{"covered by others": np.nan})
    sils = {k: _silhouette_crop(d, frame) for k, d in enumerate(dets)}
    heights = {k: float((T_OUT[2, :3] @ np.asarray(d["t"], float) + T_OUT[2, 3]) / 1000.0) for k, d in enumerate(dets)}
    ok = ok.assign(**{"covered by others": [covered_by_others(n, dets, frame, sils, heights) for n in ok.index]})
    free = ok["covered by others"] <= FREE_COVER
    return pd.concat([ok[free].sort_values("height (m)", ascending=False),
                      ok[~free].sort_values(["covered by others", "height (m)"], ascending=[True, False])])


def scan_3d(save=True):
    """run_cycle of the notebook: 3D capture -> pose -> picks.
    Returns {frame, dets, picks (DataFrame by rank, empty = nothing graspable), image, summary}."""
    t0 = time.perf_counter()
    f = capture_frame()
    f["wall"] = bin_wall(f)
    f["solid"] = solid_mask(f)
    t1 = time.perf_counter()
    dets, rejected, n_masks = estimate_poses(f)
    t2 = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log(f"{f['source']} | capture {t1 - t0:.1f} s, pose {t2 - t1:.1f} s | "
        + (f"blue bin found (rim z {f['wall']['top_z']:.3f} m)"
           + (f", skipping parts within {BIN_WALL_MARGIN} m of its wall" if BIN_WALL_MARGIN > 0 else "")
           if f["wall"] else "NO blue bin found")
        + " | gripper + wrist camera checked against the measured cloud")
    if f.get("timing"):
        log("camera: " + fmt_timing(f["timing"]))
    res = {"frame": f, "dets": dets, "picks": pd.DataFrame(), "image": Image.fromarray(f["rgb"]),
           "summary": {"capture (s)": t1 - t0, "pose (s)": t2 - t1, "masks": n_masks, "verified": len(dets),
                       "confident": 0, "graspable": 0, "selected": 0}}
    if not dets:
        log("no verified side plate in this frame")
        return res
    df = pose_rows(dets, f)
    best, full = top_picks(df, dets, f)
    res["full"] = full                                               # every detection with its checks (auto_pick.py)
    res["summary"] |= {"confident": int(df.confident.sum()), "graspable": int(full["grasp ok"].sum()),
                       "selected": len(best)}
    log(f"{n_masks} detector masks -> {len(dets)} verified ({len(rejected)} rejected) -> {int(df.confident.sum())} confident "
        f"-> {int(full['grasp ok'].sum())} graspable -> {len(best)} selected | frame: {FRAME}")
    reasons = full[full.confident.astype(bool)].apply(lambda r: why_not(r, detail=False), axis=1, result_type="reduce")
    if (reasons != "").any():
        log("confident but not graspable: " + ", ".join(f"{k} {v}" for k, v in reasons[reasons != ""].value_counts().items()))
    if best.empty:
        log("no graspable side plate in this frame")
        return res

    # Poses for ee_6d.py (src): robot base, metres; "x y z roll pitch yaw" is read as INTRINSIC XYZ (R = Rx Ry Rz),
    # "x y z qx qy qz qw" as a quaternion. Angles of another convention would tilt the tool the wrong way.
    rows, out = [], []
    for n, r in best.iterrows():
        T_obj, T_tcp, part_z_up = tcp_pose(dets[n])
        rot = Rotation.from_matrix(T_tcp[:3, :3])
        rpy = rot.as_euler("XYZ", degrees=True)
        part_rpy = Rotation.from_matrix(T_obj[:3, :3]).as_euler("XYZ", degrees=True)
        qx, qy, qz, qw = rot.as_quat()
        safety = safety_issue(T_tcp[:3, 3])
        rows.append({"rank": r["rank"], "det": n, "conf score": r["conf score"],
                     "tcp x (m)": T_tcp[0, 3], "tcp y (m)": T_tcp[1, 3], "tcp z (m)": T_tcp[2, 3],
                     "tcp roll (deg)": rpy[0], "tcp pitch (deg)": rpy[1], "tcp yaw (deg)": rpy[2],
                     "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                     "part x (m)": T_obj[0, 3], "part y (m)": T_obj[1, 3], "part z (m)": T_obj[2, 3],
                     "part roll (deg)": part_rpy[0], "part pitch (deg)": part_rpy[1], "part yaw (deg)": part_rpy[2],
                     "part axis": "up" if part_z_up else "down", "tool z (base)": np.round(T_tcp[:3, 2], 4).tolist(),
                     "tool tilt (deg)": float(np.degrees(np.arccos(np.clip(-T_tcp[2, 2], -1, 1)))),
                     "tool yaw (deg)": r["tool yaw (deg)"], "tool hits": r["tool hits"],
                     "approach tilt (deg)": r["approach tilt (deg)"], "hole blocked": r["hole blocked"],
                     "ring blocked": r["ring blocked"], "wall dist (m)": r["wall dist (m)"], "height (m)": r["height (m)"],
                     "covered by others": r["covered by others"], "occluded": r["occluded"],
                     "fit error (mm)": r["fit error (mm)"], "robot safety": safety or "ok"})
        out.append({"rank": int(r["rank"]), "conf_score": float(r["conf score"]),
                    "tcp_position_m": np.round(T_tcp[:3, 3], 6).tolist(),
                    "tcp_rpy_intrinsic_XYZ_deg": np.round(rpy, 3).tolist(),
                    "tcp_quat_xyzw": np.round([qx, qy, qz, qw], 6).tolist(), "T_base_tcp": np.round(T_tcp, 6).tolist(),
                    "part_position_m": np.round(T_obj[:3, 3], 6).tolist(),
                    "part_rpy_intrinsic_XYZ_deg": np.round(part_rpy, 3).tolist(), "T_base_part": np.round(T_obj, 6).tolist(),
                    "part_axis": "up" if part_z_up else "down", "tool_yaw_deg": float(r["tool yaw (deg)"]),
                    "robot_safety": safety or "ok"})
    picks = pd.DataFrame(rows).set_index("rank")

    for k, r in picks.iterrows():
        xyz = f"{r['tcp x (m)']:.4f} {r['tcp y (m)']:.4f} {r['tcp z (m)']:.4f}"
        tz = r["tool z (base)"]
        log(f"#{k:<2d} score {r['conf score']:.3f} | part axis {r['part axis']:4s} -> tool z down "
            f"| tool tilted {r['tool tilt (deg)']:.1f} deg from straight down, turned {r['tool yaw (deg)']:+.0f} deg "
            f"about its axis (RealSense stand + cable up), tip moving toward "
            f"({tz[0]:+.2f} x, {tz[1]:+.2f} y) of the base | safety: {r['robot safety']}")
        log(f"    ee_6d quat: {xyz} {r['qx']:.6f} {r['qy']:.6f} {r['qz']:.6f} {r['qw']:.6f}")

    # rank labels at each origin
    rgb_vis = f["rgb"].copy()
    if f["wall"]:                                                    # detected blue bin in cyan
        rgb_vis[f["wall"]["mask"]] = (0.5 * rgb_vis[f["wall"]["mask"]] + (0, 127, 127)).astype(np.uint8)
    img = render_2d(rgb_vis, f["K"], [d if n in set(best.index) else None for n, d in enumerate(dets)], model, None,
                    labels="none", title=f"{len(best)} picks (number = pick order)")
    dr = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=28)
    for n, r in best.iterrows():
        c = np.asarray(f["K"]) @ np.asarray(dets[n]["t"])
        u, v = c[0] / c[2], c[1] / c[2]
        dr.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(255, 230, 0))
        dr.text((u + 8, v - 30), str(int(r["rank"])), fill=(255, 230, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    res.update(picks=picks, image=img)

    if save:
        d = OUT_DIR / stamp
        d.mkdir(parents=True, exist_ok=True)
        json.dump({"source": f["source"], "frame": "base", "unit": "m", "rpy": "intrinsic XYZ (ee_6d.py)",
                   "T_cam_base": T_CAM_BASE.tolist(),
                   "tcp": f"finger tip {INSERT_DEPTH} mm inside the upper hole entrance, tool z along the part axis pointing down", "picks": out},
                  open(d / "picks.json", "w"), indent=1)
        p = picks.copy()
        p["tool z (base)"] = p["tool z (base)"].map(lambda a: " ".join(f"{x:.4f}" for x in a))
        p.to_csv(d / "picks.csv", float_format="%.6f")
        json.dump(dets, open(d / "detections.json", "w"))
        img.save(d / "overlay.png")
        Image.fromarray(f["rgb"]).save(d / "rgb.png")
        np.save(d / "depth.npy", f["depth"].astype(np.float32))
        log(f"saved -> {d}")
    return res


# -------------------------------------------------------------------------------------------- 2D correction
def ncc_match(ref, cur, mask, search):
    """Masked normalised cross-correlation of ref (pixels in mask) over cur, shifts up to +-search px.
    Returns (du, dv) of the best match, its score and the score at zero shift."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1].astype(float)
    t = ref[y0:y1, x0:x1].astype(float)
    n = m.sum()
    tz = m * (t - (m * t).sum() / n)                                 # zero-mean template inside the mask
    H, W = cur.shape
    sy0, sy1, sx0, sx1 = max(y0 - search, 0), min(y1 + search, H), max(x0 - search, 0), min(x1 + search, W)
    img = cur[sy0:sy1, sx0:sx1].astype(float)
    corr = lambda a, b: fftconvolve(a, b[::-1, ::-1], mode="valid")
    s1, s2 = corr(img, m), corr(img * img, m)
    ncc = corr(img, tz) / np.sqrt(np.maximum((tz ** 2).sum() * (s2 - s1 ** 2 / n), 1e-9))
    i, j = np.unravel_index(np.argmax(ncc), ncc.shape)
    return (int(sx0 + j - x0), int(sy0 + i - y0)), float(ncc[i, j]), float(ncc[y0 - sy0, x0 - sx0])


def target_of(T_tcp):
    return {"xyz": T_tcp[:3, 3].tolist(), "quat_xyzw": Rotation.from_matrix(T_tcp[:3, :3]).as_quat().tolist()}


def pick_queue(picks, dets, scan):
    queue = []
    for k, r in picks.iterrows():
        if r["robot safety"] != "ok":
            log(f"scan {scan} #{k}: robot safety {r['robot safety']}, skipped")
            continue
        det = dets[int(r["det"])]
        d = {"R": det["R"], "t": det["t"], "tool_yaw": det.get("tool_yaw", 0.0)}
        queue.append({"name": f"{scan}-{k}", "rank": int(k), "pose": d, "T_tcp": tcp_pose(d)[1], "shift (mm)": 0.0})
    return queue


def track_2d(queue, ref, cur, frame):
    """Find every queued part in the new 2D image; correct its position or drop it. Returns (kept, dropped)."""
    K = np.asarray(frame["K"], float)
    kept, dropped = [], []
    for q in queue:
        (du, dv), score, zero = ncc_match(ref, cur, silhouette(q["pose"], frame), TRACK_SEARCH_PX)
        t = np.asarray(q["pose"]["t"], float)
        dt = np.array([du * t[2] / K[0, 0], dv * t[2] / K[1, 1], 0.0])  # same depth, shifted in the image
        moved = float(np.linalg.norm(dt))
        if score < TRACK_MIN_NCC:
            dropped.append((q, f"changed (match {score:.2f})"))
            continue
        if moved > TRACK_MAX_SHIFT_MM:
            dropped.append((q, f"moved {moved:.0f} mm"))
            continue
        pose = q["pose"] | {"t": (t + dt).tolist()}
        T_tcp = tcp_pose(pose)[1]
        issue = safety_issue(T_tcp[:3, 3])
        wd = wall_dist(pose, frame)
        if issue or wd < BIN_WALL_MARGIN:
            dropped.append((q, issue or f"near bin wall {wd:.3f} m"))
            continue
        kept.append(q | {"pose": pose, "T_tcp": T_tcp, "shift (mm)": q["shift (mm)"] + moved, "match": score})
    return kept, dropped


def draw_2d(base, queue, frame, current=None):
    """2D image (or the scan's rgb) with the picks that can still be grasped (number = pick order) and, in orange,
    the part being picked now."""
    if base.ndim == 2:
        g = np.clip(base / max(np.percentile(base, 99.5), 1.0), 0, 1)
        img = Image.fromarray((g * 255).astype(np.uint8)).convert("RGB")
    else:
        img = Image.blend(Image.fromarray(base).convert("RGB"), Image.new("RGB", base.shape[1::-1]), 0.35)
    dr = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=28)
    K = np.asarray(frame["K"], float)
    marks = [(q, str(n), (0, 255, 0), 3) for n, q in enumerate(queue, 1)]
    if current is not None:
        marks.append((current, "now", (255, 140, 0), 6))
    for q, text, colour, width in marks:
        t = np.asarray(q["pose"]["t"], float)
        c = K @ t
        u, v, rad = c[0] / c[2], c[1] / c[2], 27.85 * K[0, 0] / t[2]
        dr.ellipse((u - rad, v - rad, u + rad, v + rad), outline=colour, width=width)
        dr.text((u + 8, v - 30), text, fill=(255, 230, 0) if current is not q else colour, font=font, stroke_width=3,
                stroke_fill=(0, 0, 0))
    return img


def queue_rows(queue):
    return [(n, p["name"], f"{p['T_tcp'][0, 3]:.4f}", f"{p['T_tcp'][1, 3]:.4f}", f"{p['T_tcp'][2, 3]:.4f}",
             f"{np.degrees(np.arccos(np.clip(-p['T_tcp'][2, 2], -1, 1))):.1f}",
             f"{p['match']:.2f}" if "match" in p else "-", f"{p['shift (mm)']:.1f}") for n, p in enumerate(queue, 1)]


def check_2d(queue, ref, frame, label):
    """2D image now: correct / drop the queued picks against the previous 2D image ref. Returns (queue, new ref)."""
    cur, tm = capture_2d()
    t0 = time.perf_counter()
    queue, dropped = track_2d(queue, ref, cur, frame)
    t_track = time.perf_counter() - t0
    log(f"2D check {label}: camera {fmt_timing(tm)} | match {t_track:.2f} s -> {len(queue)} graspable"
        + (", dropped " + ", ".join(f"{d['name']} {why}" for d, why in dropped) if dropped else ""))
    post("dropped", [(d["name"], why) for d, why in dropped])
    if dropped and len(dropped) > max(2, len(queue)):
        log("  many picks changed at once: bin changed, or the robot arm in the camera view at its home pose?")
    post("image", draw_2d(cur, queue, frame), f"2D 확인 ({label}): 남은 pick {len(queue)}개")
    post("queue", queue_rows(queue))
    return queue, cur


# ----------------------------------------------------------------------------------------------------------- robot
class Robot:
    """auto_pick_ee6d.py --stdin with the ROS python: one target per line, one "@@" reply per finished cycle."""

    def __init__(self):
        self.p = None
        self.replies = queue_mod.Queue()
        self.lock = threading.Lock()                                 # stdin: pick loop + window (home_now)

    def _send(self, obj):
        with self.lock:
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()

    def start(self):
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
        cmd = f"{ROS_SETUP} && /usr/bin/python3 -u {Path(__file__).resolve().parent / 'auto_pick_ee6d.py'} --stdin --ee6d {EE_6D}"
        self.p = subprocess.Popen(["bash", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()
        r = self.reply()
        if not r.get("ready"):
            raise RuntimeError(f"robot not ready: {r.get('error')}")
        log("robot ready (terminal 1 + 2 found)")

    def _pump(self):
        for line in self.p.stdout:
            if line.startswith("@@ "):
                self.replies.put(json.loads(line[3:]))
            elif line.strip():
                log("[robot] " + line.rstrip())
        self.replies.put(None)

    def reply(self):
        r = self.replies.get()
        if r is None:
            raise RuntimeError("auto_pick_ee6d.py exited (see the [robot] lines)")
        return r

    def pick(self, q):
        self._send({"rank": q["name"], **target_of(q["T_tcp"])})
        return self.reply()

    def home(self):
        """Arm to home through sequence_real_action's /go_home (only when it is idle; nothing sent when already home)."""
        self._send({"cmd": "home"})
        return self.reply()

    def home_now(self):
        """Pause button: /go_home right away, also during a pick (the approach is aborted, a grasped part is placed
        first). No reply: the running pick() returns "aborted" or "done"."""
        if self.p is not None and self.p.poll() is None:
            self._send({"cmd": "home_now"})

    def check(self, queue):
        """{pick name: (ok, reason)}: the IK sequence_real_action will need (pre-grasp, dive) via /compute_ik."""
        self._send({"cmd": "check", "targets": [{"rank": q["name"], **target_of(q["T_tcp"])} for q in queue]})
        return {r["rank"]: (r["ok"], r["reason"]) for r in self.reply()["results"]}

    def close(self, wait=5.0):
        """quit: auto_pick_ee6d.py sends nothing more; a cycle already sent still finishes in sequence_real_action."""
        if self.p is None:
            return
        if self.p.poll() is None:
            try:
                with self.lock:
                    self.p.stdin.write("quit\n")
                    self.p.stdin.flush()
                    self.p.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            self.p.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            return                                                   # still in a cycle: leave it, it quits after
        self.pump.join(timeout=1.0)


class DryRobot:
    """--no-robot: logs the target that would be sent and waits instead of a robot cycle."""

    def __init__(self, seconds):
        self.seconds = seconds

    def start(self):
        log(f"--no-robot: nothing is sent, each pick waits {self.seconds:g} s")

    def pick(self, q):
        (x, y, z), (qx, qy, qz, qw) = q["T_tcp"][:3, 3], Rotation.from_matrix(q["T_tcp"][:3, :3]).as_quat()
        log(f"[dry] #{q['name']}: ee_6d quat {x:.4f} {y:.4f} {z:.4f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
        time.sleep(self.seconds)
        return {"rank": q["name"], "status": "done", "seconds": self.seconds}

    def home(self):
        return {"cmd": "home", "status": "home", "seconds": 0.0}

    def home_now(self):
        pass

    def check(self, queue):
        return {q["name"]: (True, "") for q in queue}

    def close(self):
        pass


# ------------------------------------------------------------------------------------------------------- pick loop
class PickLoop(threading.Thread):
    """3D scan -> picks one at a time with a 2D check after each -> new 3D scan when none is left, until stopped.
    Pause and rescan send the arm home at once (a running approach is aborted); stop takes effect between robot cycles.
    Rescan drops the queue, takes a new 3D scan and pauses there: start (resume) picks from the new scan."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.stop_ev, self.pause_ev, self.rescan_ev = threading.Event(), threading.Event(), threading.Event()
        self.hold_ev = threading.Event()                             # rescan button: pause after the new scan
        self.pause_text = None                                       # status line of the next pause
        self.robot = None
        self.robot_busy = False                                      # arm never came to rest: no home move at the end
        self.t_start = None
        self.stats = {"done": 0, "failed": 0, "unreachable": 0, "scans": 0, "cycles": []}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_csv = None if args.no_save else OUT_DIR / f"session_{stamp}.csv"

    def run(self):
        try:
            self._run()
        except Exception as e:
            log(traceback.format_exc())
            state("error", f"오류: {e}")
        else:
            state("stopped", "정지됨")
        finally:
            if self.robot is not None:
                if not self.robot_busy:
                    try:
                        self._home("at the end", strict=False)
                    except Exception as e:
                        log(f"home at the end: {e}")
                self.robot.close()
            post("loop_end")

    def _home(self, why, strict=True):
        """Arm to home before a capture (it must be out of the camera view) and at start / end.
        sequence_real_action does the move and only when idle, so its own home moves are never overlapped."""
        state("home", f"home으로 이동 ({why})")
        r = self.robot.home()
        if r["status"] == "done":
            log(f"home ({why}): reached in {r['seconds']} s")
        elif r["status"] != "home":
            msg = f"home ({why}): {r['status']} (see terminal 2)"
            if r["status"] == "busy":
                self.robot_busy = True
            if strict:
                raise RuntimeError(msg + ": the arm may be in the camera view, stopped")
            log("WARNING: " + msg)

    def _reachable(self, queue):
        """Queue without the picks the robot cannot reach (IK of pre-grasp or dive fails in terminal 1's MoveIt)."""
        if not queue:
            return queue
        state("ik", f"IK 확인 ({len(queue)}개)")
        res = self.robot.check(queue)
        for q in queue:
            ok, why = res[q["name"]]
            if not ok:
                self.stats["unreachable"] += 1
                log(f"pick {q['name']}: unreachable, skipped ({why})")
                post("dropped", [(q["name"], "unreachable: " + why)])
        self._post_stats()
        return [q for q in queue if res[q["name"]][0]]

    def _post_stats(self):
        c = self.stats["cycles"]
        post("stats", {"done": self.stats["done"], "failed": self.stats["failed"],
                       "unreachable": self.stats["unreachable"], "scans": self.stats["scans"],
                       "cycle": float(np.mean(c[-10:])) if c else None, "t_start": self.t_start})

    def _hold(self):
        """Wait while paused. Returns (stop requested, was paused)."""
        if not self.pause_ev.is_set():
            return self.stop_ev.is_set(), False
        self._home("pause", strict=False)                            # wait at the ready pose, out of the camera view
        state("paused", self.pause_text or "일시정지 (재개: 시작)")
        self.pause_text = None
        log("paused")
        while self.pause_ev.is_set() and not self.stop_ev.is_set():
            self.stop_ev.wait(0.1)
        if not self.stop_ev.is_set():
            log("resumed")
        return self.stop_ev.is_set(), True

    def _idle(self, seconds):
        """Wait before the next scan; a rescan request ends it early. Returns True when stop was requested."""
        t_end = time.time() + seconds
        while not (self.stop_ev.is_set() or self.rescan_ev.is_set()) and time.time() < t_end:
            self.stop_ev.wait(0.1)
        if self.rescan_ev.is_set() and not self.stop_ev.is_set():
            log("rescan requested")
        return self.stop_ev.is_set()

    def _record(self, q, r):
        if self.session_csv is None:
            return
        new = not self.session_csv.exists()
        with open(self.session_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "pick", "status", "cycle_s", "correction_2d_mm", "tcp_x_m", "tcp_y_m", "tcp_z_m"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), q["name"], r["status"], r["seconds"],
                        round(q["shift (mm)"], 1), *np.round(q["T_tcp"][:3, 3], 4)])

    def _run(self):
        a = self.args
        self.t_start = time.time()
        self._post_stats()
        state("connect", "로봇 연결 중")
        self.robot = DryRobot(a.dry_seconds) if a.no_robot else Robot()
        self.robot.start()
        self._home("start")
        while True:
            stop, _ = self._hold()
            if stop:
                return
            self._home("before the 3D scan")
            state("scan", "3D 촬영 + pose 추정")
            self.rescan_ev.clear()
            res = scan_3d(save=not a.no_save)
            self.stats["scans"] += 1
            scan = self.stats["scans"]
            s = res["summary"]
            post("scan", s)
            post("scan_result", scan, res)
            post("image", res["image"], f"3D 촬영 {scan}: 선택 {s['selected']}개")
            queue = pick_queue(res["picks"], res["dets"], scan) if not res["picks"].empty else []
            queue = self._reachable(queue)
            post("queue", queue_rows(queue))
            self._post_stats()
            if self.hold_ev.is_set():                                # rescan button: show the new scan, pick after start
                self.hold_ev.clear()
                self.rescan_ev.clear()                               # this scan answers a press during the scan too
                self.pause_text = f"다시 촬영 완료: pick {len(queue)}개 (시작: 집기 시작)"
                self.pause_ev.set()
            if not queue:
                if getattr(a, "stop_when_empty", False):             # the notebook's auto_pick_2d: done
                    log("a new 3D scan found nothing graspable -> done")
                    return
                if self.pause_ev.is_set():                           # wait for start, then scan again
                    continue
                state("idle", f"잡을 부품 없음: {a.idle:.0f} s 뒤 다시 촬영")
                if self._idle(a.idle):
                    return
                continue
            frame, ref = res["frame"], res["frame"].get("gray2d")
            checked = True                                           # the scan is fresh: no 2D check before pick 1
            while queue:
                stop, paused = self._hold()
                if stop:
                    return
                checked = checked and not paused                     # the bin may have changed while paused
                if self.rescan_ev.is_set():
                    log("rescan requested")
                    break
                if a.max_picks is not None and self.stats["done"] >= a.max_picks:
                    log(f"{a.max_picks} parts picked (--max-picks)")
                    return
                if not checked and ref is not None:
                    self._home("before the 2D check")
                    state("check", "2D 확인 (재개 전)")
                    queue, ref = check_2d(queue, ref, frame, "after the pause")
                    checked = True
                    continue
                q = queue.pop(0)
                if q["shift (mm)"] > 0 and not self._reachable([q]):     # moved by the 2D correction: check again
                    post("queue", queue_rows(queue))
                    continue
                state("pick", f"#{q['name']} 집는 중 (남은 pick {len(queue)}개)")
                post("image", draw_2d(ref if ref is not None else frame["rgb"], queue, frame, current=q),
                     f"#{q['name']} 집는 중")
                post("queue", queue_rows(queue))
                post("pick_start", q["name"])
                r = self.robot.pick(q)
                post("pick_result", q["name"], r["status"], r["seconds"])
                self._record(q, r)
                if r["status"] == "aborted":                         # paused / rescan mid-cycle
                    if self.rescan_ev.is_set():
                        log(f"pick {q['name']}: aborted for the rescan, back at home ({r['seconds']} s)")
                    else:
                        log(f"pick {q['name']}: aborted by the pause, back at home ({r['seconds']} s); "
                            "retried after resume")
                    queue.insert(0, q)
                    post("queue", queue_rows(queue))
                    continue
                if r["status"] == "done":
                    self.stats["done"] += 1
                    self.stats["cycles"].append(r["seconds"])
                elif r["status"] != "busy":
                    self.stats["failed"] += 1
                log(f"pick {q['name']}: {r['status']} ({r['seconds']} s, 2D correction {q['shift (mm)']:.1f} mm)")
                self._post_stats()
                if r["status"] == "busy":
                    self.robot_busy = True
                    raise RuntimeError("the arm does not come to rest (see terminal 2): stopped")
                if self.stop_ev.is_set():
                    return
                if a.max_picks is not None and self.stats["done"] >= a.max_picks:   # no extra scan / 2D check
                    log(f"{a.max_picks} parts picked (--max-picks)")
                    return
                if ref is None:                                      # no 2D camera (--offline): new 3D scan instead
                    break
                # one part moved: look at the bin again with the 2D camera and correct / drop the rest
                self._home("before the 2D check")                    # after "done" the arm is home: nothing sent
                state("check", "2D 확인")
                queue, ref = check_2d(queue, ref, frame, f"after {q['name']}")
            if queue == [] and not self.rescan_ev.is_set():
                log("no graspable pick left -> new 3D scan")


# -------------------------------------------------------------------------------------------------------------- UI
def run_ui(args):
    import tkinter as tk
    from tkinter import font as tkfont, messagebox, ttk
    from PIL import ImageTk

    BG, PANEL, CARD, LINE = "#0e1217", "#151b23", "#1c2530", "#2a3542"
    TEXT, MUTED = "#e6edf3", "#8b98a5"
    GREEN, AMBER, RED, BLUE, CYAN, GREY = "#2ea86f", "#d99a1e", "#d9534f", "#3b82f6", "#14a3b8", "#5b6673"
    STATE = {"init": ("준비 중", BLUE), "ready": ("대기", GREY), "connect": ("연결", BLUE), "scan": ("3D 촬영", BLUE),
             "home": ("home 이동", BLUE), "ik": ("IK 확인", CYAN), "pick": ("집는 중", GREEN), "check": ("2D 확인", CYAN), "idle": ("부품 없음", AMBER),
             "paused": ("일시정지", AMBER), "stopped": ("정지", GREY), "error": ("오류", RED)}

    root = tk.Tk()
    root.title("GDRNPP Bin Picking · Lite6")
    root.configure(bg=BG)
    root.geometry("1600x960")
    root.minsize(1200, 760)
    fams = set(tkfont.families())
    FONT = next((f for f in ("NanumBarunGothic", "NanumGothic", "Noto Sans CJK KR") if f in fams), "TkDefaultFont")
    MONO = next((f for f in ("NanumGothicCoding", "DejaVu Sans Mono") if f in fams), "TkFixedFont")

    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=26,
                 bordercolor=LINE, borderwidth=0, font=(FONT, 10))
    st.configure("Treeview.Heading", background=CARD, foreground=MUTED, relief="flat", font=(FONT, 10, "bold"))
    st.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])          # no light frame around the table
    st.map("Treeview", background=[("selected", "#24476b")])
    st.map("Treeview.Heading", background=[("active", CARD)])
    st.configure("Vertical.TScrollbar", background=CARD, troughcolor=PANEL, bordercolor=PANEL, arrowcolor=MUTED)

    def card(parent, **kw):
        return tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE, **kw)

    # header
    head = tk.Frame(root, bg=BG)
    head.pack(fill="x", padx=16, pady=(14, 8))
    tk.Label(head, text="GDRNPP Bin Picking", bg=BG, fg=TEXT, font=(FONT, 20, "bold")).pack(side="left")
    mode = ("오프라인 프레임" if args.offline else "Photoneo") + " · " + ("로봇 없음 (dry run)" if args.no_robot else "Lite6")
    tk.Label(head, text="  " + mode, bg=BG, fg=MUTED, font=(FONT, 12)).pack(side="left", pady=(6, 0))
    clock = tk.Label(head, text="00:00:00", bg=BG, fg=MUTED, font=(MONO, 16))
    clock.pack(side="right")
    status = tk.Label(head, text="", bg=BG, fg=TEXT, font=(FONT, 13))
    status.pack(side="right", padx=14)
    badge = tk.Label(head, text="", bg=BLUE, fg="white", font=(FONT, 12, "bold"), padx=14, pady=4)
    badge.pack(side="right")

    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=16)
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2, minsize=520)
    body.rowconfigure(0, weight=1)

    # left: image
    left = card(body)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    caption = tk.Label(left, text="", bg=CARD, fg=TEXT, font=(FONT, 12, "bold"), anchor="w")
    caption.pack(fill="x", padx=12, pady=(10, 4))
    view = tk.Label(left, bg="#0a0d11")
    view.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # right: stats, scan funnel, queue, buttons
    right = tk.Frame(body, bg=BG)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(2, weight=1)
    right.columnconfigure(0, weight=1)
    grid = tk.Frame(right, bg=BG)
    grid.grid(row=0, column=0, sticky="ew")
    stat_vals = {}
    for i, (key, name) in enumerate((("done", "집은 부품"), ("failed", "실패 / IK로 제외"), ("rate", "시간당 pick"),
                                     ("scans", "3D 촬영"), ("cycle", "평균 사이클"), ("left", "남은 pick"))):
        c = card(grid)
        c.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 8, 0), pady=(0, 8))
        grid.columnconfigure(i % 3, weight=1)
        tk.Label(c, text=name, bg=CARD, fg=MUTED, font=(FONT, 10)).pack(anchor="w", padx=12, pady=(8, 0))
        stat_vals[key] = tk.Label(c, text="-", bg=CARD, fg=TEXT, font=(FONT, 24, "bold"))
        stat_vals[key].pack(anchor="w", padx=12, pady=(0, 8))
    stat_vals["done"].configure(fg=GREEN)
    stat_vals["failed"].configure(fg=RED)

    fc = card(right)
    fc.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    tk.Label(fc, text="마지막 3D 촬영", bg=CARD, fg=MUTED, font=(FONT, 10)).pack(anchor="w", padx=12, pady=(8, 0))
    funnel = tk.Label(fc, text="-", bg=CARD, fg=TEXT, font=(FONT, 12), anchor="w", justify="left")
    funnel.pack(fill="x", padx=12, pady=(2, 10))

    qc = card(right)
    qc.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
    tk.Label(qc, text="다음 pick 순서 (robot base, m)", bg=CARD, fg=MUTED, font=(FONT, 10)).pack(anchor="w", padx=12, pady=(8, 4))
    cols = (("n", "순서", 44), ("pick", "pick", 64), ("x", "tcp x", 74), ("y", "tcp y", 74), ("z", "tcp z", 74),
            ("tilt", "기울기°", 66), ("match", "2D 일치", 66), ("shift", "보정 mm", 70))
    tree = ttk.Treeview(qc, columns=[c[0] for c in cols], show="headings", height=10)
    for key, name, w in cols:
        tree.heading(key, text=name)
        tree.column(key, width=w, anchor="center", stretch=True)
    tree.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    bar = tk.Frame(right, bg=BG)
    bar.grid(row=3, column=0, sticky="ew", pady=(0, 4))

    def button(text, colour, cmd):
        b = tk.Button(bar, text=text, command=cmd, bg=colour, fg="white", activebackground=colour,
                      activeforeground="white", disabledforeground="#c9d1d9", relief="flat", bd=0,
                      font=(FONT, 13, "bold"), padx=10, pady=10, cursor="hand2", highlightthickness=0)
        b.pack(side="left", fill="x", expand=True, padx=(0, 8))
        return b

    tk.Label(right, text="일시정지: 바로 home으로 (잡기 전이면 접근을 멈추고, 잡은 뒤면 놓고 감). "
                         "다시 촬영: 바로 home으로 가서 (일시정지와 같음) 남은 pick을 버리고 3D로 다시 찍은 뒤 멈춤, "
                         "시작을 누르면 새 pick으로 집음. "
                         "정지: 지금 사이클이 끝난 뒤. "
                         "로봇을 바로 멈추려면 terminal 2에서 Ctrl+C.",
             bg=BG, fg=MUTED, font=(FONT, 9), wraplength=520, justify="left").grid(row=4, column=0, sticky="w")

    # log
    lc = card(root)
    lc.pack(fill="x", padx=16, pady=(10, 14))
    logbox = tk.Text(lc, height=9, bg=PANEL, fg="#c9d1d9", insertbackground=TEXT, relief="flat", font=(MONO, 10),
                     wrap="none", highlightthickness=0, padx=10, pady=6)
    sb = ttk.Scrollbar(lc, orient="vertical", command=logbox.yview)
    logbox.configure(yscrollcommand=sb.set, state="disabled")
    sb.pack(side="right", fill="y")
    logbox.pack(fill="both", expand=True)
    logbox.tag_configure("warn", foreground=AMBER)
    logbox.tag_configure("err", foreground=RED)
    logbox.tag_configure("ok", foreground=GREEN)
    logbox.tag_configure("robot", foreground="#7aa2f7")

    ui = {"loop": None, "img": None, "tk": None, "ready": False, "t_start": None, "state": "init", "closing": False}

    def set_state(key, text=""):
        ui["state"] = key
        name, colour = STATE[key]
        badge.configure(text=name, bg=colour)
        status.configure(text=text)
        refresh_buttons()

    def show_image(img=None):
        if img is not None:
            ui["img"] = img
        if ui["img"] is None:
            return
        w, h = max(view.winfo_width() - 4, 50), max(view.winfo_height() - 4, 50)
        im = ui["img"]
        s = min(w / im.width, h / im.height)
        ui["tk"] = ImageTk.PhotoImage(im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)), Image.BILINEAR))
        view.configure(image=ui["tk"])

    def add_log(msg):
        tag = ("err" if ("Error" in msg or "Traceback" in msg or "failed" in msg) else
               "warn" if ("WARNING" in msg or "dropped" in msg or "not started" in msg) else
               "ok" if (": done" in msg or "saved ->" in msg) else "robot" if msg.startswith("[robot]") else "")
        logbox.configure(state="normal")
        logbox.insert("end", msg + "\n", tag)
        if int(logbox.index("end-1c").split(".")[0]) > 3000:
            logbox.delete("1.0", "500.0")
        logbox.see("end")
        logbox.configure(state="disabled")

    def running():
        return ui["loop"] is not None and ui["loop"].is_alive()

    def on_start(hold=False):
        if running():
            ui["loop"].pause_ev.clear()
            return
        ui["loop"] = PickLoop(args)
        if hold:
            ui["loop"].hold_ev.set()
        ui["loop"].start()
        refresh_buttons()

    def on_pause():
        if running():
            ui["loop"].pause_ev.set()
            if ui["loop"].robot is not None:
                ui["loop"].robot.home_now()                          # at once, also in the middle of a pick
            status.configure(text="home으로 이동 후 일시정지 (잡은 부품이 있으면 놓고 감)")

    def on_rescan():
        # home at once, drop the queue, new 3D scan, then wait: start picks from the new scan
        if not running():
            on_start(hold=True)
            return
        loop = ui["loop"]
        loop.hold_ev.set()
        loop.rescan_ev.set()
        loop.pause_ev.clear()                                        # paused: go on to the new scan
        if loop.robot is not None:
            loop.robot.home_now()                                    # at once: a running approach is aborted
        status.configure(text="home으로 이동 후 다시 3D 촬영, 시작을 누르면 집기 (잡은 부품이 있으면 놓고 감)")
        refresh_buttons()

    def on_stop():
        if running():
            ui["loop"].stop_ev.set()
            ui["loop"].pause_ev.clear()
            status.configure(text="이번 사이클이 끝나면 정지")

    b_start = button("▶  시작", GREEN, on_start)
    b_pause = button("❚❚  일시정지", AMBER, on_pause)
    b_rescan = button("↻  다시 촬영", BLUE, on_rescan)
    b_stop = button("■  정지", RED, on_stop)

    def refresh_buttons():
        run = running()
        paused = run and ui["loop"].pause_ev.is_set()
        b_start.configure(state="normal" if ui["ready"] and (not run or paused) else "disabled",
                          text="▶  재개" if paused else "▶  시작")
        for b in (b_pause, b_stop):
            b.configure(state="normal" if run else "disabled")
        b_rescan.configure(state="normal" if run or ui["ready"] else "disabled")
        for b in (b_start, b_pause, b_rescan, b_stop):
            b.configure(bg=b.cget("activebackground") if b.cget("state") == "normal" else "#2a323c")

    def tick():
        t0 = ui["t_start"]
        if t0 is not None and running():
            e = int(time.time() - t0)
            clock.configure(text=f"{e // 3600:02d}:{e % 3600 // 60:02d}:{e % 60:02d}")
            done = int(stat_vals["done"].cget("text")) if stat_vals["done"].cget("text").isdigit() else 0
            stat_vals["rate"].configure(text=f"{done / max(e, 1) * 3600:.0f}" if e > 60 else "-")

    def poll():
        try:
            while True:
                kind, a = UI.get_nowait()
                if kind == "log":
                    add_log(a[0])
                elif kind == "state":
                    set_state(*a)
                elif kind == "image":
                    show_image(a[0])
                    caption.configure(text=a[1])
                elif kind == "queue":
                    tree.delete(*tree.get_children())
                    for row in a[0]:
                        tree.insert("", "end", values=row)
                    stat_vals["left"].configure(text=str(len(a[0])))
                elif kind == "stats":
                    s = a[0]
                    ui["t_start"] = s["t_start"]
                    stat_vals["done"].configure(text=str(s["done"]))
                    stat_vals["failed"].configure(text=f"{s['failed']} / {s['unreachable']}")
                    stat_vals["scans"].configure(text=str(s["scans"]))
                    stat_vals["cycle"].configure(text=f"{s['cycle']:.0f} s" if s["cycle"] else "-")
                elif kind == "scan":
                    s = a[0]
                    funnel.configure(text=f"마스크 {s['masks']} → 검증 {s['verified']} → confident {s['confident']} → "
                                          f"잡기 가능 {s['graspable']} → 선택 {s['selected']}\n"
                                          f"촬영 {s['capture (s)']:.1f} s · pose {s['pose (s)']:.1f} s")
                elif kind == "ready":
                    ui["ready"] = True
                    set_state("ready", "시작을 누르면 촬영과 집기를 반복합니다")
                    if args.autostart:
                        on_start()
                elif kind == "loop_end":
                    if ui["closing"]:
                        CAM.close()
                        root.destroy()
                        return
                    refresh_buttons()
                    if ui["state"] not in ("error",):
                        set_state("stopped", "정지됨. 시작을 누르면 다시 돕니다")
        except queue_mod.Empty:
            pass
        tick()
        root.after(100, poll)

    def on_close():
        if running() and not ui["closing"]:
            if not messagebox.askokcancel("종료", "지금 사이클이 끝나면 로봇을 home으로 보낸 뒤 창을 닫습니다.\n"
                                                 "종료할까요?"):
                return
            ui["closing"] = True                                     # loop_end closes the window
            on_stop()
            status.configure(text="이번 사이클 + home 이동 뒤 종료 (한 번 더 닫으면 바로 종료)")
            return
        if ui["loop"] is not None and ui["loop"].robot is not None:
            ui["loop"].robot.close()
        CAM.close()
        root.destroy()

    view.bind("<Configure>", lambda e: show_image())
    root.protocol("WM_DELETE_WINDOW", on_close)
    signal.signal(signal.SIGINT, lambda *_: root.after(0, on_close))
    set_state("init", "모델 로딩 + 카메라 연결 중")

    def init():
        try:
            load_models()
            if OFFLINE is None:
                CAM.start()
            post("ready")
        except Exception as e:
            log(traceback.format_exc())
            post("state", "error", f"초기화 실패: {e}")

    threading.Thread(target=init, daemon=True).start()
    root.after(100, poll)
    root.mainloop()


def main():
    global OFFLINE, UI
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-robot", action="store_true", help="do not send anything to the robot (dry run)")
    p.add_argument("--offline", nargs=2, metavar=("BOP_ROOT", "VIEW"), help="saved BOP frame instead of the camera")
    p.add_argument("--max-picks", type=int, help="stop after this many picked parts (default: no limit)")
    p.add_argument("--idle", type=float, default=10.0, help="s between 3D scans while nothing is graspable")
    p.add_argument("--dry-seconds", type=float, default=3.0, help="s per pick with --no-robot")
    p.add_argument("--no-save", action="store_true", help="do not write runs/realtime/<time>/ and the session csv")
    p.add_argument("--no-ui", action="store_true", help="console only")
    p.add_argument("--autostart", action="store_true", help="start picking as soon as the models are loaded")
    args = p.parse_args()
    if args.offline:
        OFFLINE = (args.offline[0], int(args.offline[1]))
    if not args.no_ui:
        UI = queue_mod.Queue()
        run_ui(args)
        return
    load_models()
    loop = PickLoop(args)
    try:
        if OFFLINE is None:
            CAM.start()
        if not args.autostart:
            input("Enter = start picking (Ctrl+C stops after the running cycle) ")
        loop.run()
    except KeyboardInterrupt:
        loop.stop_ev.set()
        log("\ninterrupted: no more targets are sent (the running cycle finishes in sequence_real_action)")
    finally:
        if loop.robot is not None:
            loop.robot.close()
        CAM.close()


if __name__ == "__main__":
    main()
