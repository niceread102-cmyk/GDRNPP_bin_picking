#!/usr/bin/env python3
"""Capture RGB / depth / point cloud from a Photoneo camera over GigE Vision (harvesters).

Uses the `photoneo_genicam` helpers from photoneo-python-examples/GigE-V/harvesters/advanced and writes

    camera/
        camera.json                 fx, fy, cx, cy, width, height, depth_scale (same format as shard camera.json)
        rgb/<idx>.png               uint8 RGB
        depth/<idx>.png             uint16, * depth_scale -> mm (0 = invalid)
        pointcloud/<idx>.ply        camera frame, mm, valid points only, with per-point colour
        undistort_map.npy           (H, W, 2) float32: ideal pinhole pixel (u, v) of every sensor pixel (lens distortion)

Default = the camera the models were trained for (sim and data/pseudo_gt_20_bop): primary camera space at
1236x1032 (fx ~1412) with the colour texture mapped onto it, so rgb, depth and the point cloud are pixel-aligned.
The point cloud uses the device coordinate maps (ProjectedC * CoordinateMap), so it is distortion-free; rgb/depth are
raw sensor pixels (distorted, up to ~15 px at the image corners) and --to-bop resamples them onto the pinhole K.

Run with the harvesters venv, which has harvesters/genicam installed:
    ~/Projects/photoneo-python-examples/GigE-V/harvesters/.venv/bin/python camera_convert_data.py [serial] \
        [--count N | --interactive]
Convert the captures to the BOP layout read by infer.py / bin_dataset.py (no GT; any python with numpy + PIL):
    .venv/bin/python camera_convert_data.py --to-bop data/real_capture_bop
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXAMPLES_DIR = Path.home() / "Projects/photoneo-python-examples/GigE-V/harvesters/advanced"
DEFAULT_OUT_DIR = PROJECT_DIR / "camera"
DEFAULT_MODELS_DIR = PROJECT_DIR / "data/pseudo_gt_70_bop/models"
DEPTH_SCALE = 0.1  # stored uint16 * DEPTH_SCALE = mm  (max 6553.5 mm)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("serial", nargs="?", help="device serial number (default: first device found)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output dir (default: ./camera)")
    p.add_argument("--count", type=int, default=1, help="number of frames to capture")
    p.add_argument("--interactive", action="store_true", help="press Enter to capture, q + Enter to quit")
    p.add_argument("--camera-space", default="PrimaryCamera", choices=["PrimaryCamera", "ColorCamera"],
                   help="PrimaryCamera = training camera (default); ColorCamera = native colour camera resolution")
    p.add_argument("--resolution", default="1236x1032",
                   help="primary camera resolution (default 1236x1032 = training data); '' keeps the device value")
    p.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR,
                   help="photoneo-python-examples GigE-V/harvesters/advanced (contains photoneo_genicam)")
    p.add_argument("--timeout", type=float, default=10.0, help="fetch timeout [s]")
    p.add_argument("--to-bop", type=Path, metavar="DST",
                   help="no capture: convert --out captures to a BOP dataset at DST (rewrites DST/train_real)")
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS_DIR, help="CAD models dir copied by --to-bop")
    p.add_argument("--keep-distortion", action="store_true", help="--to-bop: copy raw images (no undistortion)")
    return p.parse_args()


def texture_to_rgb(component) -> np.ndarray:
    """Texture component -> (H, W, 3) uint8 RGB."""
    h, w, fmt = component.height, component.width, component.data_format
    data = component.data.copy()
    if fmt == "RGB8":
        return data.reshape(h, w, 3)
    shift = {"Mono8": 0, "Mono10": 2, "Mono12": 4, "Mono16": 8}.get(fmt)
    if shift is None:
        raise ValueError(f"Unsupported texture pixel format: {fmt}")
    mono = (data.astype(np.uint16) >> shift).clip(0, 255).astype(np.uint8).reshape(h, w)
    return np.repeat(mono[..., None], 3, axis=-1)


def depth_to_png(depth_mm: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.where(d > 0, d / DEPTH_SCALE, 0)
    return np.round(d).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Binary little-endian PLY, float xyz + uchar rgb."""
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points.T.astype(np.float32)
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


def next_index(out: Path) -> int:
    ids = [int(p.stem) for p in (out / "rgb").glob("*.png") if p.stem.isdigit()]
    return max(ids) + 1 if ids else 0


def undistort_lookup(ab: np.ndarray, iters: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Invert the sensor -> pinhole pixel map `ab` (H, W, 2): for each pinhole pixel, the sensor pixel (row, col)
    to sample (fixed-point iteration, the distortion is smooth) and whether it lies on the sensor."""
    from scipy.ndimage import map_coordinates
    h, w, _ = ab.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    du, dv = ab[..., 0] - u, ab[..., 1] - v
    s, t = u.copy(), v.copy()
    for _ in range(iters):
        s = u - map_coordinates(du, [t, s], order=1, mode="nearest")
        t = v - map_coordinates(dv, [t, s], order=1, mode="nearest")
    inside = (s >= 0) & (s <= w - 1) & (t >= 0) & (t <= h - 1)
    return np.stack([t, s]), inside


def undistort(img: np.ndarray, rc: np.ndarray, inside: np.ndarray, order: int) -> np.ndarray:
    """Resample an (H, W[, C]) sensor image onto the pinhole grid; order 0 for depth (no mixing across edges)."""
    from scipy.ndimage import map_coordinates
    chans = img[..., None] if img.ndim == 2 else img
    out = np.stack([map_coordinates(chans[..., c].astype(np.float32), rc, order=order, mode="constant")
                    for c in range(chans.shape[-1])], axis=-1)
    out[~inside] = 0
    out = np.round(out).astype(img.dtype) if order else out.astype(img.dtype)
    return out[..., 0] if img.ndim == 2 else out


def export_bop(src: Path, dst: Path, models_dir: Path = DEFAULT_MODELS_DIR, scene: str = "000000",
               undistorted: bool = True) -> list[int]:
    """camera/ captures -> BOP layout of data/pseudo_gt_20_bop without labels (empty scene_gt, zero masks).

    Views are renumbered 0..N-1 in capture order; dst/sources.json maps view -> capture file name.
    Frames whose size differs from camera.json are skipped. Returns the capture ids that were exported.
    undistorted: resample rgb/depth onto the pinhole K with camera/undistort_map.npy (the pipeline back-projects
    depth with K only); False copies the raw sensor images.
    """
    src, dst = Path(src), Path(dst)
    cam = json.load(open(src / "camera.json"))
    lookup = None
    if undistorted:
        # camera.json names the map written in the same session; an older capture script leaves a stale map behind
        if "undistort_map" not in cam:
            raise FileNotFoundError(f"{src}/camera.json has no undistort_map (captured with an older script?): "
                                    "recapture with camera_convert_data.py, or pass undistorted=False")
        ab = np.load(src / cam["undistort_map"])
        if ab.shape[:2] != (cam["height"], cam["width"]):
            raise ValueError(f"undistort map {ab.shape[:2]} does not match camera.json {cam['height']}x{cam['width']}")
        lookup = undistort_lookup(ab)
    K = [cam["fx"], 0.0, cam["cx"], 0.0, cam["fy"], cam["cy"], 0.0, 0.0, 1.0]
    out = dst / "train_real" / scene
    if out.exists():
        shutil.rmtree(out)
    for sub in ("rgb", "depth", "mask_visib_inst"):
        (out / sub).mkdir(parents=True)
    if not (dst / "models").exists():
        shutil.copytree(models_dir, dst / "models")

    ids = sorted(int(p.stem) for p in (src / "rgb").glob("*.png") if p.stem.isdigit()
                 and (src / "depth" / p.name).exists())
    gt, info, cams, sources, kept = {}, {}, {}, {}, []
    for cid in ids:
        name = f"{cid:06d}.png"
        rgb = Image.open(src / "rgb" / name)
        if rgb.size != (cam["width"], cam["height"]):
            print(f"skip {name}: {rgb.size[0]}x{rgb.size[1]} != camera.json {cam['width']}x{cam['height']}")
            continue
        v = len(kept)
        if lookup is None:
            rgb.convert("RGB").save(out / "rgb" / f"{v:06d}.png")
            shutil.copyfile(src / "depth" / name, out / "depth" / f"{v:06d}.png")
        else:
            Image.fromarray(undistort(np.asarray(rgb.convert("RGB")), *lookup, order=1)).save(out / "rgb" / f"{v:06d}.png")
            depth = np.asarray(Image.open(src / "depth" / name), dtype=np.uint16)
            Image.fromarray(undistort(depth, *lookup, order=0)).save(out / "depth" / f"{v:06d}.png")
        Image.fromarray(np.zeros((cam["height"], cam["width"]), np.uint16)).save(out / "mask_visib_inst" / f"{v:06d}.png")
        gt[str(v)], info[str(v)] = [], []
        cams[str(v)] = {"cam_K": K, "depth_scale": cam["depth_scale"]}
        sources[str(v)] = {"capture": name, "undistorted": lookup is not None}
        kept.append(cid)
    for fname, data in (("scene_gt.json", gt), ("scene_gt_info.json", info), ("scene_camera.json", cams)):
        (out / fname).write_text(json.dumps(data, indent=1))
    (dst / "sources.json").write_text(json.dumps(sources, indent=1))
    return kept


def capture(args) -> int:
    sys.path.insert(0, str(args.examples_dir))
    os.environ.setdefault("XDG_SESSION_TYPE", "x11")

    from genicam.genapi import NodeMap
    from harvesters.core import Harvester

    from photoneo_genicam.components import enable_components, enabled_components
    from photoneo_genicam.default_gentl_producer import producer_path
    from photoneo_genicam.features import enable_software_trigger
    from photoneo_genicam.pointcloud import calculate_point_cloud_from_projc, pre_fetch_coordinate_maps
    from photoneo_genicam.user_set import load_default_user_set
    from photoneo_genicam.utils import DeviceType, data_stream_reset, detect_device_type, logger

    for sub in ("rgb", "depth", "pointcloud"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    with Harvester() as h:
        h.add_file(str(producer_path), check_existence=True, check_validity=True)
        h.update()
        if not h.device_info_list:
            logger.error("No GigE Vision device found.")
            return 1
        serial = args.serial or h.device_info_list[0].serial_number
        logger.info(f"Connecting to: {serial}")

        with h.create({"serial_number": serial}) as ia:
            features: NodeMap = ia.remote_device.node_map
            logger.info(f"Device Firmware version: {features.DeviceFirmwareVersion.value}")

            load_default_user_set(features)
            enable_software_trigger(features)

            device_type = detect_device_type(features)
            is_color = device_type in (DeviceType.MOTIONCAM_3D_COLOR, DeviceType.PHOXI_SCANNER_GEN3)
            if args.resolution:
                res = f"Resolution_{args.resolution}"
                if res in features.Resolution.symbolics:
                    features.Resolution.value = res
                else:
                    logger.warning(f"{res} not available ({features.Resolution.symbolics}), keeping "
                                   f"{features.Resolution.value}")
            if is_color:
                features.CameraSpace.value = args.camera_space

            # Coordinate maps and intrinsics are those of the selected camera space / resolution.
            logger.info("Pre-fetch CoordinateMaps")
            coord_map = pre_fetch_coordinate_maps(ia)
            f = features.Scan3dFocalLength.value
            intr = {"cx": features.Scan3dPrincipalPointU.value, "cy": features.Scan3dPrincipalPointV.value,
                    "fx": f, "fy": f * features.Scan3dAspectRatio.value}

            enable_components(features, ["Intensity", "Range"])
            if is_color:
                ts_node = "TextureSource" if device_type == DeviceType.PHOXI_SCANNER_GEN3 else "CameraTextureSource"
                features.get_node(ts_node).value = "Color"
                features.ComponentSelector.value = "Intensity"
                if "RGB8" in features.PixelFormat.symbolics:
                    features.PixelFormat.value = "RGB8"
            features.Scan3dOutputMode.value = "ProjectedC"
            comp_names = enabled_components(features)

            data_stream_reset(ia)
            ia.start()

            idx = next_index(args.out)
            n_done = 0
            while args.interactive or n_done < args.count:
                if args.interactive:
                    if input(f"[{idx:06d}] Enter = capture, q = quit: ").strip().lower() == "q":
                        break

                features.TriggerSoftware.execute()
                with ia.fetch(timeout=args.timeout) as buffer:
                    parts = dict(zip(comp_names, buffer.payload.components))
                    tex, rng = parts["Intensity"], parts["Range"]
                    rgb = texture_to_rgb(tex)
                    depth_mm = rng.data.copy().astype(np.float32).reshape(rng.height, rng.width)

                if rgb.shape[:2] != depth_mm.shape:
                    logger.warning(f"texture {rgb.shape[:2]} and depth {depth_mm.shape} sizes differ")

                pcl = calculate_point_cloud_from_projc(depth_mm.reshape(-1), coord_map)
                valid = np.isfinite(pcl).all(axis=1) & (pcl[:, 2] > 0)
                rgb_flat = rgb.reshape(-1, 3)
                if len(rgb_flat) != len(pcl):
                    rgb_flat = np.full((len(pcl), 3), 255, np.uint8)

                if n_done == 0:
                    cam = {**intr, "width": int(depth_mm.shape[1]), "height": int(depth_mm.shape[0]),
                           "depth_scale": DEPTH_SCALE, "camera_space": args.camera_space,
                           "undistort_map": "undistort_map.npy"}
                    cam_path = args.out / "camera.json"
                    if cam_path.exists():
                        old = json.load(open(cam_path))
                        if (old["width"], old["height"]) != (cam["width"], cam["height"]) or \
                                not np.allclose([old[k] for k in ("fx", "fy", "cx", "cy")],
                                                [cam[k] for k in ("fx", "fy", "cx", "cy")], atol=0.5):
                            logger.warning(f"camera.json changed ({old['width']}x{old['height']} -> "
                                           f"{cam['width']}x{cam['height']}); older frames no longer match it")
                    cam_path.write_text(json.dumps(cam, indent=1))
                    # coord_map rows = ((a - ppu) / f, (b - ppv) / (f * aspect), 1): (a, b) = pinhole pixel
                    ab = np.stack([coord_map[:, 0] * intr["fx"] + intr["cx"], coord_map[:, 1] * intr["fy"] + intr["cy"]],
                                  axis=-1).reshape(*depth_mm.shape, 2)
                    np.save(args.out / "undistort_map.npy", ab.astype(np.float32))

                name = f"{idx:06d}"
                Image.fromarray(rgb).save(args.out / "rgb" / f"{name}.png")
                Image.fromarray(depth_to_png(depth_mm)).save(args.out / "depth" / f"{name}.png")
                write_ply(args.out / "pointcloud" / f"{name}.ply", pcl[valid], rgb_flat[valid])
                logger.info(f"Saved {name}: {rgb.shape[1]}x{rgb.shape[0]}, "
                            f"{valid.sum()} valid points ({valid.mean():.1%})")
                idx += 1
                n_done += 1

            ia.stop()
    logger.info(f"Captured {n_done} frame(s) -> {args.out}")
    return 0


def main():
    args = parse_args()
    if args.to_bop:
        kept = export_bop(args.out, args.to_bop, args.models, undistorted=not args.keep_distortion)
        print(f"{len(kept)} frame(s) -> {args.to_bop}/train_real/000000")
        return 0
    return capture(args)


if __name__ == "__main__":
    sys.exit(main())
