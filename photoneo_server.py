#!/usr/bin/env python3
"""Photoneo capture server: keeps one connection to the scanner open and captures on request.

data_conversion/camera_convert_data.py connects for every call (~4.5 s of a ~6 s capture); here the connection is made once, so a
capture costs only the shot. Commands on stdin, one JSON object per line; every reply is one stdout line "@@ <json>":

    {"cmd": "3d"}    3D scan (colour texture + depth) and its point cloud, then a 2D reference image
                     -> rgb/<idx>.png, depth/<idx>.png, pointcloud/<idx>.ply, gray2d/<idx>.npy
    {"cmd": "2d"}    2D image only (CameraOnlyMode: primary camera, no projector pattern, same pixels as the scan)
                     -> gray2d/<idx>_<n>.npy
    {"cmd": "quit"}
Replies carry the written file names and the time (s) of every step. Files use the camera_convert_data.py layout
(camera.json, undistort_map.npy in --out). gray2d = raw intensity of the primary camera (uint16 .npy, fast to write).

Run with the harvesters venv (the notebook starts it):
    ~/Projects/photoneo-python-examples/GigE-V/harvesters/.venv/bin/python photoneo_server.py --out runs/realtime/capture
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data_conversion.camera_convert_data import (DEFAULT_EXAMPLES_DIR, DEPTH_SCALE, depth_to_png, next_index, texture_to_rgb,
                                 write_ply)

T = time.perf_counter


def reply(**kw):
    print("@@ " + json.dumps(kw), flush=True)


class Scanner:
    def __init__(self, args):
        sys.path.insert(0, str(args.examples_dir))
        os.environ.setdefault("XDG_SESSION_TYPE", "x11")
        from harvesters.core import Harvester
        from photoneo_genicam.default_gentl_producer import producer_path
        from photoneo_genicam.features import enable_software_trigger
        from photoneo_genicam.pointcloud import calculate_point_cloud_from_projc, pre_fetch_coordinate_maps
        from photoneo_genicam.user_set import load_default_user_set
        from photoneo_genicam.utils import data_stream_reset
        from photoneo_genicam.components import enable_components, enabled_components
        self._pcl, self._reset = calculate_point_cloud_from_projc, data_stream_reset
        self._enable, self._enabled = enable_components, enabled_components
        self.out, self.timeout = args.out, args.timeout
        for sub in ("rgb", "depth", "pointcloud", "gray2d"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

        t0 = T()
        self.h = Harvester()
        self.h.add_file(str(producer_path), check_existence=True, check_validity=True)
        self.h.update()
        if not self.h.device_info_list:
            raise RuntimeError("no GigE Vision device found")
        serial = args.serial or self.h.device_info_list[0].serial_number
        self.ia = self.h.create({"serial_number": serial})
        self.f = f = self.ia.remote_device.node_map
        t1 = T()
        load_default_user_set(f)
        enable_software_trigger(f)
        res = f"Resolution_{args.resolution}"
        if res in f.Resolution.symbolics:
            f.Resolution.value = res
        f.CameraSpace.value = "PrimaryCamera"
        self.coord_map = pre_fetch_coordinate_maps(self.ia)
        fl = f.Scan3dFocalLength.value
        self.intr = {"cx": f.Scan3dPrincipalPointU.value, "cy": f.Scan3dPrincipalPointV.value,
                     "fx": fl, "fy": fl * f.Scan3dAspectRatio.value}
        self.model, self.serial = f.DeviceModelName.value, serial
        self.mode, self.running, self.names = None, False, []
        self.last3d, self.n2d = None, 0
        self.ready_timing = {"connect": t1 - t0, "setup": T() - t1}

    def set_mode(self, mode):
        """'3d' = colour texture + depth scan, '2d' = CameraOnlyMode intensity. Returns the switch time (s)."""
        if mode == self.mode:
            return 0.0
        t0, f = T(), self.f
        if self.running:
            self.ia.stop()
            self.running = False
        f.CameraOnlyMode.value = mode == "2d"
        self._enable(f, ["Intensity", "Range"] if mode == "3d" else ["Intensity"])
        if mode == "3d":
            f.TextureSource.value = "Color"
        f.ComponentSelector.value = "Intensity"
        pf = "RGB8" if mode == "3d" else "Mono16"
        if pf in f.PixelFormat.symbolics:
            f.PixelFormat.value = pf
        f.Scan3dOutputMode.value = "ProjectedC"
        self.names = self._enabled(f)
        self._reset(self.ia)
        self.ia.start()
        self.running, self.mode = True, mode
        return T() - t0

    def shot(self):
        self.f.TriggerSoftware.execute()
        with self.ia.fetch(timeout=self.timeout) as buf:
            return {n: (c.data.copy(), c.width, c.height, c.data_format) for n, c in zip(self.names, buf.payload.components)}

    def gray(self, comp):
        data, w, h, fmt = comp
        if fmt.startswith("Mono") and data.dtype != np.uint8:
            return data.reshape(h, w).astype(np.uint16)
        return np.asarray(Image.fromarray(texture_to_rgb(type("C", (), {"data": data, "width": w, "height": h,
                                                                      "data_format": fmt}))).convert("L"))

    def write_camera_json(self, shape):
        cam = {**self.intr, "width": int(shape[1]), "height": int(shape[0]), "depth_scale": DEPTH_SCALE,
               "camera_space": "PrimaryCamera", "undistort_map": "undistort_map.npy"}
        (self.out / "camera.json").write_text(json.dumps(cam, indent=1))
        cm = self.coord_map
        ab = np.stack([cm[:, 0] * self.intr["fx"] + self.intr["cx"], cm[:, 1] * self.intr["fy"] + self.intr["cy"]],
                      axis=-1).reshape(*shape, 2)
        np.save(self.out / "undistort_map.npy", ab.astype(np.float32))

    def capture_2d(self, name):
        tm = {"mode_2d": self.set_mode("2d")}
        t0 = T()
        comp = self.shot()["Intensity"]
        tm["shot_2d"] = T() - t0
        t0 = T()
        np.save(self.out / "gray2d" / name, self.gray(comp))
        tm["save_2d"] = T() - t0
        return tm

    def capture_3d(self):
        tm = {"mode_3d": self.set_mode("3d")}
        t0 = T()
        parts = self.shot()
        tm["scan_3d"] = T() - t0
        (tex, w, h, fmt), rng = parts["Intensity"], parts["Range"]
        rgb = texture_to_rgb(type("C", (), {"data": tex, "width": w, "height": h, "data_format": fmt}))
        depth = rng[0].astype(np.float32).reshape(rng[2], rng[1])
        t0 = T()
        pcl = self._pcl(depth.reshape(-1), self.coord_map)
        valid = np.isfinite(pcl).all(axis=1) & (pcl[:, 2] > 0)
        tm["pointcloud"] = T() - t0
        if self.last3d is None:
            self.write_camera_json(depth.shape)
        idx = next_index(self.out)
        name = f"{idx:06d}"
        t0 = T()
        Image.fromarray(rgb).save(self.out / "rgb" / f"{name}.png", compress_level=1)
        Image.fromarray(depth_to_png(depth)).save(self.out / "depth" / f"{name}.png", compress_level=1)
        tm["save_rgbd"] = T() - t0
        t0 = T()
        write_ply(self.out / "pointcloud" / f"{name}.ply", pcl[valid], rgb.reshape(-1, 3)[valid])
        tm["save_ply"] = T() - t0
        self.last3d, self.n2d = name, 0
        tm.update(self.capture_2d(f"{name}.npy"))           # 2D reference of this scan (same pixels)
        return {"name": f"{name}.png", "points": int(valid.sum()), "timing": tm}

    def close(self):
        try:
            if self.running:
                self.ia.stop()
            self.f.CameraOnlyMode.value = False
            self.ia.destroy()
        finally:
            self.h.reset()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("serial", nargs="?", help="device serial number (default: first device found)")
    p.add_argument("--out", type=Path, default=Path("runs/realtime/capture"))
    p.add_argument("--resolution", default="1236x1032")
    p.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    p.add_argument("--timeout", type=float, default=10.0, help="fetch timeout [s]")
    args = p.parse_args()
    try:
        cam = Scanner(args)
    except Exception as e:
        reply(ready=False, error=f"{type(e).__name__}: {e}")
        return 1
    reply(ready=True, model=cam.model, serial=cam.serial, timing=cam.ready_timing)
    # SIGTERM (binpick/binpick_real.py taking the scanner over) ends through the finally below: the device is released cleanly
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                cmd = json.loads(line).get("cmd")
                if cmd == "quit":
                    break
                if cmd == "3d":
                    reply(ok=True, **cam.capture_3d())
                elif cmd == "2d":
                    if cam.last3d is None:
                        raise RuntimeError("no 3D scan yet: the first command must be 3d")
                    cam.n2d += 1
                    name = f"{cam.last3d}_{cam.n2d:02d}.npy"
                    reply(ok=True, name=name, timing=cam.capture_2d(name))
                else:
                    raise ValueError(f"unknown command {cmd!r}")
            except Exception as e:
                reply(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
