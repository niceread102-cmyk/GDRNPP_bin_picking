"""Validation displays: 2D overlays (projected 3D boxes + axes) and 3D scenes (Open3D window or Plotly).

Detections are the per-frame dicts written by infer.py. Colours:
    green   confident and correct (ADD-S < ok_mm)           red     confident but WRONG (dangerous)
    orange  not confident (pose unverified)                 white   GT label where the pose is not ok (thin)
    magenta GT part (>= min visible px) with no detection   (without GT: green = confident, orange = not)
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"ok": (40, 220, 90), "wrong": (255, 40, 40), "unconf": (255, 165, 0), "conf": (40, 220, 90),
          "gt": (255, 255, 255), "missed": (255, 0, 255)}
AXIS_COLORS = [(255, 60, 60), (60, 255, 60), (70, 140, 255)]          # x, y, z
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
LEGEND = {"ok": "confident, correct", "wrong": "confident, WRONG", "unconf": "not confident",
          "conf": "confident", "missed": "GT missed by detector", "gt": "GT label (pose not ok)"}


def box_corners(model) -> np.ndarray:
    """8 corners (model frame, mm) of the CAD model's axis-aligned bounding box."""
    lo, size = model.xyz_min, model.xyz_size
    return np.array([[lo[0] + size[0] * i, lo[1] + size[1] * j, lo[2] + size[2] * k]
                     for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)[[0, 2, 4, 6, 1, 3, 5, 7]]


def status(det: dict, ok_mm: float = 1.0) -> str:
    if det.get("add_s_refined") is None:                 # no GT match (or real data)
        return "conf" if det.get("confident") else "unconf"
    if det.get("confident"):
        return "ok" if det["add_s_refined"] < ok_mm else "wrong"
    return "unconf"


def _project(K, pts):
    p = pts @ K.T
    return p[:, :2] / p[:, 2:3]


def gt_overlay_sets(objs: dict, dets: list[dict], min_visib_px: int, ok_mm: float = 1.0):
    """GT poses to draw: the label of each detection that is NOT confident-and-correct (thin white, shows
    where the pose should be) and visible GT parts with no detection at all (magenta)."""
    matched = {d["gt_inst_id"]: status(d, ok_mm) for d in dets if d.get("gt_inst_id") is not None}
    out = []
    for k, iid in enumerate(objs["inst_id"]):
        s = matched.get(int(iid))
        if s is not None:
            if s != "ok":
                out.append(("gt", objs["R"][k], objs["t"][k]))
        elif objs["visib_px"][k] >= min_visib_px:
            out.append(("missed", objs["R"][k], objs["t"][k]))
    return out


def select(dets: list[dict], gt, show: str, ok_mm: float = 1.0):
    """show: 'all' | 'confident' | 'problems' (confident-but-wrong, not confident, missed GT)."""
    if show == "all":
        return dets, gt
    if show == "confident":
        return [d if d.get("confident") else None for d in dets], [g for g in gt or [] if g[0] != "missed"]
    return [None if status(d, ok_mm) in ("ok", "conf") else d for d in dets], gt


def render_2d(rgb: np.ndarray, K: np.ndarray, dets: list[dict], model, gt: list | None = None,
              axes: str = "confident", labels: str = "wrong", ok_mm: float = 1.0, title: str = "") -> Image.Image:
    """Project each detection's 3D box (+ xyz axes) onto the image.

    axes: 'confident' | 'all' | 'none';  labels: 'wrong' | 'all' | 'none' (label = index and ADD-S error).
    gt: list of (kind, R, t) from gt_overlay_sets, or None.
    """
    img = Image.fromarray(rgb).convert("RGB")
    img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.35)       # darken so overlays pop
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    corners = box_corners(model)
    axis_len = 0.45 * model.diameter

    def box(R, t, color, width):
        uv = _project(K, corners @ np.asarray(R).T + np.asarray(t))
        for a, b in EDGES:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=width)
        return uv

    for kind, R, t in gt or []:
        box(R, t, COLORS[kind], 1 if kind == "gt" else 2)

    order = {"unconf": 0, "conf": 1, "ok": 1, "wrong": 2}                     # wrong ones on top
    counts = {}
    for i, det in sorted(((i, d) for i, d in enumerate(dets) if d is not None),
                         key=lambda x: order[status(x[1], ok_mm)]):
        s = status(det, ok_mm)
        counts[s] = counts.get(s, 0) + 1
        R, t = np.asarray(det["R"]), np.asarray(det["t"])
        uv = box(R, t, COLORS[s], 2)
        if axes == "all" or (axes == "confident" and det.get("confident")):
            o = _project(K, t[None])[0]
            for ax in range(3):
                e = _project(K, (t + R[:, ax] * axis_len)[None])[0]
                d.line([tuple(o), tuple(e)], fill=AXIS_COLORS[ax], width=2)
                v = (e - o) / max(np.linalg.norm(e - o), 1e-6)
                n = np.array([-v[1], v[0]])
                d.polygon([tuple(e), tuple(e - 7 * v + 4 * n), tuple(e - 7 * v - 4 * n)], fill=AXIS_COLORS[ax])
        if labels == "all" or (labels == "wrong" and s == "wrong"):
            err = det.get("add_s_refined")
            txt = f"#{i}" + (f" {err:.1f}mm" if err is not None else "")
            x, y = uv.min(0)
            d.text((x, y - 15), txt, fill=COLORS[s], font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    if gt:
        counts["missed"] = sum(1 for g in gt if g[0] == "missed")
        counts["gt"] = sum(1 for g in gt if g[0] == "gt")
    lines = [title] if title else []
    lines += [f"{counts[k]:4d}  {LEGEND[k]}" for k in ("ok", "conf", "wrong", "unconf", "missed", "gt") if k in counts]
    lines.append("axes: x red, y green, z blue")
    h = 18 * len(lines) + 10
    panel = Image.new("RGBA", (260, h), (0, 0, 0, 190))
    img.paste(panel, (8, 8), panel)
    y = 13
    for ln in lines:
        key = next((k for k, v in LEGEND.items() if ln.endswith(v)), None)
        d.text((16, y), ln, fill=COLORS.get(key, (230, 230, 230)), font=font)
        y += 18
    return img


# ------------------------------------------------------------------ 3D
def _box_lines(R, t, corners):
    c = corners @ np.asarray(R).T + np.asarray(t)
    return c, EDGES


def cloud_from_frame(rgb, xyz, valid, max_points: int = 60000, seed: int = 0):
    idx = np.flatnonzero(valid.reshape(-1))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    return xyz.reshape(-1, 3)[idx], rgb.reshape(-1, 3)[idx]


def plotly_scene(rgb, xyz, valid, dets, model, gt=None, ok_mm: float = 1.0, axes: str = "confident",
                 max_points: int = 60000, title: str = ""):
    """Interactive 3D figure (camera frame, mm): coloured point cloud, detection boxes/axes, GT boxes."""
    import plotly.graph_objects as go
    pts, col = cloud_from_frame(rgb, xyz, valid, max_points)
    fig = go.Figure(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name="point cloud",
                                 marker=dict(size=1.2, color=[f"rgb({r},{g},{b})" for r, g, b in col]),
                                 hoverinfo="skip"))
    corners = box_corners(model)
    axis_len = 0.45 * model.diameter

    def add_boxes(items, name, color, width):
        xs, ys, zs, hover = [], [], [], []
        for R, t, txt in items:
            c, edges = _box_lines(R, t, corners)
            for a, b in edges:
                xs += [c[a, 0], c[b, 0], None]
                ys += [c[a, 1], c[b, 1], None]
                zs += [c[a, 2], c[b, 2], None]
                hover += [txt, txt, None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, text=hover, hoverinfo="text",
                                       line=dict(color=f"rgb{color}", width=width)))

    groups = {}
    for i, det in enumerate(dets):
        if det is None:
            continue
        s = status(det, ok_mm)
        err = det.get("add_s_refined")
        txt = (f"#{i} {LEGEND[s]}<br>ICP fitness {det['icp_fitness']:.2f}, rmse {det['icp_rmse']:.2f} mm"
               + (f"<br>ADD-S {err:.2f} mm" if err is not None else ""))
        groups.setdefault(s, []).append((det["R"], det["t"], txt))
    for kind, R, t in gt or []:
        groups.setdefault(kind, []).append((R, t, LEGEND[kind]))
    for s in ("gt", "missed", "unconf", "conf", "ok", "wrong"):
        if s in groups:
            add_boxes(groups[s], f"{LEGEND[s]} ({len(groups[s])})", COLORS[s], 2 if s == "gt" else 4)

    for ax, name in enumerate("xyz"):
        xs, ys, zs = [], [], []
        for det in dets:
            if det is not None and (axes == "all" or (axes == "confident" and det.get("confident"))):
                R, t = np.asarray(det["R"]), np.asarray(det["t"])
                e = t + R[:, ax] * axis_len
                xs += [t[0], e[0], None]
                ys += [t[1], e[1], None]
                zs += [t[2], e[2], None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"{name}-axis", hoverinfo="skip",
                                       line=dict(color=f"rgb{AXIS_COLORS[ax]}", width=5)))
    # camera frame: x right, y down, z forward -> look along +z from the sensor
    fig.update_layout(title=title, height=750, margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"),
                      scene=dict(aspectmode="data", xaxis_title="x mm", yaxis_title="y mm", zaxis_title="z mm",
                                 camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.3, y=-0.6, z=-1.6))))
    return fig


def open3d_scene(rgb, xyz, valid, dets, model, gt=None, ok_mm: float = 1.0, axes: str = "confident",
                 screenshot: str | None = None, size=(1400, 1000)):
    """Interactive Open3D window (or an off-screen screenshot if `screenshot` is a path)."""
    import open3d as o3d
    pts, col = cloud_from_frame(rgb, xyz, valid, max_points=400000)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)
    geoms = [pcd]
    corners = box_corners(model)

    def lineset(R, t, color):
        c, edges = _box_lines(R, t, corners)
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(c), o3d.utility.Vector2iVector(np.array(edges)))
        ls.paint_uniform_color(np.array(color) / 255.0)
        return ls

    for kind, R, t in gt or []:
        geoms.append(lineset(R, t, COLORS[kind]))
    for det in (d for d in dets if d is not None):
        s = status(det, ok_mm)
        geoms.append(lineset(det["R"], det["t"], COLORS[s]))
        if axes == "all" or (axes == "confident" and det.get("confident")):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = np.asarray(det["R"]), np.asarray(det["t"])
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.45 * model.diameter).transform(T))

    vis = o3d.visualization.Visualizer()
    vis.create_window("bin pose validation", width=size[0], height=size[1], visible=screenshot is None)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.1])
    opt.point_size = 2.0
    vc = vis.get_view_control()
    vc.set_lookat(np.median(pts, axis=0))
    vc.set_front([0.25, 0.45, -1.0])      # from the sensor side, tilted so height is visible
    vc.set_up([0, -1, 0])
    vc.set_zoom(0.38)
    if screenshot:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(screenshot), do_render=True)
        vis.destroy_window()
    else:
        vis.run()
        vis.destroy_window()
