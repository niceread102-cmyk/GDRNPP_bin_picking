#!/usr/bin/env python3
"""The last two cells of notebooks/results_6dpose_real_optimize.ipynb as a script, with a window.

1. run_cycle: Photoneo 3D scan -> side plate 6D pose -> up to 15 picks (robot base TCP targets; --per-scan)
2. auto_pick_2d: the picks one at a time through ee_6d.py -> /target_pose -> sequence_real_action, a 2D image after
   each pick corrects / drops the rest, a new 3D scan when none is left, done when a new scan finds nothing graspable

The window shows the 3D scan with every detected part (yellow number = pick order, colour = why a part is not
picked), the latest 2D image (green = picks left with their order, orange = the part being picked) and the pick
table with the state of every pick. Pipeline, checks and robot hand-off are binpick_real.py's (the notebook's code
plus home before every capture, IK check before sending, gripper + wrist camera collision check, pause = home now).

Start terminal 1 (MoveIt + robot) and terminal 2 (sequence_real_action) first: lite6 real control code.txt.

    python binpick/auto_pick.py                        # camera + robot, until a scan finds nothing graspable
    python binpick/auto_pick.py 5                      # pick 5 parts, then home and done
    python binpick/auto_pick.py 5 --per-scan 3         # ... choosing at most 3 picks per 3D scan (default 15, notebook 10)
    python binpick/auto_pick.py --no-robot             # camera only, nothing is sent to the robot
    python binpick/auto_pick.py --keep-going           # do not stop when a scan finds nothing: scan again every --idle s
    python binpick/auto_pick.py --wall-margin 0.05     # parts at least 5 cm inside the bin wall (default 8 cm, 0 = off)
    python binpick/auto_pick.py --notebook-rules       # the notebook's 10 cm bin-wall margin instead of the gripper shape
Keys in the window: p = pause (robot home now), r = resume, c = new 3D scan after this pick, s = stop.
"""
import argparse
import queue as queue_mod
import signal
import threading
import traceback
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import binpick_real as bp

# colour of a confident part that is not picked, by the first failed check (why_not)
WHY_COLOUR = {"": (60, 220, 90), "gripper hits bin wall": (0, 200, 255), "gripper hits parts": (230, 60, 230),
              "out of reach": (150, 150, 150), "near bin wall": (0, 200, 255)}
OTHER_COLOUR = (255, 120, 40)                                        # covered, tilted, hole / ring blocked
PICK_STATE = {"done": "완료", "failed": "실패", "not started": "시작 안 됨", "aborted": "중단 (재시도)",
              "busy": "로봇이 멈추지 않음"}


def scan_image(res):
    """3D scan: the selected parts' boxes and pick order, and a circle on every other confident part coloured by
    the reason it is not picked."""
    f, dets, picks, full = res["frame"], res["dets"], res["picks"], res.get("full")
    rgb = f["rgb"].copy()
    if f.get("wall"):                                                # detected blue bin in cyan
        rgb[f["wall"]["mask"]] = (0.5 * rgb[f["wall"]["mask"]] + (0, 127, 127)).astype(np.uint8)
    sel = set(picks["det"].astype(int)) if not picks.empty else set()
    img = bp.render_2d(rgb, f["K"], [d if n in sel else None for n, d in enumerate(dets)], bp.model, None,
                       labels="none", title="")
    dr = ImageDraw.Draw(img)
    K = np.asarray(f["K"], float)
    uv = lambda d: (lambda c: (c[0] / c[2], c[1] / c[2]))(K @ np.asarray(d["t"], float))
    for n, d in enumerate(dets):
        if n in sel:
            continue
        u, v = uv(d)
        if full is None or not bool(full.loc[n, "confident"]):
            dr.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(110, 110, 110))
            continue
        colour = WHY_COLOUR.get(bp.why_not(full.loc[n], detail=False), OTHER_COLOUR)
        dr.ellipse((u - 16, v - 16, u + 16, v + 16), outline=colour, width=3)
    font = ImageFont.load_default(size=30)
    for k, r in picks.iterrows():
        u, v = uv(dets[int(r["det"])])
        dr.ellipse((u - 7, v - 7, u + 7, v + 7), fill=(255, 230, 0))
        dr.text((u + 9, v - 34), str(k), fill=(255, 230, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    small = ImageFont.load_default(size=18)
    legend = [("pick order", (255, 230, 0)), ("graspable", WHY_COLOUR[""]),
              (f"near bin wall (< {bp.BIN_WALL_MARGIN * 100:.0f} cm) / gripper hits it", (0, 200, 255)),
              ("gripper hits parts", (230, 60, 230)), ("out of reach", (150, 150, 150)),
              ("covered / tilted / hole blocked", OTHER_COLOUR)]
    dr.rectangle((8, 8, 400, 16 + 24 * len(legend)), fill=(0, 0, 0))
    for i, (text, colour) in enumerate(legend):
        dr.ellipse((16, 16 + 24 * i, 30, 30 + 24 * i), outline=colour, width=3)
        dr.text((40, 13 + 24 * i), text, fill=colour, font=small)
    return img


def run(args):
    import tkinter as tk
    from tkinter import font as tkfont, messagebox, ttk
    from PIL import ImageTk

    BG, PANEL, TEXT, MUTED = "#0e1217", "#151b23", "#e6edf3", "#8b98a5"
    root = tk.Tk()
    root.title("auto_pick · GDRNPP bin picking")
    root.configure(bg=BG)
    root.geometry("1700x1000")
    fams = set(tkfont.families())
    FONT = next((f for f in ("NanumBarunGothic", "NanumGothic") if f in fams), "TkDefaultFont")
    MONO = next((f for f in ("NanumGothicCoding", "DejaVu Sans Mono") if f in fams), "TkFixedFont")
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=24, font=(FONT, 10))
    st.configure("Treeview.Heading", background="#1c2530", foreground=MUTED, relief="flat", font=(FONT, 10, "bold"))
    st.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    status = tk.Label(root, text="모델 로딩 + 카메라 연결 중", bg=BG, fg=TEXT, font=(FONT, 15, "bold"), anchor="w")
    status.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 0))
    stats = tk.Label(root, text="", bg=BG, fg=MUTED, font=(FONT, 11), anchor="w")
    stats.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14)
    panes = {}
    for col, key in enumerate(("3d", "2d")):
        cap = tk.Label(root, text="3D 촬영: 찾은 부품과 집는 순서" if key == "3d" else "2D 영상: 남은 pick 순서 / 집는 중",
                       bg=BG, fg=TEXT, font=(FONT, 12, "bold"), anchor="w")
        cap.grid(row=2, column=col, sticky="ew", padx=14, pady=(8, 2))
        view = tk.Label(root, bg="#0a0d11")
        view.grid(row=3, column=col, sticky="nsew", padx=(14 if col == 0 else 4, 4 if col == 0 else 14))
        panes[key] = {"cap": cap, "view": view, "img": None, "tk": None}
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(3, weight=3)
    root.rowconfigure(4, weight=2)

    cols = (("n", "순서", 50), ("pick", "pick", 70), ("state", "상태", 130), ("x", "tcp x", 80), ("y", "tcp y", 80),
            ("z", "tcp z", 80), ("tilt", "기울기°", 70), ("yaw", "툴 회전°", 75), ("note", "비고", 330))
    tree = ttk.Treeview(root, columns=[c[0] for c in cols], show="headings")
    for key, name, w in cols:
        tree.heading(key, text=name)
        tree.column(key, width=w, anchor="w" if key == "note" else "center", stretch=key == "note")
    for tag, colour in (("done", "#2ea86f"), ("picking", "#ff9a3c"), ("failed", "#e5534b"), ("dropped", "#6b7682")):
        tree.tag_configure(tag, foreground=colour)
    tree.grid(row=4, column=0, sticky="nsew", padx=(14, 4), pady=8)
    logbox = tk.Text(root, bg=PANEL, fg="#c9d1d9", font=(MONO, 9), relief="flat", wrap="none", state="disabled",
                     highlightthickness=0)
    logbox.grid(row=4, column=1, sticky="nsew", padx=(4, 14), pady=8)
    bar = tk.Frame(root, bg=BG)
    bar.grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))

    ui = {"loop": None, "closing": False, "picks": {}}

    def show(key, img=None):
        p = panes[key]
        if img is not None:
            p["img"] = img
        if p["img"] is None:
            return
        w, h = max(p["view"].winfo_width() - 4, 50), max(p["view"].winfo_height() - 4, 50)
        s = min(w / p["img"].width, h / p["img"].height)
        p["tk"] = ImageTk.PhotoImage(p["img"].resize((max(int(p["img"].width * s), 1), max(int(p["img"].height * s), 1)),
                                                     Image.BILINEAR))
        p["view"].configure(image=p["tk"])

    def add_log(msg):
        logbox.configure(state="normal")
        logbox.insert("end", msg + "\n")
        if int(logbox.index("end-1c").split(".")[0]) > 2000:
            logbox.delete("1.0", "300.0")
        logbox.see("end")
        logbox.configure(state="disabled")

    def set_pick(name, **kw):
        if not tree.exists(name):
            return
        vals = dict(zip([c[0] for c in cols], tree.item(name, "values"))) | kw
        tag = {"완료": "done", "집는 중": "picking", "실패": "failed", "시작 안 됨": "failed", "빠짐": "dropped",
               "제외": "dropped"}.get(vals["state"], "")
        tree.item(name, values=[vals[c[0]] for c in cols], tags=(tag,))

    def on_scan(scan, res):
        show("3d", scan_image(res))
        s = res["summary"]
        panes["3d"]["cap"].configure(text=f"3D 촬영 {scan}: 마스크 {s['masks']} → 검증 {s['verified']} → "
                                          f"잡기 가능 {s['graspable']} → 선택 {s['selected']}  (pose {s['pose (s)']:.1f} s)")
        tree.delete(*tree.get_children())
        for k, r in res["picks"].iterrows():
            ok = r["robot safety"] == "ok"
            tree.insert("", "end", iid=f"{scan}-{k}", values=(
                k, f"{scan}-{k}", "대기" if ok else "제외", f"{r['tcp x (m)']:.4f}", f"{r['tcp y (m)']:.4f}",
                f"{r['tcp z (m)']:.4f}", f"{r['tool tilt (deg)']:.1f}", f"{r['tool yaw (deg)']:+.0f}",
                "" if ok else f"robot safety: {r['robot safety']}"))
            set_pick(f"{scan}-{k}")

    def on_queue(rows):
        names = {row[1]: row for row in rows}
        for name in tree.get_children():
            vals = dict(zip([c[0] for c in cols], tree.item(name, "values")))
            if name in names:
                n, _, x, y, z, tilt, match, shift = names[name]
                set_pick(name, n=n, x=x, y=y, z=z, tilt=tilt,
                         state="대기" if vals["state"] in ("대기", "중단 (재시도)") else vals["state"],
                         note=f"2D 일치 {match}, 보정 {shift} mm" if match != "-" else vals["note"])
            elif vals["state"] == "대기":
                set_pick(name, n="-", state="빠짐")

    def poll():
        try:
            while True:
                kind, a = bp.UI.get_nowait()
                if kind == "log":
                    add_log(a[0])
                elif kind == "state":
                    status.configure(text=f"{bp_state_names.get(a[0], a[0])}   {a[1]}")
                elif kind == "scan_result":
                    on_scan(*a)
                elif kind == "image" and not a[1].startswith("3D"):
                    show("2d", a[0])
                    panes["2d"]["cap"].configure(text=a[1])
                elif kind == "queue":
                    on_queue(a[0])
                elif kind == "dropped":
                    for name, why in a[0]:
                        set_pick(name, n="-", state="빠짐", note=why)
                elif kind == "pick_start":
                    set_pick(a[0], state="집는 중", n="▶")
                elif kind == "pick_result":
                    set_pick(a[0], state=PICK_STATE.get(a[1], a[1]), note=f"{a[2]:.0f} s",
                             n="✓" if a[1] == "done" else "✗")
                elif kind == "stats":
                    s = a[0]
                    stats.configure(text=f"집은 부품 {s['done']}" + (f" / {args.picks}" if args.picks else "")
                                         + f"  ·  한 번 촬영에 최대 {args.per_scan}개  ·  실패 {s['failed']}  ·  IK로 제외 {s['unreachable']}  ·  "
                                         f"3D 촬영 {s['scans']}" + (f"  ·  평균 사이클 {s['cycle']:.0f} s" if s["cycle"] else ""))
                elif kind == "ready":
                    start()
                elif kind == "loop_end":
                    if ui["closing"]:
                        bp.CAM.close()
                        root.destroy()
                        return
                    status.configure(text=status.cget("text") + "   (끝: 창을 닫으세요)")
        except queue_mod.Empty:
            pass
        root.after(100, poll)

    bp_state_names = {"init": "준비 중", "connect": "로봇 연결", "home": "home 이동", "scan": "3D 촬영", "ik": "IK 확인",
                      "pick": "집는 중", "check": "2D 확인", "idle": "부품 없음", "paused": "일시정지", "stopped": "정지",
                      "error": "오류"}

    def running():
        return ui["loop"] is not None and ui["loop"].is_alive()

    def start():
        ui["loop"] = bp.PickLoop(types.SimpleNamespace(
            no_robot=args.no_robot, dry_seconds=args.dry_seconds, no_save=args.no_save, max_picks=args.picks,
            idle=args.idle, stop_when_empty=not args.keep_going))
        ui["loop"].start()

    def pause(_=None):
        if running():
            ui["loop"].pause_ev.set()
            if ui["loop"].robot is not None:
                ui["loop"].robot.home_now()
            status.configure(text="일시정지: home으로 이동 (잡은 부품이 있으면 놓고 감)")

    def resume(_=None):
        if running():
            ui["loop"].pause_ev.clear()

    def rescan(_=None):
        # home at once (a running approach is aborted, a grasped part is placed first), then a new 3D scan and picking
        if running():
            ui["loop"].rescan_ev.set()
            ui["loop"].pause_ev.clear()
            if ui["loop"].robot is not None:
                ui["loop"].robot.home_now()
            status.configure(text="다시 촬영: home으로 이동 후 3D 촬영부터 다시 (잡은 부품이 있으면 놓고 감)")

    def stop(_=None):
        if running():
            ui["loop"].stop_ev.set()
            ui["loop"].pause_ev.clear()
            status.configure(text="이번 사이클이 끝나면 home으로 가서 정지")

    for text, colour, cmd in (("❚❚ 일시정지 (p)", "#d99a1e", pause), ("▶ 재개 (r)", "#2ea86f", resume),
                              ("↻ 다시 촬영 (c)", "#3b82f6", rescan), ("■ 정지 (s)", "#d9534f", stop)):
        tk.Button(bar, text=text, command=cmd, bg=colour, fg="white", activebackground=colour, relief="flat",
                  font=(FONT, 12, "bold"), padx=16, pady=6).pack(side="left", padx=(0, 8))
    tk.Label(bar, text="로봇을 바로 멈추려면 terminal 2에서 Ctrl+C", bg=BG, fg=MUTED, font=(FONT, 10)).pack(side="left", padx=8)
    for key, cmd in (("p", pause), ("r", resume), ("c", rescan), ("s", stop)):
        root.bind(key, cmd)

    def on_close():
        if running() and not ui["closing"]:
            if not messagebox.askokcancel("종료", "지금 사이클이 끝나면 로봇을 home으로 보낸 뒤 닫습니다. 종료할까요?"):
                return
            ui["closing"] = True
            stop()
            return
        if ui["loop"] is not None and ui["loop"].robot is not None:
            ui["loop"].robot.close()
        bp.CAM.close()
        root.destroy()

    for key in panes:
        panes[key]["view"].bind("<Configure>", lambda e, k=key: show(k))
    root.protocol("WM_DELETE_WINDOW", on_close)
    signal.signal(signal.SIGINT, lambda *_: root.after(0, on_close))

    def init():
        try:
            bp.load_models()
            if bp.OFFLINE is None:
                bp.CAM.start()
            bp.post("ready")
        except Exception as e:
            bp.log(traceback.format_exc())
            bp.post("state", "error", f"초기화 실패: {e}")

    threading.Thread(target=init, daemon=True).start()
    root.after(100, poll)
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("picks", nargs="?", type=int, help="parts to pick, then home and done (default: until a scan "
                                                      "finds nothing graspable)")
    p.add_argument("--per-scan", type=int, default=bp.TOP_K, help="picks chosen per 3D scan (default 15; the notebook's TOP_K was 10)")
    p.add_argument("--no-robot", action="store_true", help="do not send anything to the robot (dry run)")
    p.add_argument("--offline", nargs=2, metavar=("BOP_ROOT", "VIEW"), help="saved BOP frame instead of the camera")
    p.add_argument("--keep-going", action="store_true", help="when a scan finds nothing, scan again every --idle s")
    p.add_argument("--idle", type=float, default=10.0, help="s between 3D scans with --keep-going")
    p.add_argument("--dry-seconds", type=float, default=3.0, help="s per pick with --no-robot")
    p.add_argument("--no-save", action="store_true", help="do not write runs/realtime/<time>/ and the session csv")
    p.add_argument("--wall-margin", type=float, default=bp.BIN_WALL_MARGIN,
                   help="m: only parts at least this far (horizontally) inside the bin wall (default 0.08; the "
                        "notebook's rule was 0.10; 0 = off, the gripper-shape check alone)")
    p.add_argument("--max-overlap", type=float, default=bp.MAX_OVERLAP_SELECTED,
                   help="share of a pick's silhouette that may overlap an earlier pick of the same scan "
                        "(default 0.20; the notebook: 0.02)")
    p.add_argument("--notebook-rules", action="store_true",
                   help="the notebook's bin-wall rule (10 cm margin, tool turned 0 deg) instead of the gripper shape")
    args = p.parse_args()
    if (args.picks is not None and args.picks < 1) or args.per_scan < 1:
        p.error("picks and --per-scan must be at least 1")
    bp.TOP_K = args.per_scan
    bp.BIN_WALL_MARGIN = args.wall_margin
    bp.MAX_OVERLAP_SELECTED = args.max_overlap
    if args.offline:
        bp.OFFLINE = (args.offline[0], int(args.offline[1]))
    if args.notebook_rules:
        bp.BIN_WALL_MARGIN, bp.TOOL_HIT_MAX, bp.TOOL_YAWS = 0.10, 10 ** 9, (0,)
    bp.UI = queue_mod.Queue()
    run(args)


if __name__ == "__main__":
    main()
