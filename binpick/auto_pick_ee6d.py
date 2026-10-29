"""Pick the notebook's TCP targets one after another: ee_6d.py publisher -> /target_pose -> sequence_real_action.

sequence_real_action ignores a target while a cycle runs and reports nothing when it ends, so the end of a cycle is
read from /joint_states: after a target is sent the arm must start moving, and the cycle is done when it is back at
robot.poses.home and stands still. A target that never starts the arm (safety box, IK) or an arm that stops
elsewhere (planning failed mid-cycle) counts as failed and the next target is sent.

Runs with the ROS python (/usr/bin/python3, ROS + lite6_ws sourced); the notebook starts it:

    python3 binpick/auto_pick_ee6d.py runs/realtime/auto_pick_targets.json     # the targets of the file (at most 15)
    python3 binpick/auto_pick_ee6d.py --stdin                                   # one target per stdin line, see below
targets JSON: {"targets": [{"rank": 1, "xyz": [x, y, z], "quat_xyzw": [qx, qy, qz, qw]}, ...]} (robot base, m)
--stdin: each line is one target object (or "quit"); after each cycle one stdout line
"@@ {"rank": k, "status": "done" | "not started" | "failed" | "busy" | "aborted", "seconds": s}", so the caller can look at the
bin (2D camera) before it sends the next target. "@@ {"ready": true}" is printed once the robot topics are up.
--stdin line {"cmd": "home"}: arm to robot.poses.home through sequence_real_action's /go_home (it moves only when idle,
so its own startup / end-of-cycle home moves are never overlapped); nothing is sent when the arm is already home.
Reply "@@ {"cmd": "home", "status": "home" (already) | "done" | "not started" | "failed" | "busy" | "no subscriber",
"seconds": s}".
--stdin line {"cmd": "home_now"} (no reply): /go_home at once, also while a cycle is followed (pause): before the grasp
sequence_real_action stops the approach and goes home, and that target is reported "aborted"; after the grasp it
places the part first.
--stdin line {"cmd": "check", "targets": [...]}: the IK calls sequence_real_action makes for each target (pre-grasp
seeded with robot.poses.init, dive end seeded with the pre-grasp solution; same config, same /compute_ik), so an
unreachable target is skipped instead of failing on the robot. Reply "@@ {"cmd": "check", "results": [{"rank": k,
"ok": bool, "reason": str}, ...]}".
"""
import argparse
import importlib.util
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

from my_lite6_control.config import default_config_path, load_config, named_pose
from my_lite6_control.motion import build_ik_request, eef_target_from_tcp, ik_joint_targets, pre_grasp_pose

EE_6D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),   # project root (binpick/..)
                     "lite6_control/lite6_ws/src/my_lite6_control/my_lite6_control/ee_6d.py")


def load_ee6d(path):
    spec = importlib.util.spec_from_file_location("ee_6d", os.path.expanduser(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ArmWatch:
    """Latest arm joint values and the time of the last movement."""

    def __init__(self, node, names, topic, move_tol):
        self.names, self.move_tol = list(names), move_tol
        self.q = self.q_ref = None
        self.t_move = time.monotonic()
        node.create_subscription(JointState, topic, self._cb, 10)

    def _cb(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if not all(n in pos for n in self.names):
            return
        self.q = np.array([pos[n] for n in self.names])
        if self.q_ref is None or np.abs(self.q - self.q_ref).max() > self.move_tol:
            self.q_ref, self.t_move = self.q.copy(), time.monotonic()

    def still_for(self):
        return time.monotonic() - self.t_move


def spin_until(node, cond, timeout):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if cond():
            return True
    return False


def pick_one(node, arm, at_home, t, args, tag="", abort=None, at_goal=None):
    """Send one target and follow the cycle. Returns (status, seconds).
    abort: threading.Event set by "home_now" (sequence_real_action stops the approach and goes home); a cycle that
    then ends at home without passing the drop pose (at_goal) is "aborted", not "done"."""
    k = t["rank"]
    if abort is not None:
        abort.clear()
    # sequence_real_action is idle once the arm stood still for a while (it ignores targets while busy)
    if not spin_until(node, lambda: arm.still_for() > 3.0, args.cycle_timeout):
        print(f"#{k}: the arm does not come to rest", flush=True)
        return "busy", 0.0
    (x, y, z), (qx, qy, qz, qw) = map(float, t["xyz"]), map(float, t["quat_xyzw"])
    print(f"\n{tag}#{k}: ee_6d quat {x:.4f} {y:.4f} {z:.4f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}", flush=True)
    q0 = arm.q.copy()
    node.publish_target_quat(x, y, z, qx, qy, qz, qw)

    moved = lambda: np.abs(arm.q - q0).max() > 0.01
    if not spin_until(node, lambda: moved() or (abort is not None and abort.is_set()), args.start_timeout):
        print(f"#{k}: the arm did not move -> rejected (safety box / IK), see terminal 2", flush=True)
        return "not started", 0.0
    if not moved():                                                  # home_now before the arm left home
        spin_until(node, lambda: arm.still_for() > 3.0 and at_home(), args.cycle_timeout)
        print(f"#{k}: aborted before the arm moved", flush=True)
        return "aborted", 0.0
    print(f"#{k}: moving ...", flush=True)
    t0 = time.monotonic()
    done = lambda: at_home() and arm.still_for() > 1.0
    stalled = lambda: not at_home() and arm.still_for() > args.stall
    placed = [False]                                                 # the arm passed the drop pose

    def finished():
        placed[0] = placed[0] or (at_goal is not None and at_goal())
        return done() or stalled()

    spin_until(node, finished, args.cycle_timeout)
    dt = time.monotonic() - t0
    if done() and abort is not None and abort.is_set() and not placed[0]:
        print(f"#{k}: aborted (home requested before the grasp), back at home in {dt:.0f} s", flush=True)
        return "aborted", dt
    if done():
        print(f"#{k}: done in {dt:.0f} s (back at home)", flush=True)
        return "done", dt
    why = f"stopped away from home for {args.stall:.0f} s" if stalled() else f"timeout {args.cycle_timeout:.0f} s"
    print(f"#{k}: failed ({why}), see terminal 2", flush=True)
    return "failed", dt


def go_home(node, arm, at_home, home_pub, args):
    """Arm to home unless it is there. Returns (status, seconds)."""
    if arm.q is not None and at_home() and arm.still_for() > 1.0:
        return "home", 0.0
    # a cycle (or sequence_real_action's startup home move) may still run: wait until the arm rests
    if not spin_until(node, lambda: arm.still_for() > 3.0, args.cycle_timeout):
        print("home: the arm does not come to rest", flush=True)
        return "busy", 0.0
    if at_home():
        return "home", 0.0
    if home_pub.get_subscription_count() == 0:
        print(f"home: nobody subscribes {home_pub.topic_name}: restart terminal 2 (sequence_real_action) after "
              "colcon build of my_lite6_control", flush=True)
        return "no subscriber", 0.0
    print("\nhome: requesting robot.poses.home", flush=True)
    q0 = arm.q.copy()
    home_pub.publish(Empty())
    if not spin_until(node, lambda: np.abs(arm.q - q0).max() > 0.01, args.start_timeout):
        print("home: the arm did not move (sequence_real_action busy?), see terminal 2", flush=True)
        return "not started", 0.0
    t0 = time.monotonic()
    done = lambda: at_home() and arm.still_for() > 1.0
    stalled = lambda: not at_home() and arm.still_for() > args.stall
    spin_until(node, lambda: done() or stalled(), args.cycle_timeout)
    dt = time.monotonic() - t0
    if done():
        print(f"home: reached in {dt:.0f} s", flush=True)
        return "done", dt
    print("home: failed, see terminal 2", flush=True)
    return "failed", dt


def joint_state(cfg, q):
    return JointState(name=list(cfg.robot.joint_names), position=[float(v) for v in q])


def solve_ik(node, client, pose, seed, cfg):
    """sequence_real_action's _move_to IK for TCP pose. Returns (joint values, None) or (None, reason)."""
    fut = client.call_async(build_ik_request(eef_target_from_tcp(pose, cfg), seed, cfg))
    if not spin_until(node, fut.done, float(cfg.planning.ik.timeout_s) + 5.0):
        return None, "compute_ik did not answer"
    return ik_joint_targets(fut.result(), cfg)


def check_target(node, client, t, cfg):
    """Can sequence_real_action reach target t? Its cycle: init pose -> pre-grasp (IK seeded with init) -> dive along
    the tool z (IK seeded with the pre-grasp solution). Returns (ok, reason)."""
    grasp = Pose()
    grasp.position.x, grasp.position.y, grasp.position.z = map(float, t["xyz"])
    (grasp.orientation.x, grasp.orientation.y, grasp.orientation.z,
     grasp.orientation.w) = map(float, t["quat_xyzw"])
    pre = pre_grasp_pose(grasp, cfg.sequence.approach_dist_m)
    q_pre, err = solve_ik(node, client, pre, joint_state(cfg, named_pose(cfg, "init")), cfg)
    if err:
        return False, f"pre-grasp IK: {err}"
    _, err = solve_ik(node, client, pre_grasp_pose(pre, -cfg.sequence.descend_dist_m), joint_state(cfg, q_pre), cfg)
    if err:
        return False, f"dive IK: {err}"
    return True, ""


class StdinReader(threading.Thread):
    """stdin lines read in the background, so "home_now" works while a cycle is followed: it is handled at once
    (on_home_now), every other line is queued for stdin_targets."""

    def __init__(self, on_home_now):
        super().__init__(daemon=True)
        self.on_home_now = on_home_now
        self.lines = queue.Queue()

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "quit":
                break
            msg = json.loads(line)
            if msg.get("cmd") == "home_now":
                self.on_home_now()
            else:
                self.lines.put(msg)
        self.lines.put(None)


def stdin_targets(node, reader):
    """Target objects from the stdin reader, one per line, while the node keeps spinning (joint states stay fresh)."""
    while True:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            msg = reader.lines.get_nowait()
        except queue.Empty:
            continue
        if msg is None:
            return
        yield msg


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("targets", nargs="?", help="targets JSON (robot base, m, quaternion xyzw)")
    p.add_argument("--stdin", action="store_true", help="read one target per line from stdin, report each cycle")
    p.add_argument("--ee6d", default=EE_6D, help="ee_6d.py whose TargetPublisher sends /target_pose")
    p.add_argument("--start-timeout", type=float, default=20.0, help="s for the arm to start moving after a target")
    p.add_argument("--stall", type=float, default=20.0, help="s standing still away from home = cycle failed")
    p.add_argument("--cycle-timeout", type=float, default=240.0, help="s for one whole cycle")
    p.add_argument("--home-tol", type=float, default=0.02, help="rad, max joint error to count as at home")
    p.add_argument("--max-targets", type=int, default=15,
                   help="targets JSON: pick at most this many, in file order (= binpick_real TOP_K)")
    args = p.parse_args()
    if not args.stdin and not args.targets:
        p.error("give a targets JSON or --stdin")

    cfg = load_config(default_config_path("real.yaml"))
    home = np.array(named_pose(cfg, "home"))

    rclpy.init()
    ee6d = load_ee6d(args.ee6d)
    node = ee6d.TargetPublisher()
    arm = ArmWatch(node, cfg.robot.joint_names, cfg.ros.joint_states_topic, move_tol=2e-3)
    home_pub = node.create_publisher(Empty, cfg.ros.home_request_topic, 10)
    ik_client = node.create_client(GetPositionIK, cfg.ros.compute_ik_service)
    at_home = lambda: arm.q is not None and np.abs(arm.q - home).max() < args.home_tol
    goal = np.array(named_pose(cfg, "goal"))                         # drop pose: passing it = the part was placed
    at_goal = lambda: arm.q is not None and np.abs(arm.q - goal).max() < 0.05
    abort = threading.Event()

    def home_now():
        """Pause: /go_home at once, also during a cycle (sequence_real_action aborts an approach, finishes a place)."""
        if at_home() and arm.still_for() > 1.0:
            return
        abort.set()
        home_pub.publish(Empty())
        print("home now: /go_home sent", flush=True)

    results = []
    try:
        err = None
        if not spin_until(node, lambda: arm.q is not None, 10.0):
            err = f"no {cfg.ros.joint_states_topic}: is terminal 1 (MoveIt + robot) running?"
        elif not spin_until(node, lambda: node.target_pub.get_subscription_count() > 0, 10.0):
            err = "nobody subscribes /target_pose: is terminal 2 (sequence_real_action) running?"
        elif args.stdin and not ik_client.wait_for_service(timeout_sec=10.0):
            err = f"no {cfg.ros.compute_ik_service} service: is terminal 1 (MoveIt move_group) running?"
        if err:
            if args.stdin:
                print("@@ " + json.dumps({"ready": False, "error": err}), flush=True)
            sys.exit(err)

        if args.stdin:
            print("@@ " + json.dumps({"ready": True}), flush=True)
            reader = StdinReader(home_now)
            reader.start()
            for t in stdin_targets(node, reader):
                if t.get("cmd") == "home":
                    status, dt = go_home(node, arm, at_home, home_pub, args)
                    print("@@ " + json.dumps({"cmd": "home", "status": status, "seconds": round(dt, 1)}), flush=True)
                    continue
                if t.get("cmd") == "check":
                    res = []
                    for tg in t["targets"]:
                        ok, why = check_target(node, ik_client, tg, cfg)
                        res.append({"rank": tg["rank"], "ok": ok, "reason": why})
                    print("@@ " + json.dumps({"cmd": "check", "results": res}), flush=True)
                    continue
                status, dt = pick_one(node, arm, at_home, t, args, abort=abort, at_goal=at_goal)
                results.append((t["rank"], status))
                print("@@ " + json.dumps({"rank": t["rank"], "status": status, "seconds": round(dt, 1)}), flush=True)
        else:
            targets = json.load(open(args.targets))["targets"]
            if len(targets) > args.max_targets:
                print(f"{len(targets)} targets in the file: picking the first {args.max_targets} (--max-targets)",
                      flush=True)
                targets = targets[:args.max_targets]
            for i, t in enumerate(targets, 1):
                status, _ = pick_one(node, arm, at_home, t, args, tag=f"[{i}/{len(targets)}] ")
                if status == "busy":
                    print("stopping", flush=True)
                    break
                results.append((t["rank"], status))
    except KeyboardInterrupt:
        print("\ninterrupted: no more targets are sent (the running cycle finishes in sequence_real_action)", flush=True)
    finally:
        print("\nsummary: " + (", ".join(f"#{k} {s}" for k, s in results) or "nothing sent"), flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
