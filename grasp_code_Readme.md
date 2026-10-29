# Side plate grasp: 실행 코드 정리

Photoneo 촬영 → 6D pose → TCP 목표 → Lite6 이동. 단위는 m, deg.

## 0. 환경 (처음 한 번)

```bash
# ROS (~/.bashrc 에 이미 있음)
source /opt/ros/jazzy/setup.bash
source ~/Projects/GDRNPP_bin_picking/lite6_control/xarm_ros2/install/setup.bash
source ~/Projects/GDRNPP_bin_picking/lite6_control/lite6_ws/install/setup.bash

# lite6_control 파이썬 코드를 고친 뒤 ros2 run 에 반영하려면 (lite6_ws 에서, src 안에서 하지 말 것)
cd ~/Projects/GDRNPP_bin_picking/lite6_control/lite6_ws
colcon build --symlink-install --packages-select my_lite6_control
```

TCP 길이: `~/Projects/GDRNPP_bin_picking/lite6_control/lite6_ws/src/my_lite6_control/config/robots/lite6.yaml` 의 `tcp_offset_m: [0.0, 0.0, 0.106]` (빌드 없이 노드 재시작으로 적용).

## 1. 카메라 → robot base 캘리브레이션 (카메라나 로봇을 옮겼을 때만)

```bash
cd ~/Projects/xarm_lite6/cam2robot_pos_calibration
source ~/Projects/photoneo-python-examples/GigE-V/harvesters/.venv/bin/activate

python test_camera.py      # 체커보드 검출 확인 (data/test_camera_*/detection.png)
python touch_collect.py    # 보드 고정, TCP 끝으로 코너 찍기
python touch_solve.py      # -> result/T_base_cam.yaml
python calib_test.py       # 보드 코너 위로 이동해서 오차 확인
```

새 행렬(`result/T_base_cam.yaml` 의 `matrix`)을 아래 세 곳에 넣습니다.

- `notebooks/results_6dpose_real.ipynb` 마지막 셀의 `T_CAM_BASE` (m)
- `camera/T_base_cam.json` (같은 행렬, 이동값만 mm)
- `~/Projects/GDRNPP_bin_picking/lite6_control/lite6_ws/src/install/.../my_lite6_control/obj_6d.py`, `test_cam.py` 의 `T_cam_base` (이 파일들을 쓸 때만)

## 2. 촬영 + pose + TCP 목표 (노트북)

`notebooks/results_6dpose_real.ipynb` 를 VSCode 에서 엽니다.

1. 처음 한 번: 위에서부터 모두 실행 (모델 로딩 약 10초).
2. 그다음: **측정 셀(`picks = run_cycle()`, 자동 집기 셀 바로 위)만 다시 실행**하면 새로 찍고 pick 을 출력합니다.
3. pick 마다 `ee_6d rpy :` / `ee_6d quat:` 줄을 출력합니다 (robot base, m, rpy = ee_6d.py 와 같은 intrinsic XYZ).
4. 마지막 셀 맨 위 설정: `MAX_TILT = 60.0` (이보다 기운 부품은 제외), `GRIP_YAW = "down"`
5. 결과 저장: `runs/realtime/<시각>/` (`picks.json`, `picks.csv`, `overlay.png`, ...)

다른 사람이 파일을 고쳤으면 노트북을 저장하기 전에 `Ctrl+Shift+P` → **File: Revert File** 로 다시 불러옵니다.

## 3. ROS 로 집기

```bash
# 터미널 1: MoveIt + 실제 로봇
ros2 launch xarm_moveit_config lite6_moveit_realmove_custom.launch.py robot_ip:=192.168.1.150

# 터미널 2: pick 시퀀스 (/target_pose 대기)
ros2 run my_lite6_control sequence_real_action robot.gripper.service_setup=/home/joon/Projects/GDRNPP_bin_picking/lite6_control/xarm_ros2/install/setup.bash

# 터미널 3: 목표 보내기 -> 노트북의 "ee_6d rpy :" 또는 "ee_6d quat:" 줄 뒤 숫자를 그대로 붙여 넣기
python3 ~/Projects/GDRNPP_bin_picking/lite6_control/lite6_ws/src/my_lite6_control/my_lite6_control/ee_6d.py
```

```
input target pos x y z [roll pitch yaw | qx qy qz qw]: 0.3203 0.0496 0.0229 -153.17 -23.56 5.70
```

`ee_6d.py` 의 rpy 는 intrinsic XYZ 입니다. 노트북의 `ee_6d` 줄을 그대로 넣으세요. `ros2 run my_lite6_control ee_6d` (예전 빌드, extrinsic xyz)는 쓰지 마세요.

### 자동으로 1~10번 집기

터미널 1, 2만 띄운 상태에서, 노트북의 측정 셀(`picks = run_cycle()`) 다음에 있는 **맨 마지막 셀(`auto_pick(picks)`)**을 실행합니다.

- `AUTO_PICK_RANKS = list(range(1, 11))` 순서대로 `ee_6d.py`의 publisher로 `/target_pose`를 보냅니다(`binpick/auto_pick_ee6d.py`, ROS python으로 실행).
- `/joint_states`를 보고 로봇이 home으로 돌아오면 다음 번호를 보냅니다.
- 로봇이 움직이지 않은 번호(안전 범위·IK 거부)나 도중에 멈춘 번호는 실패로 표시하고 건너뜁니다.
- 셀을 멈추면 다음 번호를 보내지 않습니다. 바로 멈추려면 터미널 2를 Ctrl+C 하세요.

터미널에서 직접 실행할 수도 있습니다(ROS 환경 source 후).

```bash
cd ~/Projects/GDRNPP_bin_6DOF_segmentation
/usr/bin/python3 binpick/auto_pick_ee6d.py runs/realtime/auto_pick_targets.json
```

### 2D 보정하며 집기 (`notebooks/results_6dpose_real_optimize.ipynb`)

1. 위에서부터 모두 실행합니다. 5번 셀이 `photoneo_server.py`(카메라 연결 유지, harvesters venv)를 띄웁니다.
2. 측정 셀(`picks = run_cycle()`)을 실행합니다. 3D 스캔과 2D 기준 영상을 같이 찍고, 위가 비어 있고 높은 부품부터 순서를 매깁니다(`PICK_ORDER = "top"`).
3. 맨 마지막 셀(`auto_pick_2d()`)을 실행합니다. 하나를 집을 때마다 2D로 다시 찍어 남은 pick 위치를 보정하거나 빼고, 잡을 수 있는 pick만 그림에 표시합니다. pick이 없으면 3D로 다시 찍습니다.

로봇 쪽은 `binpick/auto_pick_ee6d.py --stdin`(목표를 한 줄씩 받아 한 번씩 집음)으로 실행됩니다. 카메라 서버만 따로 시험하려면:

```bash
cd ~/Projects/GDRNPP_bin_6DOF_segmentation
printf '{"cmd":"3d"}\n{"cmd":"2d"}\n{"cmd":"quit"}\n' | \
  ~/Projects/photoneo-python-examples/GigE-V/harvesters/.venv/bin/python photoneo_server.py --out runs/realtime/capture
```

실제 자세 확인:

```bash
ros2 run tf2_ros tf2_echo link_base link_eef
```
