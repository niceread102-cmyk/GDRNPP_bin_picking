# 연구노트용 그림 (2026-10-16)

| 파일 | 내용 | 출처 |
|---|---|---|
| 01_depth_verification_before_photoneo_view59.jpg | 깊이 검증 전: bin 테두리·바깥 오검출 8개(주황) | results_photoneo_summary.ipynb §5 |
| 02_depth_verification_after_photoneo_view59.jpg | 깊이 검증 후: 부품 10개 모두 confident, 의사 정답과 1 mm 이내 | results_photoneo_summary.ipynb §5 |
| 03_detector_sim_trained_photoneo_view34.jpg | 시뮬레이션 학습 detector: 오른쪽 부품 2개 놓침(자홍) | results_photoneo_summary.ipynb §6 |
| 04_detector_finetuned_photoneo_view34.jpg | fine-tuning한 detector: 5개 모두 0.4~0.7 mm | results_photoneo_summary.ipynb §6 |
| 05_full_bin_real_capture_rgb.jpg | 가득 찬 실제 bin 원본(정답 없음) | results_photoneo_summary.ipynb §7 |
| 06_full_bin_depth_explained_map.jpg | 초록 = confident pose가 depth를 설명(67%), 빨강 = 설명 안 됨 | results_photoneo_summary.ipynb §7 |
| 07_pick_candidates_real_full_bin.jpg | pick 후보: 노란 원 = 손가락(구멍), 파란 원 = 그리퍼 몸체(치수 임시값) | results_photoneo_summary.ipynb §10 |
| 08_isaac_sim_6d_pose_output.jpg | Isaac Sim 집기용 pose 8개: 노란 원 = 출력 위치(위쪽 끝면), 축 x 빨강 y 초록 z 파랑 | results_summary_sim_grippose.ipynb §6 |
