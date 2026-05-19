# 1.환경내 requirements 설치
```bash
# 현재 위치가 ~/ros2_ws 일 때
pip install -r src/vision/requirements.txt
```

# 2.카메라 열기

```bash
sb
is
ros2 run realsense2_camera realsense2_camera_node --ros-args \
-p enable_color:=true \
-p enable_depth:=true \
-p enable_infra1:=false \
-p enable_infra2:=false \
-p align_depth.enable:=true \
-p enable_sync:=true \
-p depth_module.depth_profile:="640x480x30" \
-p rgb_camera.color_profile:="640x480x30" \
-p enable_gyro:=false \
-p enable_accel:=false
```

# 2-1 카메라 상태 확인
토픽이 인식된다면 카메라는 정상입니다.

```bash
ros2 topic echo /camera/camera/color/camera_info --once
ros2 topic echo /camera/camera/aligned_depth_to_color/camera_info --once
```

# 3.검출 노드
아래를 동작 시키면, yolo 이후 처리가 같이 들어가 결과가 보입니다.

```bash
ros2 run vision yolo_3d_node --ros-args \
  -p model_path:=/home/orinagx/ros2_ws/src/vision/yolo_models/train1_05_04/weights/best.pt \
  -p color_topic:=/camera/camera/color/image_raw \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  -p roi_padding:=60 \
  -p use_ransac_plane:=true \
  -p use_image_contour:=true \
  -p clamp_final_rect_to_bbox:=true \
  -p max_outside_bbox_ratio:=0.10 \
  -p image_edge_percentile:=80.0 \
  -p clahe_clip_limit:=2.0 \
  -p clahe_tile_size:=8 \
  -p roi_contrast_alpha:=1.10 \
  -p roi_contrast_beta:=0.0 \
  -p object_height_threshold:=0.008 \
  -p min_depth_overlap_ratio:=0.20 \
  -p pca_ratio_threshold_rect:=1.35 \
  -p show_depth_side_by_side:=true \
  -p cv_main_text_scale:=0.72 \
  -p cv_main_text_thickness:=2 \
  -p cv_sub_text_scale:=0.45 \
  -p cv_sub_text_thickness:=1 \
  -p cv_text_line_gap:=24 \
  -p show_sub_debug_text:=true \
  -p min_final_rect_yolo_iou:=0.40 \
  -p use_shape_id_validation:=true \
  -p shape_square_aspect_max:=1.35 \
  -p shape_rectangle_aspect_min:=1.55 \
  -p shape_bbox_fill_square_threshold:=0.45 \
  -p use_pose_filter:=true \
  -p pose_filter_type:=median \
  -p pose_filter_window:=5 \
  -p publish_pose_plot_topics:=true \
  -p estimation_enabled_on_start:=true \
  -p estimation_center_top_n:=3
```

# 4.검출 결과 확인
rqt 이미지 뷰어로 현재 상태를 확인 가능

```bash
sb
is
ros2 run rqt_image_view rqt_image_view
```



# valid확인
현재 이미지내 어떤 객체들이 보이는지 확인

```bash
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: 'vw'}"
```

# 특정객체 포즈 요구
2*2 초록 브릭의 4D출력 요청시 다음 이용.
이외 comand: 측 문자열 변경시 사용가능 (예:2r 2g 2b 2y 4r 4g 4b 4y)

```bash
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: '2g'}"
```


---
비 중요
---

# 포즈추정중지
GPU 사용량 최적화를 위해 포즈 추정 중지시
```bash
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: 'sp_est'}"
```

# 포즈추정 다시 시작
기본적으로 1번 실행시 같이 실행되지만 최적화를 위해 잠시 중지후 다시 시작할때

```bash
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: 'st_est'}"
```
