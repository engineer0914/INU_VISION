<video width="100%" controls>
  <source src="INU_VISION_TEST.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>




카메라 열기

```jsx
sb
isreal
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p align_depth.enable:=true \
  -p enable_sync:=true \
  -p depth_module.depth_profile:=640x480x30 \
  -p rgb_camera.color_profile:=640x480x30
```

카메라 확인

```jsx
ros2 topic echo /camera/camera/color/camera_info --once
ros2 topic echo /camera/camera/aligned_depth_to_color/camera_info --once
```

검출중

```jsx
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

검출 결과 확인

```jsx
sb
is
ros2 run rqt_image_view rqt_image_view
```

포즈추정시작

```jsx
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: 'st_est'}"
```

valid확인

```jsx
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: 'vw'}"
```

특정객체 포즈 요구

```jsx
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: '2g'}"
```

포즈추정중지

```jsx
ros2 service call /vision/pose_estimator msgs_pkg/srv/PoseEstimator "{command: 'sp_est'}"
```