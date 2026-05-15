#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

from msgs_pkg.srv import GetObjectPose


class Yolo3DNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_node')

        # ============================================================
        # Parameters
        # ============================================================
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')

        self.declare_parameter('model_path', 'yolo_models/train1_05_04/weights/best.pt')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('publish_debug_image', True)

        # ROI / depth parameters
        self.declare_parameter('roi_padding', 60)
        self.declare_parameter('bbox_inner_margin', 0)

        self.declare_parameter('valid_depth_min', 0.15)
        self.declare_parameter('valid_depth_max', 1.20)

        self.declare_parameter('floor_percentile', 90.0)
        self.declare_parameter('object_height_threshold', 0.008)

        self.declare_parameter('min_component_area', 150)
        self.declare_parameter('min_contour_area', 200)

        # RANSAC plane parameters
        self.declare_parameter('use_ransac_plane', True)
        self.declare_parameter('ransac_iterations', 120)
        self.declare_parameter('ransac_distance_threshold', 0.004)
        self.declare_parameter('ransac_max_points', 3000)
        self.declare_parameter('ransac_min_floor_points', 100)
        self.declare_parameter('ransac_min_inliers', 80)

        # Image contour parameters
        self.declare_parameter('use_image_contour', True)
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_size', 8)
        self.declare_parameter('image_edge_percentile', 85.0)
        self.declare_parameter('image_contour_min_area', 80.0)
        self.declare_parameter('image_search_margin', 8)

        # Contour validation parameters
        self.declare_parameter('pca_ratio_threshold_rect', 1.35)
        self.declare_parameter('max_yaw_diff_deg', 25.0)
        self.declare_parameter('min_depth_overlap_ratio', 0.20)

        # YOLO bbox constraint
        self.declare_parameter('max_outside_bbox_ratio', 0.10)
        self.declare_parameter('clamp_final_rect_to_bbox', True)

        # Final pose fusion
        self.declare_parameter('center_weight_image', 0.60)
        self.declare_parameter('center_weight_yolo', 0.40)

        # Debug image
        self.declare_parameter('show_depth_side_by_side', True)
        self.declare_parameter('print_pose_each_frame', False)

        # ============================================================
        # CV text style parameters
        # 여기서 화면에 표시되는 XYZ/YAW 글자 크기 조절
        # ============================================================
        self.declare_parameter('cv_text_scale', 0.75)
        self.declare_parameter('cv_text_thickness', 2)
        self.declare_parameter('cv_text_line_gap', 28)

        # ============================================================
        # Read parameters
        # ============================================================
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value

        self.conf_thres = float(self.get_parameter('conf_thres').value)

        self.roi_padding = int(self.get_parameter('roi_padding').value)
        self.bbox_inner_margin = int(self.get_parameter('bbox_inner_margin').value)

        self.valid_depth_min = float(self.get_parameter('valid_depth_min').value)
        self.valid_depth_max = float(self.get_parameter('valid_depth_max').value)

        self.floor_percentile = float(self.get_parameter('floor_percentile').value)
        self.object_height_threshold = float(self.get_parameter('object_height_threshold').value)

        self.min_component_area = int(self.get_parameter('min_component_area').value)
        self.min_contour_area = int(self.get_parameter('min_contour_area').value)

        self.use_ransac_plane = bool(self.get_parameter('use_ransac_plane').value)
        self.ransac_iterations = int(self.get_parameter('ransac_iterations').value)
        self.ransac_distance_threshold = float(self.get_parameter('ransac_distance_threshold').value)
        self.ransac_max_points = int(self.get_parameter('ransac_max_points').value)
        self.ransac_min_floor_points = int(self.get_parameter('ransac_min_floor_points').value)
        self.ransac_min_inliers = int(self.get_parameter('ransac_min_inliers').value)

        self.use_image_contour = bool(self.get_parameter('use_image_contour').value)
        self.clahe_clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        self.clahe_tile_size = int(self.get_parameter('clahe_tile_size').value)
        self.image_edge_percentile = float(self.get_parameter('image_edge_percentile').value)
        self.image_contour_min_area = float(self.get_parameter('image_contour_min_area').value)
        self.image_search_margin = int(self.get_parameter('image_search_margin').value)

        self.pca_ratio_threshold_rect = float(self.get_parameter('pca_ratio_threshold_rect').value)
        self.max_yaw_diff_deg = float(self.get_parameter('max_yaw_diff_deg').value)
        self.min_depth_overlap_ratio = float(self.get_parameter('min_depth_overlap_ratio').value)

        self.max_outside_bbox_ratio = float(self.get_parameter('max_outside_bbox_ratio').value)
        self.clamp_final_rect_to_bbox = bool(self.get_parameter('clamp_final_rect_to_bbox').value)

        self.center_weight_image = float(self.get_parameter('center_weight_image').value)
        self.center_weight_yolo = float(self.get_parameter('center_weight_yolo').value)

        self.show_depth_side_by_side = bool(self.get_parameter('show_depth_side_by_side').value)
        self.print_pose_each_frame = bool(self.get_parameter('print_pose_each_frame').value)

        self.cv_text_scale = float(self.get_parameter('cv_text_scale').value)
        self.cv_text_thickness = int(self.get_parameter('cv_text_thickness').value)
        self.cv_text_line_gap = int(self.get_parameter('cv_text_line_gap').value)

        # Normalize center weights
        weight_sum = self.center_weight_image + self.center_weight_yolo
        if weight_sum <= 1e-6:
            self.center_weight_image = 0.6
            self.center_weight_yolo = 0.4
        else:
            self.center_weight_image /= weight_sum
            self.center_weight_yolo /= weight_sum

        # ============================================================
        # YOLO Load
        # ============================================================
        pkg_share_dir = get_package_share_directory('vision')
        abs_model_path = os.path.join(
            pkg_share_dir,
            self.get_parameter('model_path').value
        )

        self.get_logger().info(f"YOLO 모델 로드 중: {abs_model_path}")
        self.model = YOLO(abs_model_path)
        self.model.to(self.get_parameter('device').value)

        # ============================================================
        # Command mapping
        # ============================================================
        self.cmd_to_cls = {
            '2b': '2x2_blue',
            '2r': '2x2_red',
            '2g': '2x2_green',
            '2y': '2x2_yellow',
            '4b': '4x2_blue',
            '4r': '4x2_red',
            '4g': '4x2_green',
            '4y': '4x2_yellow'
        }

        # ============================================================
        # State
        # ============================================================
        self.cv_bridge = CvBridge()
        self.intrinsics = None
        self.latest_objects = []

        # ============================================================
        # ROS Communication
        # ============================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.color_sub = message_filters.Subscriber(
            self,
            Image,
            self.color_topic,
            qos_profile=qos
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.info_callback,
            10
        )

        if self.get_parameter('publish_debug_image').value:
            self.debug_pub = self.create_publisher(
                Image,
                '/vision/debug_image',
                10
            )

        self.srv = self.create_service(
            GetObjectPose,
            '/vision/get_object_pose',
            self.handle_get_pose
        )

        self.get_logger().info("YOLO + ImageContour + Depth/RANSAC pose node 시작.")
        self.get_logger().info(f"color_topic       : {self.color_topic}")
        self.get_logger().info(f"depth_topic       : {self.depth_topic}")
        self.get_logger().info(f"camera_info_topic : {self.info_topic}")

    # ============================================================
    # ROS callbacks
    # ============================================================
    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': float(msg.k[0]),
                'fy': float(msg.k[4]),
                'ppx': float(msg.k[2]),
                'ppy': float(msg.k[5]),
                'width': int(msg.width),
                'height': int(msg.height)
            }

            self.get_logger().info(
                f"카메라 정보 수신 완료: "
                f"{msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"ppx={msg.k[2]:.2f}, ppy={msg.k[5]:.2f}"
            )

    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None:
            return

        try:
            color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_raw = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

            # 내부 계산은 meter로 유지
            if depth_raw.dtype == np.uint16:
                depth_m = depth_raw.astype(np.float32) * 0.001
            else:
                depth_m = depth_raw.astype(np.float32)

        except Exception as e:
            self.get_logger().warn(f"이미지 변환 에러: {e}")
            return

        img_h, img_w = color_img.shape[:2]

        if depth_m.shape[0] != img_h or depth_m.shape[1] != img_w:
            self.get_logger().warn(
                f"Color/Depth 해상도 불일치: "
                f"color={img_w}x{img_h}, depth={depth_m.shape[1]}x{depth_m.shape[0]}. "
                f"반드시 /camera/camera/aligned_depth_to_color/image_raw 를 사용하세요."
            )
            return

        results = self.model(
            color_img,
            conf=self.conf_thres,
            verbose=False
        )[0]

        debug_img = color_img.copy()
        depth_vis = self.make_depth_colormap(depth_m)
        current_frame_objects = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            x1 = self.clamp(x1, 0, img_w - 1)
            x2 = self.clamp(x2, 0, img_w - 1)
            y1 = self.clamp(y1, 0, img_h - 1)
            y2 = self.clamp(y2, 0, img_h - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            # 초록색: YOLO bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # padding ROI
            x1p, y1p, x2p, y2p = self.expand_bbox(
                x1, y1, x2, y2,
                self.roi_padding,
                img_w,
                img_h
            )

            # 파란색: padding ROI
            cv2.rectangle(debug_img, (x1p, y1p), (x2p, y2p), (255, 0, 0), 1)

            roi_color = color_img[y1p:y2p, x1p:x2p]
            roi_depth_m = depth_m[y1p:y2p, x1p:x2p]

            core_bbox_in_roi = (
                x1 - x1p,
                y1 - y1p,
                x2 - x1p,
                y2 - y1p
            )

            # ========================================================
            # 1) Depth/RANSAC object mask for Z and validation
            # ========================================================
            if self.use_ransac_plane:
                depth_mask, depth_debug = self.make_ransac_plane_object_mask(
                    roi_depth_m=roi_depth_m,
                    core_bbox=core_bbox_in_roi,
                    x_offset=x1p,
                    y_offset=y1p
                )

                if np.count_nonzero(depth_mask) == 0:
                    depth_mask, depth_debug = self.get_depth_object_mask_limited_to_bbox(
                        roi_depth_m,
                        core_bbox_in_roi
                    )
                    depth_debug['method'] = 'percentile_fallback'
            else:
                depth_mask, depth_debug = self.get_depth_object_mask_limited_to_bbox(
                    roi_depth_m,
                    core_bbox_in_roi
                )
                depth_debug['method'] = 'percentile'

            # ========================================================
            # 2) Main rect from image contour
            # ========================================================
            selected = None

            if self.use_image_contour:
                selected = self.estimate_rect_from_image_contour(
                    roi_color=roi_color,
                    core_bbox=core_bbox_in_roi,
                    depth_mask=depth_mask,
                    cls_name=cls_name
                )

            # ========================================================
            # 3) Fallback: depth contour rect
            # ========================================================
            if selected is None:
                selected = self.estimate_rect_from_depth_mask(
                    depth_mask=depth_mask,
                    cls_name=cls_name,
                    core_bbox=core_bbox_in_roi
                )

                if selected is not None:
                    selected['source'] = 'depth_fallback'

            valid_pose = False
            X = Y = Z = 0.0     # 내부 저장은 meter
            yaw = 0.0           # degree
            cx = cy = 0
            area = 0.0
            height_mm = 0.0
            rect_source = 'none'
            pca_ratio = 0.0
            yaw_mode = 'normal'
            outside_ratio = 0.0

            if selected is not None:
                rect = selected['rect']

                # 최종 노란 박스가 YOLO bbox 밖으로 나가지 않도록 clamp
                if self.clamp_final_rect_to_bbox:
                    rect = self.clamp_rect_to_bbox(rect, core_bbox_in_roi)

                rect_source = str(selected.get('source', 'image'))
                pca_ratio = float(selected.get('pca_ratio', 0.0))
                outside_ratio = self.rect_outside_bbox_ratio(rect, core_bbox_in_roi)

                # clamp 이후 yaw/area 다시 계산
                yaw_rect = self.normalize_rect_angle(rect)

                if self.is_square_class(cls_name):
                    yaw = self.square_yaw_from_yolo_bottom_left(
                        rect,
                        core_bbox_in_roi
                    )
                    yaw_mode = 'square_yolo_bl_0_90'
                else:
                    yaw = yaw_rect
                    yaw_mode = 'normal'

                area = float(max(rect[1][0] * rect[1][1], 1.0))

                rect_cx_roi = float(rect[0][0])
                rect_cy_roi = float(rect[0][1])

                yolo_cx_roi = (core_bbox_in_roi[0] + core_bbox_in_roi[2]) / 2.0
                yolo_cy_roi = (core_bbox_in_roi[1] + core_bbox_in_roi[3]) / 2.0

                final_cx_roi = (
                    self.center_weight_image * rect_cx_roi +
                    self.center_weight_yolo * yolo_cx_roi
                )
                final_cy_roi = (
                    self.center_weight_image * rect_cy_roi +
                    self.center_weight_yolo * yolo_cy_roi
                )

                cx = int(round(final_cx_roi + x1p))
                cy = int(round(final_cy_roi + y1p))

                # Z는 depth mask 내부 median을 우선 사용
                z = self.get_object_median_depth(roi_depth_m, depth_mask)

                # depth mask가 실패했으면 selected rect 내부 depth로 fallback
                if z <= 0.0:
                    z = self.get_rect_median_depth(
                        roi_depth_m=roi_depth_m,
                        rect=rect,
                        floor_depth=depth_debug.get('floor_depth', 0.0)
                    )

                if z > 0.0 and self.is_pixel_inside(cx, cy, img_w, img_h):
                    X = (cx - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                    Y = (cy - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                    Z = z
                    valid_pose = True

                    floor_d = depth_debug.get('floor_depth', 0.0)
                    body_d = depth_debug.get('body_depth', 0.0)

                    if floor_d > 0.0 and body_d > 0.0:
                        height_mm = (floor_d - body_d) * 1000.0
                    else:
                        height_mm = float(depth_debug.get('height_mm', 0.0))

                    box_pts = cv2.boxPoints(rect)
                    box_pts = np.intp(box_pts)
                    box_pts[:, 0] += x1p
                    box_pts[:, 1] += y1p

                    # 노란색: 최종 선택 rect
                    cv2.drawContours(debug_img, [box_pts], 0, (0, 255, 255), 2)
                    cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)

                    cv2.drawContours(depth_vis, [box_pts], 0, (0, 255, 255), 2)
                    cv2.circle(depth_vis, (cx, cy), 5, (0, 0, 255), -1)

            current_frame_objects.append({
                'cls_name': cls_name,
                'conf': conf,
                'valid': valid_pose,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'cx': cx,
                'cy': cy,
                'X': float(X),  # meter
                'Y': float(Y),  # meter
                'Z': float(Z),  # meter
                'rz': float(yaw),
                'height_mm': float(height_mm),
                'area': float(area),
                'floor_depth': float(depth_debug.get('floor_depth', 0.0)),
                'body_depth': float(depth_debug.get('body_depth', 0.0)),
                'method': str(depth_debug.get('method', 'unknown')),
                'rect_source': rect_source,
                'pca_ratio': pca_ratio,
                'yaw_mode': yaw_mode,
                'outside_ratio': float(outside_ratio)
            })

            if valid_pose:
                self.draw_debug_text(
                    debug_img=debug_img,
                    x1=x1,
                    y1=y1,
                    y2=y2,
                    cls_name=cls_name,
                    conf=conf,
                    X=X,
                    Y=Y,
                    Z=Z,
                    yaw=yaw,
                    height_mm=height_mm,
                    area=area,
                    depth_debug=depth_debug,
                    rect_source=rect_source,
                    pca_ratio=pca_ratio,
                    yaw_mode=yaw_mode,
                    outside_ratio=outside_ratio
                )

                if self.print_pose_each_frame:
                    self.get_logger().info(
                        f"{cls_name} "
                        f"x={X*1000.0:.1f} mm, y={Y*1000.0:.1f} mm, z={Z*1000.0:.1f} mm, "
                        f"yaw={yaw:.2f} deg, "
                        f"src={rect_source}, yaw_mode={yaw_mode}, "
                        f"pcaR={pca_ratio:.2f}, outside={outside_ratio:.3f}, "
                        f"H={height_mm:.1f} mm"
                    )
            else:
                cv2.putText(
                    debug_img,
                    f"{cls_name} pose failed",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.cv_text_scale,
                    (0, 0, 255),
                    self.cv_text_thickness
                )

        self.latest_objects = current_frame_objects

        if self.get_parameter('publish_debug_image').value:
            if self.show_depth_side_by_side:
                debug_out = np.hstack([debug_img, depth_vis])
            else:
                debug_out = debug_img

            self.debug_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(debug_out, 'bgr8')
            )

    # ============================================================
    # Service
    # ============================================================
    def handle_get_pose(self, request, response):
        response.success = False
        response.x = 0.0
        response.y = 0.0
        response.z = 0.0
        response.rz = 0.0
        response.view_result = ""

        if self.intrinsics is None:
            self.get_logger().warn("카메라 정보가 아직 없습니다.")
            return response

        cmd = request.command
        self.get_logger().info(f"서비스 요청 수신: {cmd}")

        if cmd == 'vw':
            counts = {k: 0 for k in self.cmd_to_cls.keys()}

            for obj in self.latest_objects:
                for short_cmd, cls_name in self.cmd_to_cls.items():
                    if obj['cls_name'] == cls_name:
                        counts[short_cmd] += 1

            response.view_result = "//".join(
                [f"{k}_{v}" for k, v in counts.items() if v > 0]
            )
            response.success = True
            return response

        if cmd not in self.cmd_to_cls:
            self.get_logger().warn(f"알 수 없는 command: {cmd}")
            return response

        target_cls = self.cmd_to_cls[cmd]

        best_obj = None
        min_dist = float('inf')

        img_cx = self.intrinsics['width'] / 2.0
        img_cy = self.intrinsics['height'] / 2.0

        for obj in self.latest_objects:
            if obj['cls_name'] != target_cls:
                continue
            if not obj['valid']:
                continue

            box_cx = (obj['x1'] + obj['x2']) / 2.0
            box_cy = (obj['y1'] + obj['y2']) / 2.0

            dist = math.hypot(box_cx - img_cx, box_cy - img_cy)

            if dist < min_dist:
                min_dist = dist
                best_obj = obj

        if best_obj is None:
            self.get_logger().info(f"{target_cls} 중 valid pose 객체가 없습니다.")
            return response

        # ============================================================
        # Service response는 mm 단위로 반환
        # 내부 latest_objects는 meter 유지
        # ============================================================
        response.x = float(best_obj['X'] * 1000.0)
        response.y = float(best_obj['Y'] * 1000.0)
        response.z = float(best_obj['Z'] * 1000.0)
        response.rz = float(best_obj['rz'])
        response.success = True

        self.get_logger().info(
            f"[SERVICE RESULT] {target_cls}: "
            f"x={response.x:.1f} mm, y={response.y:.1f} mm, "
            f"z={response.z:.1f} mm, yaw={response.rz:.2f} deg, "
            f"src={best_obj.get('rect_source', 'unknown')}, "
            f"mode={best_obj.get('yaw_mode', 'unknown')}, "
            f"pcaR={best_obj.get('pca_ratio', 0.0):.2f}, "
            f"outside={best_obj.get('outside_ratio', 0.0):.3f}"
        )

        return response

    # ============================================================
    # Image contour estimation
    # ============================================================
    def estimate_rect_from_image_contour(self, roi_color, core_bbox, depth_mask, cls_name):
        h, w = roi_color.shape[:2]
        bx1, by1, bx2, by2 = core_bbox

        bx1 = self.clamp(bx1, 0, w - 1)
        by1 = self.clamp(by1, 0, h - 1)
        bx2 = self.clamp(bx2, 0, w - 1)
        by2 = self.clamp(by2, 0, h - 1)

        if bx2 <= bx1 or by2 <= by1:
            return None

        edge = self.make_lab_gradient_edge(roi_color)

        search_mask = np.zeros((h, w), dtype=np.uint8)

        sx1 = self.clamp(bx1 - self.image_search_margin, 0, w - 1)
        sy1 = self.clamp(by1 - self.image_search_margin, 0, h - 1)
        sx2 = self.clamp(bx2 + self.image_search_margin, 0, w - 1)
        sy2 = self.clamp(by2 + self.image_search_margin, 0, h - 1)

        search_mask[sy1:sy2, sx1:sx2] = 255

        edge = cv2.bitwise_and(edge, search_mask)

        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            edge,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        candidates = []

        for cnt in contours:
            cnt_area = float(cv2.contourArea(cnt))
            if cnt_area < self.image_contour_min_area:
                continue

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            rect_area = float(max(rw * rh, 1.0))

            if rect_area < self.image_contour_min_area:
                continue

            outside_ratio = self.rect_outside_bbox_ratio(rect, core_bbox)

            if outside_ratio > self.max_outside_bbox_ratio:
                continue

            yaw_rect = self.normalize_rect_angle(rect)
            yaw_pca, pca_ratio, pca_reliable = self.compute_pca_yaw_with_confidence(cnt)
            yaw_diff = self.angle_diff_axis_deg(yaw_rect, yaw_pca)

            if self.is_square_class(cls_name):
                yaw = yaw_rect
                yaw_mode = 'square_yolo_bl_0_90'
                pca_penalty = 0.0
            else:
                yaw_mode = 'normal'
                yaw = yaw_rect

                if pca_reliable and yaw_diff <= self.max_yaw_diff_deg:
                    yaw = yaw_rect
                    pca_penalty = yaw_diff / max(self.max_yaw_diff_deg, 1.0)
                elif not pca_reliable:
                    pca_penalty = 1.0
                else:
                    continue

            overlap_ratio = self.compute_rect_depth_overlap_ratio(rect, depth_mask)

            if np.count_nonzero(depth_mask) > 0 and overlap_ratio < self.min_depth_overlap_ratio:
                continue

            bbox_score = self.score_rect_with_yolo_bbox(rect, core_bbox, cls_name)
            overlap_score = 1.0 - overlap_ratio

            total_score = (
                bbox_score +
                0.8 * pca_penalty +
                0.8 * overlap_score +
                1.2 * outside_ratio
            )

            candidates.append({
                'rect': rect,
                'yaw': float(yaw),
                'area': rect_area,
                'score': float(total_score),
                'source': 'image',
                'pca_ratio': float(pca_ratio),
                'yaw_diff': float(yaw_diff),
                'overlap_ratio': float(overlap_ratio),
                'outside_ratio': float(outside_ratio),
                'yaw_mode': yaw_mode
            })

        if not candidates:
            return None

        candidates.sort(key=lambda c: c['score'])
        return candidates[0]

    def make_lab_gradient_edge(self, roi_color):
        lab = cv2.cvtColor(roi_color, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        tile = max(2, self.clahe_tile_size)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(tile, tile)
        )
        L_eq = clahe.apply(L)

        grad_L = self.scharr_mag_u8(L_eq)
        grad_A = self.scharr_mag_u8(A)
        grad_B = self.scharr_mag_u8(B)

        grad = np.maximum.reduce([grad_L, grad_A, grad_B])

        vals = grad.reshape(-1)
        if len(vals) < 10:
            return np.zeros(grad.shape, dtype=np.uint8)

        thr = float(np.percentile(vals, self.image_edge_percentile))

        edge = np.zeros(grad.shape, dtype=np.uint8)
        edge[grad >= thr] = 255

        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        return edge

    def scharr_mag_u8(self, channel):
        ch = cv2.GaussianBlur(channel, (5, 5), 0)

        gx = cv2.Scharr(ch, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(ch, cv2.CV_32F, 0, 1)

        mag = cv2.magnitude(gx, gy)
        mag_u8 = cv2.convertScaleAbs(mag)

        return mag_u8

    # ============================================================
    # Depth fallback rect
    # ============================================================
    def estimate_rect_from_depth_mask(self, depth_mask, cls_name, core_bbox=None):
        contours, _ = cv2.findContours(
            depth_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))

        if area < self.min_contour_area:
            return None

        rect = cv2.minAreaRect(cnt)

        if core_bbox is not None and self.clamp_final_rect_to_bbox:
            rect = self.clamp_rect_to_bbox(rect, core_bbox)

        yaw_rect = self.normalize_rect_angle(rect)

        if self.is_square_class(cls_name):
            if core_bbox is not None:
                yaw = self.square_yaw_from_yolo_bottom_left(rect, core_bbox)
                yaw_mode = 'square_yolo_bl_0_90'
            else:
                yaw = abs(yaw_rect)
                if yaw > 90.0:
                    yaw = 180.0 - yaw
                yaw_mode = 'square_0_90_fallback'
        else:
            yaw = yaw_rect
            yaw_mode = 'normal'

        _, pca_ratio, _ = self.compute_pca_yaw_with_confidence(cnt)

        return {
            'rect': rect,
            'yaw': float(yaw),
            'area': float(max(rect[1][0] * rect[1][1], 1.0)),
            'source': 'depth',
            'pca_ratio': float(pca_ratio),
            'yaw_mode': yaw_mode,
            'outside_ratio': 0.0
        }

    # ============================================================
    # Depth / RANSAC object mask
    # ============================================================
    def make_ransac_plane_object_mask(self, roi_depth_m, core_bbox, x_offset, y_offset):
        h, w = roi_depth_m.shape[:2]

        debug = {
            'method': 'ransac',
            'plane_found': False,
            'plane': None,
            'inlier_count': 0,
            'floor_depth': 0.0,
            'body_depth': 0.0,
            'height_mm': 0.0
        }

        if h <= 0 or w <= 0:
            return np.zeros((h, w), dtype=np.uint8), debug

        valid = np.logical_and(
            roi_depth_m > self.valid_depth_min,
            roi_depth_m < self.valid_depth_max
        )

        if np.count_nonzero(valid) < self.ransac_min_floor_points:
            return np.zeros((h, w), dtype=np.uint8), debug

        bx1, by1, bx2, by2 = core_bbox

        m = self.bbox_inner_margin
        bx1 = self.clamp(bx1 + m, 0, w - 1)
        by1 = self.clamp(by1 + m, 0, h - 1)
        bx2 = self.clamp(bx2 - m, 0, w - 1)
        by2 = self.clamp(by2 - m, 0, h - 1)

        if bx2 <= bx1 or by2 <= by1:
            return np.zeros((h, w), dtype=np.uint8), debug

        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[by1:by2, bx1:bx2] = 255

        floor_candidate = np.logical_and(valid, core_mask == 0)

        if np.count_nonzero(floor_candidate) < self.ransac_min_floor_points:
            floor_candidate = valid.copy()

        ys, xs = np.where(floor_candidate)

        if len(xs) < self.ransac_min_floor_points:
            return np.zeros((h, w), dtype=np.uint8), debug

        if len(xs) > self.ransac_max_points:
            idx = np.random.choice(len(xs), self.ransac_max_points, replace=False)
            xs_s = xs[idx]
            ys_s = ys[idx]
        else:
            xs_s = xs
            ys_s = ys

        z_s = roi_depth_m[ys_s, xs_s]
        pts = self.deproject_roi_points(xs_s, ys_s, z_s, x_offset, y_offset)

        plane, inlier_count = self.ransac_plane_numpy(
            pts,
            iterations=self.ransac_iterations,
            threshold=self.ransac_distance_threshold
        )

        debug['plane_found'] = plane is not None
        debug['plane'] = plane
        debug['inlier_count'] = int(inlier_count)

        if plane is None or inlier_count < self.ransac_min_inliers:
            return np.zeros((h, w), dtype=np.uint8), debug

        ys_all, xs_all = np.where(valid)
        z_all = roi_depth_m[ys_all, xs_all]
        pts_all = self.deproject_roi_points(xs_all, ys_all, z_all, x_offset, y_offset)

        a, b, c, d = plane
        normal = np.array([a, b, c], dtype=np.float32)
        denom = float(np.linalg.norm(normal))

        if denom < 1e-8:
            return np.zeros((h, w), dtype=np.uint8), debug

        signed_dist = (pts_all @ normal + d) / denom

        dist_img = np.zeros((h, w), dtype=np.float32)
        dist_img[ys_all, xs_all] = signed_dist

        core_valid_dist = dist_img[np.logical_and(core_mask > 0, valid)]

        if len(core_valid_dist) > 10:
            med_core = float(np.median(core_valid_dist))
            if med_core < 0:
                signed_dist = -signed_dist
                dist_img = -dist_img
                plane = (-a, -b, -c, -d)
                debug['plane'] = plane

        object_mask = np.zeros((h, w), dtype=np.uint8)

        obj_points = signed_dist > self.object_height_threshold
        object_mask[ys_all[obj_points], xs_all[obj_points]] = 255

        object_mask = cv2.bitwise_and(object_mask, core_mask)

        object_mask = self.clean_mask(object_mask)
        object_mask = self.keep_largest_component(
            object_mask,
            min_area=self.min_component_area
        )

        obj_depth = roi_depth_m[object_mask > 0]
        obj_depth = obj_depth[
            np.logical_and(
                obj_depth > self.valid_depth_min,
                obj_depth < self.valid_depth_max
            )
        ]

        if len(obj_depth) > 0:
            debug['body_depth'] = float(np.median(obj_depth))

        floor_depth_values = roi_depth_m[floor_candidate]
        floor_depth_values = floor_depth_values[
            np.logical_and(
                floor_depth_values > self.valid_depth_min,
                floor_depth_values < self.valid_depth_max
            )
        ]

        if len(floor_depth_values) > 0:
            debug['floor_depth'] = float(np.median(floor_depth_values))

        if np.count_nonzero(object_mask) > 0:
            obj_dist = dist_img[object_mask > 0]
            if len(obj_dist) > 0:
                debug['height_mm'] = float(np.median(obj_dist) * 1000.0)

        return object_mask, debug

    def deproject_roi_points(self, xs, ys, zs, x_offset, y_offset):
        u = xs.astype(np.float32) + float(x_offset)
        v = ys.astype(np.float32) + float(y_offset)
        z = zs.astype(np.float32)

        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']

        X = (u - ppx) * z / fx
        Y = (v - ppy) * z / fy
        Z = z

        pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
        return pts

    def ransac_plane_numpy(self, pts, iterations=120, threshold=0.004):
        if pts is None or pts.shape[0] < 3:
            return None, 0

        n = pts.shape[0]
        best_plane = None
        best_inlier_count = 0

        for _ in range(iterations):
            try:
                ids = np.random.choice(n, 3, replace=False)
            except ValueError:
                break

            p1, p2, p3 = pts[ids]

            v1 = p2 - p1
            v2 = p3 - p1

            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-8:
                continue

            normal = normal / norm
            d = -float(np.dot(normal, p1))

            dist = np.abs(pts @ normal + d)
            inlier_count = int(np.count_nonzero(dist < threshold))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_plane = (
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(d)
                )

        return best_plane, best_inlier_count

    # ============================================================
    # Percentile fallback object mask
    # ============================================================
    def get_depth_object_mask_limited_to_bbox(self, roi_depth_m, core_bbox):
        debug = {
            'method': 'percentile',
            'floor_depth': 0.0,
            'body_depth': 0.0,
            'height_mm': 0.0,
            'inlier_count': 0
        }

        h, w = roi_depth_m.shape[:2]

        if h <= 0 or w <= 0:
            return np.zeros((h, w), dtype=np.uint8), debug

        valid = np.logical_and(
            roi_depth_m > self.valid_depth_min,
            roi_depth_m < self.valid_depth_max
        )

        if np.count_nonzero(valid) < 50:
            return np.zeros((h, w), dtype=np.uint8), debug

        valid_depth = roi_depth_m[valid]

        floor_depth = float(np.percentile(valid_depth, self.floor_percentile))
        debug['floor_depth'] = floor_depth

        bx1, by1, bx2, by2 = core_bbox

        m = self.bbox_inner_margin
        bx1 = self.clamp(bx1 + m, 0, w - 1)
        by1 = self.clamp(by1 + m, 0, h - 1)
        bx2 = self.clamp(bx2 - m, 0, w - 1)
        by2 = self.clamp(by2 - m, 0, h - 1)

        core_mask = np.zeros((h, w), dtype=np.uint8)

        if bx2 <= bx1 or by2 <= by1:
            return np.zeros((h, w), dtype=np.uint8), debug

        core_mask[by1:by2, bx1:bx2] = 255

        object_candidate = np.logical_and(
            valid,
            roi_depth_m < floor_depth - self.object_height_threshold
        ).astype(np.uint8) * 255

        object_mask = cv2.bitwise_and(object_candidate, core_mask)

        object_mask = self.clean_mask(object_mask)
        object_mask = self.keep_largest_component(
            object_mask,
            min_area=self.min_component_area
        )

        obj_depth = roi_depth_m[object_mask > 0]
        obj_depth = obj_depth[
            np.logical_and(
                obj_depth > self.valid_depth_min,
                obj_depth < self.valid_depth_max
            )
        ]

        if len(obj_depth) > 0:
            body_depth = float(np.median(obj_depth))
            debug['body_depth'] = body_depth
            debug['height_mm'] = float((floor_depth - body_depth) * 1000.0)

        return object_mask, debug

    def get_object_median_depth(self, roi_depth_m, object_mask):
        vals = roi_depth_m[object_mask > 0]
        vals = vals[
            np.logical_and(
                vals > self.valid_depth_min,
                vals < self.valid_depth_max
            )
        ]

        if len(vals) < 5:
            return 0.0

        return float(np.median(vals))

    def get_rect_median_depth(self, roi_depth_m, rect, floor_depth=0.0):
        h, w = roi_depth_m.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.intp(box_pts)
        cv2.drawContours(mask, [box_pts], 0, 255, thickness=-1)

        vals = roi_depth_m[mask > 0]
        vals = vals[
            np.logical_and(
                vals > self.valid_depth_min,
                vals < self.valid_depth_max
            )
        ]

        if floor_depth > 0.0:
            vals = vals[vals < floor_depth - self.object_height_threshold]

        if len(vals) < 5:
            return 0.0

        return float(np.median(vals))

    def keep_largest_component(self, mask, min_area=150):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        if num_labels <= 1:
            return np.zeros_like(mask, dtype=np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]

        if len(areas) == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        largest_label = int(np.argmax(areas)) + 1
        largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])

        if largest_area < min_area:
            return np.zeros_like(mask, dtype=np.uint8)

        out = np.zeros_like(mask, dtype=np.uint8)
        out[labels == largest_label] = 255

        return out

    # ============================================================
    # YOLO bbox hard constraint
    # ============================================================
    def clamp_rect_to_bbox(self, rect, bbox):
        x1, y1, x2, y2 = bbox

        box_pts = cv2.boxPoints(rect).astype(np.float32)

        box_pts[:, 0] = np.clip(box_pts[:, 0], x1, x2)
        box_pts[:, 1] = np.clip(box_pts[:, 1], y1, y2)

        clamped_rect = cv2.minAreaRect(box_pts.astype(np.float32))

        return clamped_rect

    def rect_outside_bbox_ratio(self, rect, bbox):
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)

        box_pts = cv2.boxPoints(rect)

        outside = 0.0

        for px, py in box_pts:
            if px < x1:
                outside += (x1 - px) / bw
            if px > x2:
                outside += (px - x2) / bw
            if py < y1:
                outside += (y1 - py) / bh
            if py > y2:
                outside += (py - y2) / bh

        return float(outside)

    # ============================================================
    # Validation / scoring
    # ============================================================
    def score_rect_with_yolo_bbox(self, rect, yolo_bbox, cls_name):
        (rcx, rcy), (rw, rh), _ = rect
        x1, y1, x2, y2 = yolo_bbox

        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        bbox_w = max(x2 - x1, 1.0)
        bbox_h = max(y2 - y1, 1.0)
        bbox_area = max(bbox_w * bbox_h, 1.0)

        rect_area = max(float(rw * rh), 1.0)

        center_dist = math.hypot(rcx - bbox_cx, rcy - bbox_cy)
        center_score = center_dist / max(math.hypot(bbox_w, bbox_h), 1.0)

        area_ratio = rect_area / bbox_area
        area_score = abs(math.log(area_ratio + 1e-6))

        rect_ratio = max(rw, rh) / max(min(rw, rh), 1.0)

        if self.is_square_class(cls_name):
            expected_ratio = 1.0
        else:
            expected_ratio = 2.0

        ratio_score = abs(math.log((rect_ratio + 1e-6) / expected_ratio))

        box_pts = cv2.boxPoints(rect)

        outside_penalty = 0.0
        for px, py in box_pts:
            if px < x1:
                outside_penalty += (x1 - px) / bbox_w
            if px > x2:
                outside_penalty += (px - x2) / bbox_w
            if py < y1:
                outside_penalty += (y1 - py) / bbox_h
            if py > y2:
                outside_penalty += (py - y2) / bbox_h

        score = (
            center_score * 2.0 +
            area_score * 0.8 +
            ratio_score * 1.2 +
            outside_penalty * 2.0
        )

        return float(score)

    def compute_rect_depth_overlap_ratio(self, rect, depth_mask):
        if depth_mask is None or depth_mask.size == 0:
            return 0.0

        h, w = depth_mask.shape[:2]

        rect_mask = np.zeros((h, w), dtype=np.uint8)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.intp(box_pts)
        cv2.drawContours(rect_mask, [box_pts], 0, 255, thickness=-1)

        rect_area = np.count_nonzero(rect_mask > 0)
        if rect_area == 0:
            return 0.0

        overlap = np.count_nonzero(np.logical_and(rect_mask > 0, depth_mask > 0))

        return float(overlap / rect_area)

    # ============================================================
    # Yaw utils
    # ============================================================
    def compute_pca_yaw_with_confidence(self, contour):
        if contour is None or len(contour) < 5:
            return 0.0, 1.0, False

        pts = contour.reshape(-1, 2).astype(np.float32)

        if pts.shape[0] < 5:
            return 0.0, 1.0, False

        mean = np.mean(pts, axis=0)
        centered = pts - mean

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        lambda1 = float(eigvals[0])
        lambda2 = float(eigvals[1])

        if lambda2 < 1e-6:
            ratio = 999.0
        else:
            ratio = lambda1 / lambda2

        v = eigvecs[:, 0]

        yaw = math.degrees(math.atan2(float(v[1]), float(v[0])))
        yaw = self.normalize_yaw_axis_deg(yaw)

        reliable = ratio >= self.pca_ratio_threshold_rect

        return float(yaw), float(ratio), bool(reliable)

    def normalize_rect_angle(self, rect):
        (_, _), (w, h), angle = rect

        if w < h:
            angle += 90.0

        return self.normalize_yaw_axis_deg(angle)

    def normalize_yaw_axis_deg(self, yaw):
        yaw = float(yaw)

        while yaw >= 180.0:
            yaw -= 360.0
        while yaw < -180.0:
            yaw += 360.0

        if yaw >= 90.0:
            yaw -= 180.0
        if yaw < -90.0:
            yaw += 180.0

        if abs(yaw) < 1e-6:
            yaw = 0.0

        return float(yaw)

    def square_yaw_from_yolo_bottom_left(self, rect, bbox):
        x1, y1, x2, y2 = bbox
        ref_pt = np.array([float(x1), float(y2)], dtype=np.float32)

        pts = cv2.boxPoints(rect).astype(np.float32)

        best_edge = None
        best_dist = float("inf")

        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]

            dist = self.point_to_segment_distance(ref_pt, p1, p2)

            if dist < best_dist:
                best_dist = dist
                best_edge = (p1, p2)

        if best_edge is None:
            return 0.0

        p1, p2 = best_edge

        # 화면 좌표계 기준 시계 방향 해석을 안정화하기 위해
        # 항상 x가 작은 점에서 큰 점으로 향하게 정렬
        if p2[0] < p1[0]:
            p1, p2 = p2, p1

        v = p2 - p1

        if np.linalg.norm(v) < 1e-6:
            return 0.0

        angle = math.degrees(math.atan2(float(v[1]), float(v[0])))

        angle = angle % 180.0

        if angle > 90.0:
            angle = 180.0 - angle

        if abs(angle) < 1e-6:
            angle = 0.0

        return float(angle)

    def point_to_segment_distance(self, p, a, b):
        ab = b - a
        ap = p - a

        denom = float(np.dot(ab, ab))

        if denom < 1e-6:
            return float(np.linalg.norm(p - a))

        t = float(np.dot(ap, ab) / denom)
        t = max(0.0, min(1.0, t))

        closest = a + t * ab

        return float(np.linalg.norm(p - closest))

    def angle_diff_axis_deg(self, a, b):
        diff = abs(float(a) - float(b))

        while diff >= 180.0:
            diff -= 180.0

        if diff > 90.0:
            diff = 180.0 - diff

        return abs(float(diff))

    def is_square_class(self, cls_name):
        return str(cls_name).startswith('2x2')

    # ============================================================
    # General utils
    # ============================================================
    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def expand_bbox(self, x1, y1, x2, y2, padding, width, height):
        x1p = self.clamp(x1 - padding, 0, width - 1)
        y1p = self.clamp(y1 - padding, 0, height - 1)
        x2p = self.clamp(x2 + padding, 0, width - 1)
        y2p = self.clamp(y2 + padding, 0, height - 1)

        return x1p, y1p, x2p, y2p

    def is_pixel_inside(self, x, y, width, height):
        return 0 <= x < width and 0 <= y < height

    def clamp(self, v, lo, hi):
        return int(max(lo, min(int(v), hi)))

    def make_depth_colormap(self, depth_m):
        valid = np.logical_and(
            depth_m > self.valid_depth_min,
            depth_m < self.valid_depth_max
        )

        depth_vis = np.zeros(depth_m.shape, dtype=np.uint8)

        if np.count_nonzero(valid) > 0:
            d_min = np.percentile(depth_m[valid], 2)
            d_max = np.percentile(depth_m[valid], 98)

            if d_max > d_min:
                depth_vis = np.clip(
                    (depth_m - d_min) / (d_max - d_min) * 255.0,
                    0,
                    255
                ).astype(np.uint8)

        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    def draw_debug_text(
        self,
        debug_img,
        x1,
        y1,
        y2,
        cls_name,
        conf,
        X,
        Y,
        Z,
        yaw,
        height_mm,
        area,
        depth_debug,
        rect_source,
        pca_ratio,
        yaw_mode,
        outside_ratio
    ):
        scale = self.cv_text_scale
        thickness = self.cv_text_thickness
        gap = self.cv_text_line_gap

        text_x = x1
        text_y = max(25, y1 - int(gap * 5))

        method = depth_debug.get('method', 'unknown')
        floor_d = depth_debug.get('floor_depth', 0.0)
        body_d = depth_debug.get('body_depth', 0.0)
        inliers = depth_debug.get('inlier_count', 0)

        cv2.putText(
            debug_img,
            f"{cls_name} conf:{conf:.2f}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thickness
        )

        cv2.putText(
            debug_img,
            f"XYZmm:{X*1000.0:.1f},{Y*1000.0:.1f},{Z*1000.0:.1f}",
            (text_x, text_y + gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thickness
        )

        cv2.putText(
            debug_img,
            f"yaw:{yaw:.2f}deg {yaw_mode}",
            (text_x, text_y + gap * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thickness
        )

        cv2.putText(
            debug_img,
            f"H:{height_mm:.1f}mm src:{rect_source}",
            (text_x, text_y + gap * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 0),
            max(1, thickness - 1)
        )

        cv2.putText(
            debug_img,
            f"pcaR:{pca_ratio:.2f} out:{outside_ratio:.3f}",
            (text_x, text_y + gap * 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 0),
            max(1, thickness - 1)
        )

        cv2.putText(
            debug_img,
            f"{method} in:{inliers} area:{area:.0f}",
            (text_x, min(self.intrinsics['height'] - 10, y2 + gap)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.8,
            (255, 255, 0),
            max(1, thickness - 1)
        )

        cv2.putText(
            debug_img,
            f"floor:{floor_d:.3f} body:{body_d:.3f}",
            (text_x, min(self.intrinsics['height'] - 10, y2 + gap * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.8,
            (255, 255, 0),
            max(1, thickness - 1)
        )


def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
