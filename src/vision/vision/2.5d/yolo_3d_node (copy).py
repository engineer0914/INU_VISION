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

        # depth 기반 처리 파라미터
        self.declare_parameter('roi_padding', 40)
        self.declare_parameter('bbox_inner_margin', 0)
        self.declare_parameter('valid_depth_min', 0.15)
        self.declare_parameter('valid_depth_max', 1.20)
        self.declare_parameter('floor_percentile', 85.0)
        self.declare_parameter('object_height_threshold', 0.010)
        self.declare_parameter('min_component_area', 200)
        self.declare_parameter('min_contour_area', 300)

        # debug
        self.declare_parameter('show_depth_side_by_side', True)

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
        self.show_depth_side_by_side = bool(self.get_parameter('show_depth_side_by_side').value)

        # ============================================================
        # YOLO Load
        # ============================================================
        pkg_share_dir = get_package_share_directory('vision')
        abs_model_path = os.path.join(pkg_share_dir, self.get_parameter('model_path').value)

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

        self.get_logger().info("YOLO + aligned depth 기반 3D pose node 시작.")
        self.get_logger().info(f"color_topic: {self.color_topic}")
        self.get_logger().info(f"depth_topic: {self.depth_topic}")
        self.get_logger().info(f"camera_info_topic: {self.info_topic}")

    # ============================================================
    # ROS callbacks
    # ============================================================
    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'ppx': msg.k[2],
                'ppy': msg.k[5],
                'width': msg.width,
                'height': msg.height
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
                f"aligned_depth_to_color 토픽을 확인하세요."
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

            # 이미지 범위 보정
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # 초록색: YOLO bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # padding ROI: floor depth 추정용
            x1p, y1p, x2p, y2p = self.expand_bbox(
                x1, y1, x2, y2,
                self.roi_padding,
                img_w,
                img_h
            )

            # 파란색: padding ROI
            cv2.rectangle(debug_img, (x1p, y1p), (x2p, y2p), (255, 0, 0), 1)

            roi_depth_m = depth_m[y1p:y2p, x1p:x2p]

            # 원래 YOLO bbox가 padding ROI 내부에서 차지하는 위치
            core_bbox_in_roi = (
                x1 - x1p,
                y1 - y1p,
                x2 - x1p,
                y2 - y1p
            )

            object_mask, depth_debug = self.get_depth_object_mask_limited_to_bbox(
                roi_depth_m,
                core_bbox_in_roi
            )

            valid_pose = False
            X = Y = Z = 0.0
            yaw = 0.0
            cx = cy = 0
            area = 0.0
            height_mm = 0.0

            contours, _ = cv2.findContours(
                object_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(cnt))

                if area >= self.min_contour_area:
                    rect = cv2.minAreaRect(cnt)
                    yaw = self.normalize_rect_angle(rect)

                    cx_roi = int(rect[0][0])
                    cy_roi = int(rect[0][1])
                    cx = cx_roi + x1p
                    cy = cy_roi + y1p

                    # object mask 내부 depth의 median을 Z로 사용
                    z = self.get_object_median_depth(
                        roi_depth_m,
                        object_mask
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

                        # 노란색: depth 기반 minAreaRect
                        box_pts = cv2.boxPoints(rect)
                        box_pts = np.intp(box_pts)
                        box_pts[:, 0] += x1p
                        box_pts[:, 1] += y1p

                        cv2.drawContours(debug_img, [box_pts], 0, (0, 255, 255), 2)
                        cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)

                        cv2.drawContours(depth_vis, [box_pts], 0, (0, 255, 255), 2)
                        cv2.circle(depth_vis, (cx, cy), 5, (255, 0, 0), -1)

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
                'X': X,
                'Y': Y,
                'Z': Z,
                'rz': yaw,
                'height_mm': height_mm,
                'area': area
            })

            if valid_pose:
                self.draw_debug_text(
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
                    depth_debug
                )
            else:
                cv2.putText(
                    debug_img,
                    f"{cls_name} depth failed",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
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

        # view mode
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

        # 현재는 target class 중 화면 중심에 가장 가까운 객체 선택
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

        response.x = float(best_obj['X'])
        response.y = float(best_obj['Y'])
        response.z = float(best_obj['Z'])
        response.rz = float(best_obj['rz'])
        response.success = True

        self.get_logger().info(
            f"pose response: {target_cls}, "
            f"x={response.x:.3f}, y={response.y:.3f}, "
            f"z={response.z:.3f}, yaw={response.rz:.1f}"
        )

        return response

    # ============================================================
    # Depth processing
    # ============================================================
    def get_depth_object_mask_limited_to_bbox(self, roi_depth_m, core_bbox):
        """
        padding ROI 전체를 이용해 floor_depth를 추정하되,
        실제 객체 후보는 원래 YOLO bbox 내부로 제한한다.
        """

        debug = {
            'floor_depth': 0.0,
            'body_depth': 0.0
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

        # 원래 YOLO bbox 내부 mask
        bx1, by1, bx2, by2 = core_bbox

        m = self.bbox_inner_margin
        bx1 = max(0, bx1 + m)
        by1 = max(0, by1 + m)
        bx2 = min(w, bx2 - m)
        by2 = min(h, by2 - m)

        core_mask = np.zeros((h, w), dtype=np.uint8)

        if bx2 <= bx1 or by2 <= by1:
            return np.zeros((h, w), dtype=np.uint8), debug

        core_mask[by1:by2, bx1:bx2] = 255

        # 바닥보다 가까운 영역만 object 후보
        object_candidate = np.logical_and(
            valid,
            roi_depth_m < floor_depth - self.object_height_threshold
        ).astype(np.uint8) * 255

        # 중요: object 후보를 YOLO bbox 내부로 제한
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
            debug['body_depth'] = float(np.median(obj_depth))

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

    def keep_largest_component(self, mask, min_area=200):
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
        largest_area = stats[largest_label, cv2.CC_STAT_AREA]

        if largest_area < min_area:
            return np.zeros_like(mask, dtype=np.uint8)

        out = np.zeros_like(mask, dtype=np.uint8)
        out[labels == largest_label] = 255

        return out

    # ============================================================
    # Utils
    # ============================================================
    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def normalize_rect_angle(self, rect):
        (_, _), (w, h), angle = rect

        # 긴 변 기준 yaw
        if w < h:
            angle += 90.0

        return float(angle)

    def expand_bbox(self, x1, y1, x2, y2, padding, width, height):
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(width, x2 + padding)
        y2p = min(height, y2 + padding)
        return x1p, y1p, x2p, y2p

    def is_pixel_inside(self, x, y, width, height):
        return 0 <= x < width and 0 <= y < height

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
        depth_debug
    ):
        text_x = x1
        text_y = max(20, y1 - 65)

        floor_d = depth_debug.get('floor_depth', 0.0)
        body_d = depth_debug.get('body_depth', 0.0)

        cv2.putText(
            debug_img,
            f"{cls_name} conf:{conf:.2f}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            debug_img,
            f"XYZ:{X:.3f},{Y:.3f},{Z:.3f}",
            (text_x, text_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            debug_img,
            f"yaw:{yaw:.1f} H:{height_mm:.1f}mm",
            (text_x, text_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            debug_img,
            f"floor:{floor_d:.3f} body:{body_d:.3f} area:{area:.0f}",
            (text_x, min(self.intrinsics['height'] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1
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
