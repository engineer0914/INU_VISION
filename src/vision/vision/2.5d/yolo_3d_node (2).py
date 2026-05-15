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

        # --- 파라미터 선언 ---
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('model_path', 'yolo_models/train1_05_04/weights/best.pt')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('publish_debug_image', True)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        
        # 모델 절대 경로 생성 및 로드
        pkg_share_dir = get_package_share_directory('vision')
        abs_model_path = os.path.join(pkg_share_dir, self.get_parameter('model_path').value)
        self.get_logger().info(f"YOLO 모델 로드 중: {abs_model_path}")
        
        self.model = YOLO(abs_model_path)
        self.model.to(self.get_parameter('device').value)
        self.conf_thres = float(self.get_parameter('conf_thres').value)

        # --- 매핑 딕셔너리 ---
        self.cmd_to_cls = {
            '2b': '2x2_blue', '2r': '2x2_red', '2g': '2x2_green', '2y': '2x2_yellow',
            '4b': '4x2_blue', '4r': '4x2_red', '4g': '4x2_green', '4y': '4x2_yellow'
        }

        # --- 상태 변수 ---
        self.cv_bridge = CvBridge()
        self.intrinsics = None
        self.latest_objects = [] # 실시간으로 추정된 객체 정보 저장 리스트

        # --- 통신 설정 ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        
        if self.get_parameter('publish_debug_image').value:
            self.debug_pub = self.create_publisher(Image, '/vision/debug_image', 10)

        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)
        self.get_logger().info("완성형 실시간 Yolo 3D 노드가 시작되었습니다!")

    # ============================================================
    # ROS Callbacks
    # ============================================================
    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}
            self.get_logger().info("카메라 정보 수신 완료.")

    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None:
            return

        try:
            color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_raw = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_raw.astype(np.float32) * 0.001
        except Exception as e:
            self.get_logger().warn(f"이미지 변환 에러: {e}")
            return

        # 1. 실시간 YOLO 추론
        results = self.model(color_img, conf=self.conf_thres, verbose=False)[0]
        debug_img = color_img.copy()
        current_frame_objects = []

        # 2. 검출된 모든 객체에 대해 실시간 포즈 추정 수행
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # -- ROI 추출 및 마스크 연산 --
            padding = 20
            x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
            x2p, y2p = min(self.intrinsics['width'], x2 + padding), min(self.intrinsics['height'], y2 + padding)

            roi_color = color_img[y1p:y2p, x1p:x2p]
            roi_depth_m = depth_m[y1p:y2p, x1p:x2p]

            color_mask = self.get_color_mask(roi_color)
            depth_mask = np.logical_and(roi_depth_m > 0.15, roi_depth_m < 1.2).astype(np.uint8) * 255
            object_mask = self.clean_mask(cv2.bitwise_and(color_mask, depth_mask))

            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_pose = False
            X, Y, Z, angle = 0.0, 0.0, 0.0, 0.0
            cx, cy = 0, 0

            for cnt in contours:
                if cv2.contourArea(cnt) < 500: continue
                
                rect = cv2.minAreaRect(cnt)
                angle = rect[2]
                cx, cy = int(rect[0][0]) + x1p, int(rect[0][1]) + y1p

                if cx < 0 or cy < 0 or cx >= self.intrinsics['width'] or cy >= self.intrinsics['height']:
                    continue

                z = depth_m[cy, cx]
                if z > 0:
                    X = (cx - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                    Y = (cy - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                    Z = z
                    valid_pose = True
                    break # 첫 번째 유효한 윤곽선만 처리

            # 결과를 리스트에 저장
            current_frame_objects.append({
                'cls_name': cls_name, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'valid': valid_pose, 'X': X, 'Y': Y, 'Z': Z, 'rz': angle, 'cx': cx, 'cy': cy
            })

            # -- 실시간 rqt 화면에 결과 그리기 --
            if valid_pose:
                cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(debug_img, f"{cls_name} XYZ:{X:.2f},{Y:.2f},{Z:.2f}", 
                            (x1, max(20, y1-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(debug_img, f"RZ:{angle:.1f} conf:{conf:.2f}", 
                            (x1, max(40, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(debug_img, f"{cls_name} (No Depth)", 
                            (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 최신 객체 정보 갱신 및 디버그 이미지 발행
        self.latest_objects = current_frame_objects
        if self.get_parameter('publish_debug_image').value:
            self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

    # ============================================================
    # Service Logic (실시간으로 구해진 값 중 타겟만 쏙 빼감)
    # ============================================================
    def handle_get_pose(self, request, response):
        response.success = False
        response.x = response.y = response.z = response.rz = 0.0
        response.view_result = ""

        if not self.latest_objects and self.intrinsics is None:
            self.get_logger().warn("객체가 없거나 카메라가 준비되지 않았습니다.")
            return response

        cmd = request.command
        self.get_logger().info(f"서비스 요청 수신: [{cmd}]")

        # 1. VW (View mode)
        if cmd == 'vw':
            counts = {k: 0 for k in self.cmd_to_cls.keys()}
            for obj in self.latest_objects:
                for k, v in self.cmd_to_cls.items():
                    if v == obj['cls_name']: counts[k] += 1
            response.view_result = "//".join([f"{k}_{v}" for k, v in counts.items() if v > 0])
            response.success = True
            return response

        # 2. 특정 블록 좌표 요청
        if cmd not in self.cmd_to_cls:
            return response

        target_cls = self.cmd_to_cls[cmd]
        best_obj = None
        min_dist = float('inf')
        img_cx, img_cy = self.intrinsics['width'] / 2.0, self.intrinsics['height'] / 2.0

        # 타겟 클래스 중 가장 화면 중앙에 가까운 '유효한(valid)' 객체 찾기
        for obj in self.latest_objects:
            if obj['cls_name'] == target_cls and obj['valid']:
                box_cx, box_cy = (obj['x1'] + obj['x2']) / 2.0, (obj['y1'] + obj['y2']) / 2.0
                dist = math.hypot(box_cx - img_cx, box_cy - img_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_obj = obj

        if best_obj:
            response.x, response.y, response.z = best_obj['X'], best_obj['Y'], best_obj['Z']
            response.rz = best_obj['rz']
            response.success = True
        else:
            self.get_logger().info(f"화면에 뎁스 추출이 가능한 {target_cls} 객체가 없습니다.")

        return response

    # ============================================================
    # Utils
    # ============================================================
    def get_color_mask(self, color_img):
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)), cv2.inRange(hsv, (170, 80, 50), (180, 255, 255)))
        blue = cv2.inRange(hsv, (90, 70, 40), (130, 255, 255))
        green = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
        yellow = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))
        return cv2.bitwise_or(cv2.bitwise_or(red, blue), cv2.bitwise_or(green, yellow))

    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
