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

        # --- 파라미터 선언 및 가져오기 ---
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
        rel_model_path = self.get_parameter('model_path').value
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.device = self.get_parameter('device').value
        self.publish_debug = self.get_parameter('publish_debug_image').value

        # 패키지 경로를 기반으로 절대 경로 생성
        pkg_share_dir = get_package_share_directory('vision')
        abs_model_path = os.path.join(pkg_share_dir, rel_model_path)

        self.get_logger().info(f"YOLO 모델 로드 중: {abs_model_path}")
        self.model = YOLO(abs_model_path)
        self.model.to(self.device)

        # --- 매핑 딕셔너리 (명령어 -> YOLO ID 이름) ---
        self.cmd_to_cls = {
            '2b': '2x2_blue', '2r': '2x2_red', '2g': '2x2_green', '2y': '2x2_yellow',
            '4b': '4x2_blue', '4r': '4x2_red', '4g': '4x2_green', '4y': '4x2_yellow'
        }

        # --- 상태 변수 ---
        self.cv_bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None  # uint16 원본
        self.intrinsics = None    # {fx, fy, ppx, ppy}

        # --- ROS 2 통신 설정 ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        
        # 디버그용 이미지 퍼블리셔
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/vision/debug_image', 10)

        # 핵심 서비스 서버 생성
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)
        
        self.get_logger().info("Yolo 3D 노드가 성공적으로 초기화되었습니다. 서비스 대기 중...")

    # ============================================================
    # ROS Callbacks
    # ============================================================
    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'ppx': msg.k[2], 'ppy': msg.k[5],
                'width': msg.width, 'height': msg.height
            }
            self.get_logger().info("카메라 캘리브레이션 정보 수신 완료.")

    def image_callback(self, color_msg, depth_msg):
        # 서비스 요청이 올 때 처리를 위해 최신 프레임만 버퍼에 유지합니다.
        try:
            self.latest_color = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough') # uint16
        except Exception as e:
            self.get_logger().warn(f"이미지 변환 에러: {e}")

    # ============================================================
    # Service Logic
    # ============================================================
    def handle_get_pose(self, request, response):
        # 초기화
        response.success = False
        response.x = response.y = response.z = response.rz = 0.0
        response.view_result = ""

        if self.latest_color is None or self.latest_depth is None or self.intrinsics is None:
            self.get_logger().warn("아직 카메라 데이터가 충분히 들어오지 않았습니다.")
            return response

        # 프레임 복사본 생성 (스레드 안전)
        color_img = self.latest_color.copy()
        depth_raw = self.latest_depth.copy()
        depth_m = depth_raw.astype(np.float32) * 0.001 # 리얼센스는 1 = 1mm

        cmd = request.command
        self.get_logger().info(f"서비스 요청 수신: [{cmd}]")

        # YOLO 추론 (서비스가 요청될 때만 1회 수행)
        yolo_result = self.model(color_img, conf=self.conf_thres, verbose=False)[0]
        
        # 1. VW (View mode) 요청 처리
        if cmd == 'vw':
            counts = {k: 0 for k in self.cmd_to_cls.keys()}
            for box_data in yolo_result.boxes:
                cls_name = self.model.names[int(box_data.cls[0])]
                # 일치하는 키워드 카운트 증가
                for k, v in self.cmd_to_cls.items():
                    if v == cls_name:
                        counts[k] += 1
            
            # 1 이상인 값만 포맷팅 (예: "2r_1//4g_2")
            result_list = [f"{k}_{v}" for k, v in counts.items() if v > 0]
            response.view_result = "//".join(result_list)
            response.success = True
            
            self.get_logger().info(f"VW 결과 반환: {response.view_result}")
            return response

        # 2. 특정 블록 (예: 2r, 4g) 요청 처리
        if cmd not in self.cmd_to_cls:
            self.get_logger().warn(f"알 수 없는 명령어: {cmd}")
            return response

        target_cls_name = self.cmd_to_cls[cmd]
        target_boxes = []

        # 타겟 클래스만 필터링
        for box_data in yolo_result.boxes:
            if self.model.names[int(box_data.cls[0])] == target_cls_name:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                target_boxes.append((x1, y1, x2, y2))

        if not target_boxes:
            self.get_logger().info(f"현재 화면에 {target_cls_name} 객체가 없습니다.")
            return response

        # 화면 중앙에서 가장 가까운 바운딩 박스 찾기 (피타고라스)
        img_cx, img_cy = self.intrinsics['width'] / 2.0, self.intrinsics['height'] / 2.0
        best_box = None
        min_dist = float('inf')

        for (x1, y1, x2, y2) in target_boxes:
            box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = math.hypot(box_cx - img_cx, box_cy - img_cy)
            if dist < min_dist:
                min_dist = dist
                best_box = (x1, y1, x2, y2)

        # --- 찾은 가장 가까운 객체에 대해 3D Pose 연산 수행 ---
        x1, y1, x2, y2 = best_box
        padding = 20
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(self.intrinsics['width'], x2 + padding)
        y2p = min(self.intrinsics['height'], y2 + padding)

        roi_color = color_img[y1p:y2p, x1p:x2p]
        roi_depth_m = depth_m[y1p:y2p, x1p:x2p]

        # 마스크 생성 및 외곽선 추출 (기존 test.py 로직)
        color_mask = self.get_color_mask(roi_color)
        depth_mask = np.logical_and(roi_depth_m > 0.15, roi_depth_m < 1.2).astype(np.uint8) * 255
        object_mask = cv2.bitwise_and(color_mask, depth_mask)
        object_mask = self.clean_mask(object_mask)

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_found = False
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            
            rect = cv2.minAreaRect(cnt)
            cx_roi, cy_roi = int(rect[0][0]), int(rect[0][1])
            angle = rect[2]  # deg

            # 원본 이미지 좌표로 변환
            cx = cx_roi + x1p
            cy = cy_roi + y1p
            
            if cx < 0 or cy < 0 or cx >= self.intrinsics['width'] or cy >= self.intrinsics['height']:
                continue

            z = depth_m[cy, cx]
            if z <= 0:
                continue

            # 카메라 파라미터로 수동 Deproject (pyrealsense2 의존성 제거)
            X = (cx - self.intrinsics['ppx']) * z / self.intrinsics['fx']
            Y = (cy - self.intrinsics['ppy']) * z / self.intrinsics['fy']
            Z = z

            # 성공 시 결과 매핑
            response.x = float(X)
            response.y = float(Y)
            response.z = float(Z)
            response.rz = float(angle)
            response.success = True
            
            valid_found = True
            
            # 디버그 이미지 발행 처리
            if self.publish_debug:
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_img, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(color_img, f"{cmd} XYZ:{X:.2f},{Y:.2f},{Z:.2f} RZ:{angle:.1f}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(color_img, 'bgr8'))
            
            break # 가장 큰(또는 첫 번째 유효한) 윤곽선만 처리하고 종료

        if not valid_found:
            self.get_logger().info("바운딩 박스는 찾았으나 Depth나 윤곽선 추출에 실패했습니다.")

        return response

    # ============================================================
    # 비전 유틸리티 함수 (기존 test.py)
    # ============================================================
    def get_color_mask(self, color_img):
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        blue = cv2.inRange(hsv, (90, 70, 40), (130, 255, 255))
        green = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
        yellow = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))

        mask = cv2.bitwise_or(red, blue)
        mask = cv2.bitwise_or(mask, green)
        mask = cv2.bitwise_or(mask, yellow)
        return mask

    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
