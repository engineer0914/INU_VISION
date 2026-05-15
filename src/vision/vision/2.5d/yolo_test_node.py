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

        # --- 변수 초기화 ---
        pkg_share_dir = get_package_share_directory('vision')
        abs_model_path = os.path.join(pkg_share_dir, self.get_parameter('model_path').value)
        
        self.get_logger().info(f"YOLO 모델 로드 중: {abs_model_path}")
        self.model = YOLO(abs_model_path)
        self.model.to(self.get_parameter('device').value)

        self.cmd_to_cls = {
            '2b': '2x2_blue', '2r': '2x2_red', '2g': '2x2_green', '2y': '2x2_yellow',
            '4b': '4x2_blue', '4r': '4x2_red', '4g': '4x2_green', '4y': '4x2_yellow'
        }

        self.cv_bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_yolo_result = None  # 최신 YOLO 추론 결과 저장용
        self.intrinsics = None

        # --- 통신 설정 ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.color_sub = message_filters.Subscriber(self, Image, self.get_parameter('color_topic').value, qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.get_parameter('depth_topic').value, qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.info_sub = self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self.info_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/vision/debug_image', 10)
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)

        self.get_logger().info("실시간 YOLO 모니터링 모드로 노드가 시작되었습니다.")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def image_callback(self, color_msg, depth_msg):
        try:
            color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            self.latest_color = color_img.copy()

            # 실시간 YOLO 추론 수행
            results = self.model(color_img, conf=self.get_parameter('conf_thres').value, verbose=False)[0]
            self.latest_yolo_result = results

            # 실시간으로 모든 검출 객체 그리기 (rqt 확인용)
            debug_img = color_img.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # 검출된 모든 객체에 박스 표시
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_img, f"{cls_name} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        except Exception as e:
            self.get_logger().error(f"이미지 콜백 에러: {e}")

    def handle_get_pose(self, request, response):
        response.success = False
        if self.latest_yolo_result is None or self.intrinsics is None:
            return response

        cmd = request.command
        self.get_logger().info(f"서비스 요청 수신: [{cmd}]")

        # 1. VW 모드 처리
        if cmd == 'vw':
            counts = {k: 0 for k in self.cmd_to_cls.keys()}
            for box in self.latest_yolo_result.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                for k, v in self.cmd_to_cls.items():
                    if v == cls_name: counts[k] += 1
            response.view_result = "//".join([f"{k}_{v}" for k, v in counts.items() if v > 0])
            response.success = True
            return response

        # 2. 특정 객체 좌표 처리
        if cmd not in self.cmd_to_cls: return response
        target_cls = self.cmd_to_cls[cmd]
        
        best_box = None
        min_dist = float('inf')
        img_cx, img_cy = self.intrinsics['width']/2, self.intrinsics['height']/2

        for box in self.latest_yolo_result.boxes:
            if self.model.names[int(box.cls[0])] == target_cls:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dist = math.hypot((x1+x2)/2 - img_cx, (y1+y2)/2 - img_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_box = (x1, y1, x2, y2)

        if best_box:
            x1, y1, x2, y2 = best_box
            # 깊이 데이터 처리 (기존 로직 유지)
            depth_m = self.latest_depth.astype(np.float32) * 0.001
            
            # ROI 설정 및 포즈 추정 (단순화된 예시, 필요시 기존 test.py 로직 추가)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            z = depth_m[cy, cx]
            
            if z > 0:
                response.x = (cx - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                response.y = (cy - self.intrinsics['ppy']) * z / self.intrinsics['fy']
                response.z = float(z)
                response.rz = 0.0 # 필요시 minAreaRect로 계산
                response.success = True
        else:
            self.get_logger().info(f"현재 화면에 {target_cls} 객체가 없습니다.")

        return response

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
