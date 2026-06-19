# vision_node.py
import rclpy
from rclpy.node import Node
from arm_interfaces.srv import GetTargetPose
from vision_pkg import INUVisionCall as ivc

import rclpy
from rclpy.node import Node
from arm_interfaces.srv import GetTargetPose
from vision_pkg import INUVisionCall as ivc

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.srv = self.create_service(GetTargetPose, '/get_target_pose', self.get_pose_cb)
        self.get_logger().info('[VISION] 초기화 중... VisionManager 로드')
        
        self.vision = ivc.VisionManager()
        self.get_logger().info('[VISION] vision_node 시작 완료 (INUVisionLib 기반)')

    def get_pose_cb(self, request, response):
        # 1. target_color 필드를 통해 ID 문자열을 받음 (예: "7", "999")
        target_str = request.target_color.strip()
        self.get_logger().info(f'[VISION] 서비스 요청 수신 - target ID: {target_str}')

        try:
            # 입력값이 숫자인지 확인
            if not target_str.isdigit():
                self.get_logger().error(f'[VISION] 잘못된 입력입니다. 숫자 ID를 입력하세요: {target_str}')
                response.success = False
                return response
            
            target_id = int(target_str)

            # 2. 카메라 최신 프레임 캡처
            self.vision.capture_camera(visualize=False)

            # 3. ID 번호에 따라 탐색(Search) 함수 분기
            if 1 <= target_id <= 8:
                self.get_logger().info(f'[VISION] 일반 브릭(ID:{target_id}) 탐색 모드 실행')
                self.vision.run_search(visualize=True)
            elif target_id == 999:
                self.get_logger().info('[VISION] 조립체(ID:999) 탐색 모드 실행')
                self.vision.run_search_assembly(visualize=True)
            elif target_id == 888:
                self.get_logger().info('[VISION] 조립체(ID:888) 탐색 모드 실행')
                self.vision.run_search_assembly_fine(visualize=True)
            else:
                self.get_logger().info(f'[VISION] 기타 객체(ID:{target_id}) 탐색 모드 실행')
                self.vision.run_search(visualize=True)

            # 4. 탐색된 결과에서 특정 타겟의 Pose 추출
            pose = self.vision.get_pose_by_id(target_id=target_id, local_id=0)

            # 5. 결과 반환 (Service Response)
            if pose is not None:
                response.success = True
                # ROS 표준 단위(미터)에 맞게 mm -> m 변환
                response.x = float(pose["x_mm"] / 1000.0)
                response.y = float(pose["y_mm"] / 1000.0)
                response.z = float(pose["z_mm"] / 1000.0)
                response.yaw = float(pose["yaw_deg"])
                # srv에 추가된 class_name 반환
                response.class_name = str(pose.get("class_name", ""))
                
                self.get_logger().info(
                    f'[VISION] 타겟({target_id}) 발견! X:{response.x*1000:.1f} Y:{response.y*1000:.1f} Yaw:{response.yaw:.1f} Class:{response.class_name}'
                )
            else:
                self.get_logger().error(f'[VISION] 시야에서 타겟(ID:{target_id})을 찾을 수 없습니다.')
                response.success = False

        except Exception as e:
            self.get_logger().error(f'[VISION] 처리 중 심각한 오류 발생: {e}')
            response.success = False

        return response

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
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



# def main(args=None):
#     rclpy.init(args=args)
#     node = VisionNode()

#     try:
#         node.get_logger().info('=== 🚀 페이크(Fake) 테스트 모드 시작 ===')
        
#         # 1. 가짜 Request와 Response 객체 생성
#         fake_request = GetTargetPose.Request()
#         fake_response = GetTargetPose.Response()

#         # 2. 테스트하고 싶은 ID를 여기에 입력! (예: "888", "7", "999")
#         TEST_TARGET_ID = "888" 
#         fake_request.target_color = TEST_TARGET_ID

#         node.get_logger().info(f'가짜 요청 생성 완료: 타겟 ID = {fake_request.target_color}')

#         # 3. 서비스 콜백 함수에 직접 던져서 파이썬 함수처럼 실행
#         result_response = node.get_pose_cb(fake_request, fake_response)

#         # 4. 결과 확인
#         if result_response.success:
#             node.get_logger().info('✅ 테스트 성공! 추출된 데이터:')
#             node.get_logger().info(f'X: {result_response.x:.3f} m')
#             node.get_logger().info(f'Y: {result_response.y:.3f} m')
#             node.get_logger().info(f'Z: {result_response.z:.3f} m')
#             node.get_logger().info(f'Yaw: {result_response.yaw:.1f} deg')
#             node.get_logger().info(f'Class: {result_response.class_name}')
#         else:
#             node.get_logger().error('❌ 테스트 실패: 객체를 찾지 못했거나 에러 발생')

#         # 일반적인 ROS 2 서비스 대기 모드로 돌리고 싶다면 아래 주석을 푸세요.
#         # rclpy.spin(node)

#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()