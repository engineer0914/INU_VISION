# ROS2 INU VISION Workspace

* `vision`: YOLO 기반 객체 인식 및 비전 처리 패키지
*    👉 **[Vision 패키지 설명서 및 테스트 영상 보러가기](src/vision/README.MD)**
* `msgs_pkg`: 사용자 정의 메시지 패키지
* `launch_pkg`: 통합 실행 패키지


### 주요문제: ARM코어 기반 기기내 realsense imu센서 커널 동작 불가
### 해결책: 커널 우회

* RSUSB 백엔드 빌드 및 커널 우회 설정 구조 설정
*    👉 https://github.com/engineer0914/realsense_RSUSB_Backend_Build

### 젯슨 파이토치 설치 링크
*    👉 https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
