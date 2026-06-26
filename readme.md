# ROS2 INU VISION Workspace

0626 업데이트:
- 완성체를 호출시 최상단 색을 구분하여 요값 반환.
0615 업데이트:
- AMR 테스트용 시야내 브릭 피킹 + 조립체(인지 모델 X) 중심축 추출 구조
- 객체 지향 구조 변경
    visionnode = ROS 노드상 호출
    INUVisionCall = 호출용 함수 + 객체 지향 구조
    INUVisionLib  = 하위 함수 제작 및 전처리 구조 모음 + 욜로 모델 경로 설정

## **호출 구조**

**완성체의 포지션 + YAW 값 반환**
ID 34 13 ...등등 완성체가 눕혀진 상태에서의 호출 가능 color에 id로 호출

```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '13'}"
```

비고: -


**단일 브릭의 포지션 + yaw 값**
ID 1~8까지 호출 가능 color에 id로 호출
카메라내 12시 방향을 기준으로 단축의 방향을 반환
-90~90도
```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '1'}"
```

비고:


**기타 조립체의 중심 좌표 + 요값 반환 구조**
```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '999'}"
```

```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '888'}"
```

비고:
시야내 바닥 기준 돌출된 객체의 X,Y,Z,YAW 값 반환
 - 카메라내 단일 조립체의 중심 축을 기준으로 각도 출력
 - 조립체 모델 추가 제작 이후 업데이트


