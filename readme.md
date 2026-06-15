# ROS2 INU VISION Workspace

0615 업데이트:
- AMR 테스트용 시야내 브릭 피킹 + 조립체(인지 모델 X) 중심축 추출 구조
- 객체 지향 구조 변경
    INUVisionCall = 호출용 함수 + 객체 지향 구조
    INUVisionLib  = 하위 함수 제작 및 전처리 구조 모음 + 욜로 모델 경로 설정

## **호출 구조**

**단일 브릭의 포지션 + yaw 값**
ID 1~8까지 호출 가능
```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '1'}"
```

비고:
현재 카메라내 보이는 단일 브릭의 위치와 요값을 반환함
객체의 6D 추가 개발 예정


**조립체의 중심 좌표 + 요값 반환 구조**
```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '999'}"
```

비고:
시야내 바닥 기준 돌출된 객체의 X,Y,Z,YAW 값 반환
 - 카메라내 단일 조립체의 중심 축을 기준으로 각도 출력
 - 조립체 모델 추가 제작 이후 업데이트


