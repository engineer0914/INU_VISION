# import os
# import yaml
# import pprint
# import glob
# import torch
# import cv2
# import open3d as o3d
# from sklearn.cluster import DBSCAN
# from ultralytics import YOLO
# import pyrealsense2 as rs
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import pandas as pd
# import numpy as np
# import copy
# from scipy.spatial.transform import Rotation as R

import INUVisionLib as ivl

class VisionManager:
    def __init__(self):
        self.color_rgb = None
        self.depth = None
        self.intrinsics = None
        self.scale = None
        
        self.pose_table = None
        self.class_index = None

        # ID -> 클래스 이름 매핑 딕셔너리
        self.id_to_class = {
            1: "2x2_red", 2: "2x2_green", 3: "2x2_blue", 4: "2x2_yellow",
            5: "4x2_red", 6: "4x2_green", 7: "4x2_blue", 8: "4x2_yellow",

            # assembly / depth blob mode
            999: "assembly",
            888: "assembly_fine",

            13: "Magnet",
            34: "Battery",
            81: "estop",
            241: "traffic light",
            442: "carrot",
            462: "small tree",
            711: "hammer",
            4482: "big carrot",
            8518: "burger",
            46262: "bigtree",
            48132: "icecream"
        }

    # ==========================================
    # 함수 1. 카메라 호출 함수
    # ==========================================
    def capture_camera(self, mode="mid_50", visualize=True):
        devices = ivl.get_realsense_ids()        
        if len(devices) == 0:
            raise RuntimeError("연결된 RealSense 카메라가 없습니다.")
        target_serial = list(devices.keys())[0]

        # 캡처한 데이터를 클래스 내부 보관함(self)에 저장
        print("[INFO] 카메라 데이터 캡처 중...")
        self.color_rgb, self.depth, self.intrinsics, self.scale = ivl.capture_realsense_data(
            serial_number=target_serial, 
            mode=mode, 
            warmup_frames=10,
            visualize=visualize
        )
        return self.color_rgb, self.depth, self.intrinsics, self.scale

    # ==========================================
    # 함수 2. 서치 함수
    # ==========================================
    def run_search(self, visualize=False):
        print("[INFO] 전체 객체 탐색(Search Wide) 실행 중...")
        if self.color_rgb is None:
            raise RuntimeError("카메라 데이터가 없습니다. 먼저 capture_camera()를 실행하세요.")

        # 보관함에 있던 카메라 데이터를 꺼내서 서치 함수에 넣음
        self.pose_table, self.class_index = ivl.search_wide(
            self.color_rgb, self.depth, self.intrinsics, self.scale, V_visualize=visualize
        )
        return self.pose_table, self.class_index

    # ==========================================
    # 함수 2-1. 조립체 / 덩어리 서치 함수
    # ==========================================
    def run_search_assembly(
        self,
        visualize=False,
        class_name="assembly",
        ransac_distance_threshold=0.006,
        object_min_plane_dist=0.010,
        min_area_px=80,
        morph_open_ksize=3,
        morph_close_ksize=5,
        min_contour_area=80
    ):
        print("[INFO] 조립체 객체 탐색(Search Assembly) 실행 중...")

        if self.color_rgb is None:
            raise RuntimeError("카메라 데이터가 없습니다. 먼저 capture_camera()를 실행하세요.")

        self.pose_table, self.class_index = ivl.search_assembly(
            color_rgb=self.color_rgb,
            depth=self.depth,
            intrinsics=self.intrinsics,
            scale=self.scale,
            V_visualize=visualize,
            class_name=class_name,
            ransac_distance_threshold=ransac_distance_threshold,
            object_min_plane_dist=object_min_plane_dist,
            min_area_px=min_area_px,
            morph_open_ksize=morph_open_ksize,
            morph_close_ksize=morph_close_ksize,
            min_contour_area=min_contour_area
        )

        return self.pose_table, self.class_index

    def run_search_assembly_fine(self, visualize=False):
        print("[INFO] 정밀 조립체 객체 탐색(Search Assembly Fine) 실행 중...")

        if self.color_rgb is None:
            raise RuntimeError("카메라 데이터가 없습니다. 먼저 capture_camera()를 실행하세요.")

        result_img, target_pose_info = ivl.search_assembly_fine(
            color_rgb=self.color_rgb,
            depth=self.depth,
            intrinsics=self.intrinsics,
            scale=self.scale,
            V_visualize=visualize
        )

        # 기존 pose_table, class_index 포맷에 맞게 래핑하여 저장
        if target_pose_info is not None:
            # 기존 get_pose_by_id와 호환되도록 필수 필드 추가
            target_pose_info["class_name"] = "assembly_fine"
            target_pose_info["local_id"] = 0
            target_pose_info["global_idx"] = 0

            self.pose_table = [target_pose_info]
            self.class_index = {"assembly_fine": [target_pose_info]}
        else:
            self.pose_table = []
            self.class_index = {}

        return self.pose_table, self.class_index

    # ==========================================
    # 함수 3. 서치 결과 기반 위치 반환 함수 (자동 탐색 라우팅 포함)
    # ==========================================
    def get_pose_by_id(self, target_id, local_id=0, visualize=False):
        # 1. 입력받은 ID를 먼저 문자열 클래스 이름으로 변환
        target_class_name = self.id_to_class.get(target_id)

        if target_class_name is None:
            print(f"[ERROR] 등록되지 않은 ID 번호입니다: {target_id}")
            return None

        print(f"\n[INFO] 타겟 ID [{target_id}] ➔ 클래스명 ['{target_class_name}'] 변환 완료")

        # ------------------------------------------------------------
        # [NEW] 타겟 ID에 따른 서치 알고리즘 자동 실행 (Smart Routing)
        # ------------------------------------------------------------
        if target_id == 888:
            self.run_search_assembly_fine(visualize=visualize)
        elif target_id == 999:
            self.run_search_assembly(visualize=visualize)
            
        # 888, 999가 아닌 일반 객체인데 미리 run_search()를 안 돌린 경우 에러 발생
        if self.class_index is None:
            raise RuntimeError("탐색된 인덱스가 없습니다. 일반 객체 탐색 시 먼저 run_search()를 실행하세요.")

        pose = None

        # ------------------------------------------------------------
        # 2-A. 기존 search_wide용 ivl 함수 먼저 시도
        # ------------------------------------------------------------
        try:
            pose = ivl.get_nearest_6d_pose_by_class(
                class_index=self.class_index,
                target_class_name=target_class_name,
                local_id=local_id
            )
        except Exception as e:
            # 888이나 999는 여기서 못 찾고 2-B로 넘어갑니다.
            pass

        # ------------------------------------------------------------
        # 2-B. assembly / assembly_fine 모드용 직접 검색 fallback
        # ------------------------------------------------------------
        if pose is None:
            if target_class_name in self.class_index:
                pose_list = self.class_index[target_class_name]

                if local_id < len(pose_list):
                    pose = pose_list[local_id]
                else:
                    print(f"[WARNING] '{target_class_name}' 객체는 {len(pose_list)}개만 있습니다. 요청 local_id={local_id}")
                    return None
            else:
                print(f"[WARNING] class_index 안에 '{target_class_name}' 클래스가 없습니다.")
                return None

        # ------------------------------------------------------------
        # 3. 결과 출력 및 반환
        # ------------------------------------------------------------
        if pose is not None:
            x = pose.get("x_mm", None)
            y = pose.get("y_mm", None)
            z = pose.get("z_mm", None)

            roll = pose.get("roll_deg", 0.0)
            pitch = pose.get("pitch_deg", 0.0)
            yaw = pose.get("yaw_deg", 0.0)

            print("--- 6D Pose Result ---")
            print(f"class: {pose.get('class_name', target_class_name)}")
            print(f"local_id: {pose.get('local_id', local_id)}")
            print(f"global_idx: {pose.get('global_idx', 'N/A')}")

            if x is not None and y is not None and z is not None:
                print(f"XYZ mm: {x:.1f}, {y:.1f}, {z:.1f}")
            else:
                print("XYZ mm: N/A")

            print(f"RPY deg: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")

            # --------------------------------------------------------
            # [NEW] ID 888 (assembly_fine) 전용 고급 정보 추가 출력
            # --------------------------------------------------------
            if target_id == 888:
                print("--- Additional Info (Fine Mode) ---")
                # print(f"Object Height : {pose.get('object_height_mm', 0.0):.1f} mm")
                # print(f"Top Surface Z : {pose.get('top_z_mm', 0.0):.1f} mm")
                print(f"Aspect Ratio  : {pose.get('aspect_ratio', 1.0):.2f}")

            print("----------------------")

            return pose

        else:
            print(f"[WARNING] 시야에서 '{target_class_name}' 객체를 찾을 수 없습니다.")
            return None



# # ==========================================
# # 4. 단독 실행용 테스트 코드
# # ==========================================
# if __name__ == "__main__":
#     print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")
    
#     # ---------------------------------------------------------
#     # 테스트 방법 1: 클래스를 이용한 깔끔한 테스트
#     # ---------------------------------------------------------
#     vision = VisionManager()
    
#     try:
#         vision.capture_camera(visualize=False)
#         vision.run_search(visualize=False)
        
#         # 4x2_blue (ID 7) 찾기 테스트
#         test_pose = vision.get_pose_by_id(target_id=7, local_id=0)
        
#         if test_pose:
#             print("클래스를 이용한 포즈 추출 성공!")
            
#     except Exception as e:
#         print(f"[ERROR] 테스트 중 오류 발생: {e}")




# # ==========================================
# # 5. 단독 실행용 테스트 코드_조립체
# # ==========================================
# if __name__ == "__main__":
#     print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")

#     vision = VisionManager()

#     vision.capture_camera(mode="mid_50", visualize=True)

#     pose_table, class_index = vision.run_search_assembly(
#         visualize=False,
#         object_min_plane_dist=0.010,
#         min_contour_area=80
#     )

#     pose = vision.get_pose_by_id(target_id=999, local_id=0)

# ==========================================
# 6. 단독 실행용 테스트 코드
# ==========================================
if __name__ == "__main__":
    print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")
    
    vision = VisionManager()
    
    try:
        # 카메라 한 번 켜고
        vision.capture_camera(visualize=False)
        
        # 888을 호출하면 내부에서 자동으로 run_search_assembly_fine(visualize=True) 동작!
        test_pose = vision.get_pose_by_id(target_id=888, local_id=0, visualize=True)
        
        if test_pose:
            print("✅ 888 포즈 추출 성공!")

            x = test_pose.get("x_mm", None)
            y = test_pose.get("y_mm", None)
            z = test_pose.get("z_mm", None)

            yaw = test_pose.get("yaw_deg", 0.0)
            print(f"XYZ mm: {x:.1f}, {y:.1f}, {z:.1f}")
            print(f"RPY deg: {yaw:.2f}")
            
    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {e}")