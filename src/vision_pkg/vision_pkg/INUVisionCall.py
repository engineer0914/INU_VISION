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

# import INUVisionLib as ivl
from vision_pkg import INUVisionLib as ivl


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
    def capture_camera(self, mode="mid_50", visualize=False):
        devices = ivl.get_realsense_ids()        
        if len(devices) == 0:
            raise RuntimeError("연결된 RealSense 카메라가 없습니다.")
        target_serial = list(devices.keys())[0]

        # 캡처한 데이터를 클래스 내부 보관함(self)에 저장
        print("[INFO] 카메라 데이터 캡처 중...")
        self.color_rgb, self.depth, self.intrinsics, self.scale = ivl.capture_realsense_data(
            serial_number=target_serial, 
            mode=mode, 
            visualize=visualize
        )
        return self.color_rgb, self.depth, self.intrinsics, self.scale

    # ==========================================
    # 함수 2. 서치 함수
    # ==========================================
    def run_search(self, visualize=True):
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

    # ==========================================
    # 함수 3. 서치 결과 기반 위치 반환 함수 (ID 변환 포함)
    # ==========================================
    def get_pose_by_id(self, target_id, local_id=0):
        if self.class_index is None:
            raise RuntimeError("탐색된 인덱스가 없습니다. 먼저 run_search() 또는 run_search_assembly()를 실행하세요.")

        # 1. 입력받은 ID를 문자열 클래스 이름으로 변환
        target_class_name = self.id_to_class.get(target_id)

        if target_class_name is None:
            print(f"[ERROR] 등록되지 않은 ID 번호입니다: {target_id}")
            return None

        print(f"\n[INFO] 타겟 ID [{target_id}] ➔ 클래스명 ['{target_class_name}'] 변환 완료")

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
            print(f"[INFO] ivl.get_nearest_6d_pose_by_class 사용 실패. 직접 class_index에서 검색합니다.")
            print(f"[INFO] reason: {e}")

        # ------------------------------------------------------------
        # 2-B. assembly 모드용 직접 검색 fallback
        # ------------------------------------------------------------
        if pose is None:
            if target_class_name in self.class_index:
                pose_list = self.class_index[target_class_name]

                if local_id < len(pose_list):
                    pose = pose_list[local_id]
                else:
                    print(
                        f"[WARNING] '{target_class_name}' 객체는 {len(pose_list)}개만 있습니다. "
                        f"요청 local_id={local_id}"
                    )
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
            print("----------------------")

            return pose

        else:
            print(f"[WARNING] 시야에서 '{target_class_name}' 객체를 찾을 수 없습니다.")
            return None



# ==========================================
# 4. 단독 실행용 테스트 코드
# ==========================================
if __name__ == "__main__":
    print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")
    
    # ---------------------------------------------------------
    # 테스트 방법 1: 클래스를 이용한 깔끔한 테스트
    # ---------------------------------------------------------
    vision = VisionManager()
    
    try:
        vision.capture_camera(visualize=False)
        vision.run_search(visualize=False)
        
        # 4x2_blue (ID 7) 찾기 테스트
        test_pose = vision.get_pose_by_id(target_id=7, local_id=0)
        
        if test_pose:
            print("클래스를 이용한 포즈 추출 성공!")
            
    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {e}")




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



