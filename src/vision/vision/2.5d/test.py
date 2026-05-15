import os
import json
import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# ============================================================
# YOLO 설정
# ============================================================
# 상대 경로 추천
YOLO_WEIGHT_PATH = "train1_05_04/weights/best.pt"

model = YOLO(YOLO_WEIGHT_PATH)
model.to("cuda")  # Jetson Orin에서 CUDA 사용


# ============================================================
# RealSense 설정
# ============================================================
pipeline = rs.pipeline()
config = rs.config()

WIDTH = 640
HEIGHT = 480
FPS = 30

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print(f"[INFO] Depth scale: {depth_scale}")

if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    print("[INFO] IR emitter enabled")

if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, 300)
    print("[INFO] Laser power set to 300")


# ============================================================
# Depth Filter
# ============================================================
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

USE_HOLE_FILLING = False
hole_filling = rs.hole_filling_filter()


# ============================================================
# 저장 함수
# ============================================================
def make_depth_colormap(depth_raw):
    """
    depth_raw: uint16 depth image
    시각화용 colormap 반환
    """
    depth_8u = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color


def save_capture(color_img, result_img, depth_raw, roi_records):
    """
    현재 프레임의 원본 이미지, 결과 이미지, depth 이미지, ROI 이미지 저장
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    base_dir = os.path.join(os.getcwd(), "captures", timestamp)
    os.makedirs(base_dir, exist_ok=True)

    # 전체 이미지 저장
    color_path = os.path.join(base_dir, "color_original_640x480.png")
    result_path = os.path.join(base_dir, "yolo_result_640x480.png")
    depth_raw_path = os.path.join(base_dir, "depth_raw_16bit_640x480.png")
    depth_color_path = os.path.join(base_dir, "depth_colormap_640x480.png")

    cv2.imwrite(color_path, color_img)
    cv2.imwrite(result_path, result_img)

    # raw depth는 uint16 PNG로 저장됨
    cv2.imwrite(depth_raw_path, depth_raw)

    # 사람이 보기 쉬운 depth colormap
    depth_color = make_depth_colormap(depth_raw)
    cv2.imwrite(depth_color_path, depth_color)

    metadata = {
        "timestamp": timestamp,
        "width": int(color_img.shape[1]),
        "height": int(color_img.shape[0]),
        "depth_scale": float(depth_scale),
        "yolo_weight_path": YOLO_WEIGHT_PATH,
        "num_rois": len(roi_records),
        "rois": []
    }

    # ROI 저장
    for idx, roi in enumerate(roi_records):
        prefix = f"roi_{idx:02d}"

        roi_color_path = os.path.join(base_dir, f"{prefix}_color.png")
        roi_depth_raw_path = os.path.join(base_dir, f"{prefix}_depth_raw_16bit.png")
        roi_depth_color_path = os.path.join(base_dir, f"{prefix}_depth_colormap.png")
        roi_mask_path = os.path.join(base_dir, f"{prefix}_mask.png")
        roi_result_crop_path = os.path.join(base_dir, f"{prefix}_result_crop.png")

        cv2.imwrite(roi_color_path, roi["roi_color"])
        cv2.imwrite(roi_depth_raw_path, roi["roi_depth_raw"])
        cv2.imwrite(roi_depth_color_path, make_depth_colormap(roi["roi_depth_raw"]))

        if roi["object_mask"] is not None:
            cv2.imwrite(roi_mask_path, roi["object_mask"])

        cv2.imwrite(roi_result_crop_path, roi["roi_result_crop"])

        metadata["rois"].append({
            "index": idx,
            "class_name": roi["class_name"],
            "confidence": float(roi["confidence"]),
            "bbox_xyxy": [int(v) for v in roi["bbox_xyxy"]],
            "bbox_padded_xyxy": [int(v) for v in roi["bbox_padded_xyxy"]],
            "center_pixel": [int(v) for v in roi["center_pixel"]],
            "xyz_m": [float(v) for v in roi["xyz_m"]],
            "face_state": roi["face_state"],
            "height_m": float(roi["height_m"]),
            "obj_z_m": float(roi["obj_z_m"]),
            "floor_z_m": float(roi["floor_z_m"]),
            "angle_deg": float(roi["angle_deg"]),
            "files": {
                "roi_color": os.path.basename(roi_color_path),
                "roi_depth_raw_16bit": os.path.basename(roi_depth_raw_path),
                "roi_depth_colormap": os.path.basename(roi_depth_color_path),
                "roi_mask": os.path.basename(roi_mask_path),
                "roi_result_crop": os.path.basename(roi_result_crop_path),
            }
        })

    metadata_path = os.path.join(base_dir, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"[SAVE] Capture saved to: {base_dir}")


# ============================================================
# 기존 마스크 / 포즈 보조 함수
# ============================================================
def get_color_mask(color_img):
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


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_depth_edge(depth_m, object_mask):
    depth_valid = depth_m.copy()
    depth_valid[depth_valid <= 0] = 0

    depth_norm = cv2.normalize(
        depth_valid,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_norm = cv2.bitwise_and(
        depth_norm,
        depth_norm,
        mask=object_mask
    )

    depth_blur = cv2.GaussianBlur(depth_norm, (5, 5), 0)
    depth_edge = cv2.Canny(depth_blur, 30, 90)

    return depth_edge


def estimate_face_state(roi_depth_m, object_mask):
    """
    roi_depth_m: meter 단위 depth
    object_mask: uint8 mask

    바닥 대비 물체 높이로 윗면 / 뒷면 구분
    BACK : 낮은 높이
    TOP  : 높은 높이
    """

    obj_depth_values = roi_depth_m[object_mask > 0]
    obj_depth_values = obj_depth_values[obj_depth_values > 0]

    if len(obj_depth_values) < 30:
        return "UNKNOWN", 0.0, 0.0, 0.0

    floor_mask = np.ones_like(object_mask, dtype=np.uint8)
    floor_mask[object_mask > 0] = 0

    floor_depth_values = roi_depth_m[floor_mask > 0]
    floor_depth_values = floor_depth_values[floor_depth_values > 0]

    if len(floor_depth_values) < 30:
        return "UNKNOWN", 0.0, 0.0, 0.0

    obj_z = np.median(obj_depth_values)
    floor_z = np.median(floor_depth_values)

    # 카메라에서 가까운 물체일수록 depth가 작음
    height = floor_z - obj_z

    if height < 0.012:
        face_state = "BACK"
    elif height > 0.015:
        face_state = "TOP"
    else:
        face_state = "UNKNOWN"

    return face_state, height, obj_z, floor_z


# ============================================================
# 메인 루프
# ============================================================
try:
    print("[INFO] Start camera loop")
    print("[KEY] SPACE: save current frame")
    print("[KEY] q or ESC: quit")

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        if USE_HOLE_FILLING:
            depth_frame = hole_filling.process(depth_frame)

        color_img = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        depth_m = depth_raw.astype(np.float32) * depth_scale

        # YOLO inference
        yolo_result = model(
            color_img,
            conf=0.5,
            verbose=False
        )[0]

        result = color_img.copy()
        roi_records = []

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        for box_data in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])

            conf = float(box_data.conf[0])
            cls_id = int(box_data.cls[0])
            class_name = model.names[cls_id]

            # bbox 범위 보정
            x1 = max(0, min(x1, WIDTH - 1))
            y1 = max(0, min(y1, HEIGHT - 1))
            x2 = max(0, min(x2, WIDTH - 1))
            y2 = max(0, min(y2, HEIGHT - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            padding = 20

            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(WIDTH, x2 + padding)
            y2p = min(HEIGHT, y2 + padding)

            roi_color = color_img[y1p:y2p, x1p:x2p].copy()
            roi_depth_m = depth_m[y1p:y2p, x1p:x2p].copy()
            roi_depth_raw = depth_raw[y1p:y2p, x1p:x2p].copy()

            color_mask = get_color_mask(roi_color)

            depth_mask = np.logical_and(
                roi_depth_m > 0.15,
                roi_depth_m < 1.2
            )
            depth_mask = depth_mask.astype(np.uint8) * 255

            object_mask = cv2.bitwise_and(color_mask, depth_mask)
            object_mask = clean_mask(object_mask)

            face_state, height, obj_z, floor_z = estimate_face_state(
                roi_depth_m,
                object_mask
            )

            color_edge = cv2.Canny(object_mask, 50, 150)
            depth_edge = get_depth_edge(roi_depth_m, object_mask)

            combined_edge = cv2.bitwise_or(color_edge, depth_edge)
            combined_edge = cv2.bitwise_and(combined_edge, object_mask)

            contours, _ = cv2.findContours(
                object_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            saved_this_detection = False

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < 500:
                    continue

                rect = cv2.minAreaRect(cnt)

                box_pts = cv2.boxPoints(rect)
                box_pts = np.intp(box_pts)

                box_pts[:, 0] += x1p
                box_pts[:, 1] += y1p

                cx = int(rect[0][0]) + x1p
                cy = int(rect[0][1]) + y1p

                angle = rect[2]

                if cx < 0 or cy < 0 or cx >= WIDTH or cy >= HEIGHT:
                    continue

                z = depth_m[cy, cx]

                if z <= 0:
                    continue

                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrin,
                    [cx, cy],
                    z
                )

                X = point_3d[0]
                Y = point_3d[1]
                Z = point_3d[2]

                cv2.drawContours(
                    result,
                    [box_pts],
                    0,
                    (0, 255, 255),
                    2
                )

                for p in box_pts:
                    px, py = p
                    cv2.circle(
                        result,
                        (px, py),
                        5,
                        (0, 0, 255),
                        -1
                    )

                cv2.circle(
                    result,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )

                line_gap = 25

                info_x = x1
                info_y = max(30, y1 - 105)

                cv2.putText(
                    result,
                    f"{class_name} {conf:.2f}",
                    (info_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    result,
                    f"pixel: ({cx}, {cy})",
                    (info_x, info_y + line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    result,
                    f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}",
                    (info_x, info_y + line_gap * 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    result,
                    f"face: {face_state}",
                    (info_x, info_y + line_gap * 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    result,
                    f"h:{height * 100:.1f}cm obj:{obj_z:.3f} floor:{floor_z:.3f}",
                    (info_x, info_y + line_gap * 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                angle_x = x1
                angle_y = min(result.shape[0] - 10, y2 + 35)

                cv2.putText(
                    result,
                    f"angle: {angle:.1f} deg",
                    (angle_x, angle_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # 같은 detection에서 가장 먼저 유효한 contour 하나만 저장 기록
                if not saved_this_detection:
                    roi_result_crop = result[y1p:y2p, x1p:x2p].copy()

                    roi_records.append({
                        "roi_color": roi_color,
                        "roi_depth_raw": roi_depth_raw,
                        "object_mask": object_mask.copy(),
                        "roi_result_crop": roi_result_crop,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_padded_xyxy": [x1p, y1p, x2p, y2p],
                        "center_pixel": [cx, cy],
                        "xyz_m": [X, Y, Z],
                        "face_state": face_state,
                        "height_m": height,
                        "obj_z_m": obj_z,
                        "floor_z_m": floor_z,
                        "angle_deg": angle
                    })

                    saved_this_detection = True

            # 실시간 ROI 화면
            if roi_color.size > 0:
                cv2.imshow("ROI Color", roi_color)

            if object_mask is not None and object_mask.size > 0:
                cv2.imshow("ROI Mask", object_mask)

            if combined_edge is not None and combined_edge.size > 0:
                cv2.imshow("ROI Edge", combined_edge)

        depth_colormap = make_depth_colormap(depth_raw)

        cv2.imshow("Color Original 640x480", color_img)
        cv2.imshow("Depth Colormap 640x480", depth_colormap)
        cv2.imshow("YOLO + Color + Depth Edge", result)

        key = cv2.waitKey(1) & 0xFF

        # ESC or q
        if key == 27 or key == ord("q"):
            break

        # SPACE
        elif key == 32:
            save_capture(
                color_img=color_img,
                result_img=result,
                depth_raw=depth_raw,
                roi_records=roi_records
            )

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped")