import argparse
import os
from pathlib import Path
import time

os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "16"

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from bytetrack_tracker import BatByteTracker
from detection_pipeline import TiledYoloDetector, motion_support_mask
from excel_report import BatDetectionReport
from motion_rules import LongStepMotionRule, MotionMatcher, ShortStepMotionRule
from relative_distance import detection_distance_label
from risk_zone import DangerZone, DangerZoneTracker, parse_danger_zone
from tracker import InstantFadeTracker
from video_controls import VideoControls, draw_playback_status

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best3.pt"
BLADE_MODEL_PATH = PROJECT_ROOT / "models" / "blade_best.pt"
VIDEO_PATH = PROJECT_ROOT / "videos" / "bat_vid2.mkv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "bat_detection_report.xlsx"
WINDOW_NAME = "Wind Turbine Bat Monitor"

PREDICT_CONF = 0.06
HIGH_CONFIDENCE = 0.25
PREDICT_IOU = 0.55
PREDICT_IMAGE_SIZE = 960
TILED_IMAGE_SIZE = 640
TILE_OVERLAP = 0.25
BLADE_CONFIDENCE = 0.25
BLADE_IMAGE_SIZE = 960
BLADE_IOU = 0.55
DANGER_ZONE_COLOR = (0, 0, 255)

ENABLE_TREE_MASK_FOR_YOLO = False
SHOW_RAW_DETECTIONS = False
SHOW_ALL_DETECTION_BOXES = True

TREE_MASK_POINTS = np.array(
    [[380, 0], [650, 0], [450, 480], [200, 480]],
    dtype=np.int32,
)
MIN_BOX_AREA = 8
MAX_BOX_AREA = 2400
MAX_BOX_WIDTH = 90
MAX_BOX_HEIGHT = 90
MAX_BOX_ASPECT_RATIO = 5.5
MIN_TRACK_START_SCORE = HIGH_CONFIDENCE
MIN_TRACK_UPDATE_SCORE = PREDICT_CONF
STATIONARY_CENTER_RADIUS = 8
STATIONARY_MAX_FRAMES = 5
STATIONARY_FORGET_FRAMES = 8

TRACK_LINE_MAX_JUMP = 130
TRACK_LINE_THICKNESS = 3
TRACK_SMOOTHING_WINDOW = 2
TRAIL_POINTS = 16


class StationaryDetectionFilter:
    def __init__(
        self,
        center_radius=STATIONARY_CENTER_RADIUS,
        max_stationary_frames=STATIONARY_MAX_FRAMES,
        forget_frames=STATIONARY_FORGET_FRAMES,
    ):
        self.center_radius = center_radius
        self.max_stationary_frames = max_stationary_frames
        self.forget_frames = forget_frames
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(detection):
        x1, y1, x2, y2 = detection[:4]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    def filter(self, detections):
        if len(detections) == 0:
            self._age_unmatched(set())
            return detections

        centers = [self._center(detection) for detection in detections]
        matched_track_ids = set()
        keep = []

        for detection, center in zip(detections, centers):
            track_id = self._match_track(center, matched_track_ids)
            if track_id is None:
                track_id = self._register(center)
                matched_track_ids.add(track_id)
            else:
                matched_track_ids.add(track_id)
                self._update_track(track_id, center)

            if self.tracks[track_id]["stationary_frames"] < self.max_stationary_frames:
                keep.append(detection)

        self._age_unmatched(matched_track_ids)
        if not keep:
            return np.empty((0, detections.shape[1]), dtype=np.float32)
        return np.asarray(keep, dtype=np.float32)

    def _match_track(self, center, used_track_ids):
        best_id = None
        best_distance = self.center_radius
        for track_id, track in self.tracks.items():
            if track_id in used_track_ids:
                continue
            distance = float(np.linalg.norm(center - track["center"]))
            if distance <= best_distance:
                best_id = track_id
                best_distance = distance
        return best_id

    def _register(self, center):
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = {
            "center": center,
            "anchor_center": center.copy(),
            "stationary_frames": 1,
            "missing_frames": 0,
        }
        return track_id

    def _update_track(self, track_id, center):
        track = self.tracks[track_id]
        track["center"] = center
        track["missing_frames"] = 0
        anchor_distance = float(np.linalg.norm(center - track["anchor_center"]))
        if anchor_distance <= self.center_radius:
            track["stationary_frames"] += 1
        else:
            track["anchor_center"] = center.copy()
            track["stationary_frames"] = 1

    def _age_unmatched(self, matched_track_ids):
        for track_id in list(self.tracks):
            if track_id in matched_track_ids:
                continue
            self.tracks[track_id]["missing_frames"] += 1
            if self.tracks[track_id]["missing_frames"] > self.forget_frames:
                del self.tracks[track_id]


def select_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def filter_detection_boxes(detections):
    filtered = []
    for detection in detections:
        box = detection[:4]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
            continue
        if width > MAX_BOX_WIDTH or height > MAX_BOX_HEIGHT:
            continue
        aspect_ratio = max(width / max(height, 1), height / max(width, 1))
        if aspect_ratio > MAX_BOX_ASPECT_RATIO:
            continue
        filtered.append(detection)

    if not filtered:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray(filtered, dtype=np.float32)


def smooth_points(points, window_size=TRACK_SMOOTHING_WINDOW):
    if len(points) < 2:
        return points

    smoothed = []
    for idx in range(len(points)):
        window = points[max(0, idx - window_size + 1) : idx + 1]
        smoothed.append(
            (
                int(sum(point[0] for point in window) / len(window)),
                int(sum(point[1] for point in window) / len(window)),
            )
        )
    smoothed[-1] = (int(points[-1][0]), int(points[-1][1]))
    return smoothed


def draw_detection_boxes(frame, boxes):
    for detection in boxes:
        x1, y1, x2, y2 = detection[:4].astype(int)
        confidence = float(detection[4])
        color = (220, 220, 220) if confidence >= HIGH_CONFIDENCE else (145, 145, 145)
        thickness = 2 if confidence >= HIGH_CONFIDENCE else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = detection_distance_label(detection)
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1,
        )
        label_y = y1 - 4
        text_y = y1 - 7
        if label_y - text_height < 0:
            label_y = y2 + text_height + 6
            text_y = y2 + text_height + 3
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 3),
            (x1 + text_width + 4, label_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_blade_boxes(frame, boxes):
    for detection in boxes:
        x1, y1, x2, y2 = detection[:4].astype(int)
        confidence = float(detection[4])
        color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Blade {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        label_y = y1 - 4
        text_y = y1 - 7
        if label_y - text_height < 0:
            label_y = y2 + text_height + 6
            text_y = y2 + text_height + 3
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 3),
            (x1 + text_width + 4, label_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def build_counter_lines(
    candidate_count,
    current_count,
    total_count,
    blade_count=None,
    danger_stats=None,
):
    lines = [
        (f"Candidates: {candidate_count}", (220, 220, 220)),
        (f"Active Bats: {current_count}", (0, 255, 255)),
        (f"Confirmed Flights: {total_count}", (0, 255, 0)),
    ]
    if danger_stats is not None:
        lines.append(
            (
                f"Impact Risk: {danger_stats.impact_risk_percent:.1f}%",
                DANGER_ZONE_COLOR,
            )
        )
    return lines


def draw_counter(
    frame,
    candidate_count,
    current_count,
    total_count,
    blade_count=None,
    danger_stats=None,
):
    lines = build_counter_lines(
        candidate_count,
        current_count,
        total_count,
        blade_count,
        danger_stats,
    )

    overlay = frame.copy()
    panel_height = 18 + (len(lines) * 38)
    cv2.rectangle(overlay, (10, 10), (340, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (25, 40 + (index * 38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def format_track_label(track_id, bbox):
    return f"ID:{track_id} {detection_distance_label(bbox)}"


def draw_tracked_objects(frame, tracker, objects, id_colors, current_bboxes):
    for tid, history in objects.items():
        if not tracker.is_confirmed(tid):
            continue
        if tid not in current_bboxes:
            continue

        color = id_colors[tid]
        points = list(history)[-TRAIL_POINTS:]

        x1, y1, x2, y2 = map(int, current_bboxes[tid])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        id_text = format_track_label(tid, current_bboxes[tid])
        (text_width, text_height), _ = cv2.getTextSize(
            id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            id_text,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        smoothed_points = smooth_points(points)
        for idx in range(1, len(smoothed_points)):
            p1, p2 = smoothed_points[idx - 1], smoothed_points[idx]
            if np.linalg.norm(np.subtract(p1, p2)) < TRACK_LINE_MAX_JUMP:
                cv2.line(
                    frame,
                    p1,
                    p2,
                    color,
                    TRACK_LINE_THICKNESS,
                    cv2.LINE_AA,
                )


def draw_danger_zone(frame, zone):
    cv2.ellipse(
        frame,
        (int(round(zone.center_x)), int(round(zone.center_y))),
        (int(round(zone.radius_x)), int(round(zone.radius_y))),
        float(zone.rotation_degrees),
        0,
        360,
        DANGER_ZONE_COLOR,
        2,
    )
    cv2.putText(
        frame,
        "Danger Zone",
        (int(round(zone.center_x - zone.radius_x)), int(round(zone.center_y - zone.radius_y - 8))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        DANGER_ZONE_COLOR,
        2,
        cv2.LINE_AA,
    )


def get_visible_track_ids(tracker, objects, current_bboxes):
    return {
        tid for tid in objects
        if tracker.is_confirmed(tid)
        and tid in current_bboxes
    }


def load_model(model_path):
    device = select_device()
    model = YOLO(model_path)
    try:
        model.to(device)
    except Exception as exc:
        print(f"{device} 啟動失敗，改用 CPU: {exc}")
        device = "cpu"
        model.to(device)
    print(f"啟動運算裝置: {device}")
    return model


def create_tracker(tracker_name, frame_rate):
    if tracker_name == "bytetrack":
        return BatByteTracker(
            frame_rate=frame_rate,
            high_threshold=HIGH_CONFIDENCE,
            low_threshold=PREDICT_CONF,
            new_track_threshold=HIGH_CONFIDENCE,
            track_buffer=18,
            match_threshold=0.92,
            confirmation_hits=3,
            confirmation_displacement=9.0,
            tracking_box_scale=2.4,
            stationary_radius=STATIONARY_CENTER_RADIUS,
            stationary_frames=STATIONARY_MAX_FRAMES,
        )

    low_track_score = PREDICT_CONF
    if tracker_name == "motion":
        low_track_score = MIN_TRACK_START_SCORE

    return InstantFadeTracker(
        max_disappeared=3,
        min_center_distance=6,
        history_size=24,
        min_hits=3,
        min_confirmation_displacement=10,
        min_new_track_score=MIN_TRACK_START_SCORE,
        low_track_score=low_track_score,
        max_reconnect_disappeared=2,
        max_prediction_points=0,
        stationary_radius=STATIONARY_CENTER_RADIUS,
        stationary_max_hits=STATIONARY_MAX_FRAMES,
        motion_matcher=MotionMatcher(
            short_rule=ShortStepMotionRule(max_step=34, max_fast_track_speed=24),
            long_rule=LongStepMotionRule(min_step=18, max_step=120),
            max_cost=1.20,
        ),
    )


def create_yolo_detector(
    model,
    detector_name,
    confidence,
    full_image_size,
    tiled_image_size,
    iou_threshold,
):
    return TiledYoloDetector(
        model=model,
        confidence=confidence,
        image_size=tiled_image_size if detector_name == "tiled" else full_image_size,
        grid_size=2 if detector_name == "tiled" else 1,
        overlap=TILE_OVERLAP,
        nms_iou=iou_threshold,
    )


def create_detector(model, detector_name):
    return create_yolo_detector(
        model=model,
        detector_name=detector_name,
        confidence=PREDICT_CONF,
        full_image_size=PREDICT_IMAGE_SIZE,
        tiled_image_size=TILED_IMAGE_SIZE,
        iou_threshold=PREDICT_IOU,
    )


def create_blade_detector(model):
    return create_yolo_detector(
        model=model,
        detector_name="full",
        confidence=BLADE_CONFIDENCE,
        full_image_size=BLADE_IMAGE_SIZE,
        tiled_image_size=BLADE_IMAGE_SIZE,
        iou_threshold=BLADE_IOU,
    )


def create_optional_blade_detector(
    model_path=BLADE_MODEL_PATH,
    load_model_fn=load_model,
    warn_on_missing=True,
):
    if not model_path:
        return None

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.is_file():
        if warn_on_missing:
            print(f"找不到風扇模型，略過風扇偵測: {resolved_path}")
        return None

    blade_model = load_model_fn(resolved_path)
    return create_blade_detector(blade_model)


def process_frame(
    frame,
    detector,
    stationary_filter,
    tracker,
    previous_gray,
    blade_detector=None,
    danger_tracker=None,
):
    inference_frame = frame.copy()
    if ENABLE_TREE_MASK_FOR_YOLO:
        cv2.fillPoly(inference_frame, [TREE_MASK_POINTS], (0, 0, 0))

    detections = filter_detection_boxes(detector.predict(inference_frame))
    motion_supported = motion_support_mask(previous_gray, inference_frame, detections)
    if len(detections):
        keep = (detections[:, 4] >= HIGH_CONFIDENCE) | motion_supported
        detections = detections[keep]
    detections = stationary_filter.filter(detections)

    display_frame = frame.copy()
    if danger_tracker is not None:
        draw_danger_zone(display_frame, danger_tracker.zone)

    blade_detections = np.empty((0, 5), dtype=np.float32)
    if blade_detector is not None:
        blade_detections = blade_detector.predict(inference_frame)
        draw_blade_boxes(display_frame, blade_detections)

    if SHOW_ALL_DETECTION_BOXES:
        draw_detection_boxes(display_frame, detections)
    if SHOW_RAW_DETECTIONS:
        draw_detection_boxes(display_frame, detections)

    objects, id_colors, _, current_bboxes = tracker.update(detections, display_frame)
    visible_track_ids = get_visible_track_ids(tracker, objects, current_bboxes)
    danger_stats = None
    if danger_tracker is not None:
        danger_tracker.update(visible_track_ids, current_bboxes)
        danger_stats = danger_tracker.stats(tracker.confirmed_total)

    draw_counter(
        display_frame,
        len(detections),
        len(visible_track_ids),
        tracker.confirmed_total,
        len(blade_detections) if blade_detector is not None else None,
        danger_stats,
    )
    draw_tracked_objects(display_frame, tracker, objects, id_colors, current_bboxes)

    current_gray = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY)
    return display_frame, detections, visible_track_ids, current_bboxes, current_gray


def run_video(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    display=True,
    max_frames=None,
    start_frame=0,
    preview_output=None,
    video_output=None,
    excel_output=DEFAULT_REPORT_PATH,
    detector_name="full",
    tracker_name="hybrid",
    blade_model_path=BLADE_MODEL_PATH,
    enable_blade_detection=True,
    danger_zone=None,
):
    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到影片: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型: {model_path}")

    model = load_model(model_path)
    detector = create_detector(model, detector_name)
    blade_detector = None
    if enable_blade_detection:
        blade_detector = create_optional_blade_detector(blade_model_path)
    stationary_filter = StationaryDetectionFilter()
    report = BatDetectionReport()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if danger_zone is None:
        danger_zone = DangerZone.from_frame(frame_width, frame_height)
    danger_tracker = DangerZoneTracker(danger_zone)

    start_frame = max(0, min(int(start_frame), total_frames - 1))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tracker = create_tracker(tracker_name, video_fps)
    target_frame_time_ms = 1000.0 / video_fps
    controls = None
    if display:
        controls = VideoControls(
            WINDOW_NAME,
            total_frames,
            video_fps,
            start_frame,
        )

    processed_frames = 0
    best_frame = None
    best_score = -1
    previous_gray = None
    video_writer = None
    started_at = time.perf_counter()

    try:
        while cap.isOpened():
            if max_frames is not None and processed_frames >= max_frames:
                break

            if controls is not None:
                requested_frame = controls.consume_seek()
                if requested_frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, requested_frame)
                    stationary_filter = StationaryDetectionFilter()
                    tracker = create_tracker(tracker_name, video_fps)
                    report = BatDetectionReport()
                    danger_tracker = DangerZoneTracker(danger_zone)
                    previous_gray = None
                    best_frame = None
                    best_score = -1

            loop_start_time = time.perf_counter()
            success, frame = cap.read()
            if not success:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            (
                display_frame,
                detections,
                visible_track_ids,
                current_bboxes,
                previous_gray,
            ) = process_frame(
                frame,
                detector,
                stationary_filter,
                tracker,
                previous_gray,
                blade_detector,
                danger_tracker,
            )
            processed_frames += 1
            report.add_frame(
                frame_number,
                detections,
                current_bboxes,
                danger_track_ids=danger_tracker.danger_track_ids,
                inside_danger_track_ids=danger_tracker.inside_track_ids,
                confirmed_total=tracker.confirmed_total,
            )

            if controls is not None:
                controls.sync_progress(frame_number)
                draw_playback_status(display_frame, controls.status_text())

            if video_output:
                if video_writer is None:
                    output_path = Path(video_output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        video_fps,
                        (display_frame.shape[1], display_frame.shape[0]),
                    )
                    if not video_writer.isOpened():
                        raise OSError(f"無法建立預覽影片: {output_path}")
                video_writer.write(display_frame)

            detection_count = len(detections)
            visible_count = len(visible_track_ids)
            preview_score = visible_count * 10 + detection_count
            filter_is_ready = processed_frames >= STATIONARY_MAX_FRAMES
            if filter_is_ready and preview_score > best_score:
                best_score = preview_score
                best_frame = display_frame.copy()

            if display:
                cv2.imshow(WINDOW_NAME, display_frame)
                execution_time_ms = (time.perf_counter() - loop_start_time) * 1000.0
                if controls.wait(execution_time_ms, target_frame_time_ms):
                    break
                if not controls.has_pending_seek:
                    for _ in range(controls.frame_step - 1):
                        if not cap.grab():
                            break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()

    if preview_output and best_frame is not None:
        output_path = Path(preview_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), best_frame):
            raise OSError(f"無法寫入預覽圖: {output_path}")
        print(f"預覽圖已輸出: {output_path.resolve()}")

    if video_output:
        print(f"預覽影片已輸出: {Path(video_output).resolve()}")

    if excel_output:
        excel_path = report.save(excel_output)
        print(f"Excel 報表已輸出: {excel_path}")

    elapsed = time.perf_counter() - started_at
    fps = processed_frames / elapsed if elapsed > 0 else 0
    print(
        f"完成 {processed_frames} 幀，"
        f"確認總數 {tracker.confirmed_total}，"
        f"平均處理速度 {fps:.2f} FPS"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=str(VIDEO_PATH),
        help="要分析的影片路徑",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="YOLO 模型路徑",
    )
    parser.add_argument(
        "--blade-model",
        default=str(BLADE_MODEL_PATH),
        help="風力發電風扇 YOLO 模型路徑",
    )
    parser.add_argument(
        "--danger-zone",
        default=None,
        help="危險橢圓區，格式 x,y,radius_x,radius_y 或 x,y,radius_x,radius_y,rotation；未指定時使用畫面中央預設區",
    )
    parser.add_argument("--headless", action="store_true", help="不開啟 OpenCV 視窗")
    parser.add_argument("--max-frames", type=int, default=None, help="最多處理幀數")
    parser.add_argument("--start-frame", type=int, default=0, help="開始處理的影片幀")
    parser.add_argument("--preview-output", help="輸出偵測效果預覽圖")
    parser.add_argument("--video-output", help="輸出帶框與軌跡的 MP4 影片")
    parser.add_argument(
        "--detector",
        choices=["tiled", "full"],
        default="full",
        help="tiled 適合極小目標；full 較快",
    )
    parser.add_argument(
        "--tracker",
        choices=["hybrid", "bytetrack", "motion"],
        default="hybrid",
        help="hybrid 適合快速蝙蝠；bytetrack 為官方 baseline",
    )
    parser.add_argument(
        "--excel-output",
        default=str(DEFAULT_REPORT_PATH),
        help="輸出逐幀蝙蝠座標 Excel",
    )
    parser.add_argument("--no-excel", action="store_true", help="不輸出 Excel")
    parser.add_argument("--no-blade", action="store_true", help="不執行風扇模型偵測")
    return parser.parse_args()


def main():
    args = parse_args()
    run_video(
        video_path=args.video,
        model_path=args.model,
        display=not args.headless,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        preview_output=args.preview_output,
        video_output=args.video_output,
        excel_output=None if args.no_excel else args.excel_output,
        detector_name=args.detector,
        tracker_name=args.tracker,
        blade_model_path=args.blade_model,
        enable_blade_detection=not args.no_blade,
        danger_zone=parse_danger_zone(args.danger_zone),
    )


if __name__ == "__main__":
    main()
