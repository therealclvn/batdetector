from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class DetectionBatch:
    def __init__(self, xyxy, confidence):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.zeros(len(self.xyxy), dtype=np.float32)

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, index):
        return DetectionBatch(self.xyxy[index], self.conf[index])

    @property
    def xywh(self):
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        boxes = np.atleast_2d(self.xyxy).copy()
        boxes[:, 2:] -= boxes[:, :2]
        boxes[:, :2] += boxes[:, 2:] / 2
        return boxes


class BatByteTracker:
    """ByteTrack adapter with bat-specific confirmation and display state."""

    def __init__(
        self,
        frame_rate,
        high_threshold=0.25,
        low_threshold=0.06,
        new_track_threshold=0.25,
        track_buffer=12,
        match_threshold=0.90,
        history_size=45,
        confirmation_hits=3,
        confirmation_displacement=5.0,
        tracking_box_scale=2.2,
        stationary_radius=8.0,
        stationary_frames=6,
    ):
        args = SimpleNamespace(
            track_high_thresh=high_threshold,
            track_low_thresh=low_threshold,
            new_track_thresh=new_track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            fuse_score=True,
        )
        self.tracker = BYTETracker(args=args, frame_rate=max(1, int(frame_rate)))
        self.history_size = history_size
        self.confirmation_hits = confirmation_hits
        self.confirmation_displacement = confirmation_displacement
        self.tracking_box_scale = tracking_box_scale
        self.stationary_radius = stationary_radius
        self.stationary_frames = stationary_frames

        self.objects = defaultdict(lambda: deque(maxlen=history_size))
        self.current_bboxes = {}
        self.scores = {}
        self.hits = defaultdict(int)
        self.confirmed = set()
        self.rejected = set()
        self.colors = {}
        self.confirmed_total = 0
        self.last_seen_frame = {}

    def update(self, detections, frame):
        detections = _normalize_detections(detections)
        tracker_boxes = _scale_boxes(
            detections[:, :4],
            self.tracking_box_scale,
            frame.shape[1],
            frame.shape[0],
        )
        batch = DetectionBatch(tracker_boxes, detections[:, 4])
        tracks = self.tracker.update(batch, frame)

        self.current_bboxes = {}
        active_ids = set()
        for track in tracks:
            track_box = track[:4]
            track_id = int(track[4])
            track_score = float(track[5])
            detection_index = _nearest_detection(track_box, detections)
            if detection_index is None:
                continue

            bbox = detections[detection_index, :4].copy()
            score = float(detections[detection_index, 4])
            center = _center(bbox)

            self.current_bboxes[track_id] = bbox
            self.scores[track_id] = max(score, track_score)
            self.objects[track_id].append(tuple(center.astype(int)))
            self.hits[track_id] += 1
            self.last_seen_frame[track_id] = self.tracker.frame_id
            active_ids.add(track_id)

            if self._is_stationary(track_id):
                self.rejected.add(track_id)
                self.confirmed.discard(track_id)
                self.current_bboxes.pop(track_id, None)
                active_ids.discard(track_id)
                continue

            if track_id not in self.confirmed and self._can_confirm(track_id):
                self.confirmed.add(track_id)
                self.confirmed_total += 1

            if track_id not in self.colors:
                rng = np.random.default_rng(track_id + 123)
                self.colors[track_id] = tuple(
                    int(value) for value in rng.integers(100, 255, 3)
                )

        self._cleanup()
        return self.objects, self.colors, active_ids, self.current_bboxes

    def is_confirmed(self, track_id):
        return track_id in self.confirmed and track_id not in self.rejected

    def _can_confirm(self, track_id):
        if self.hits[track_id] < self.confirmation_hits:
            return False
        points = np.asarray(self.objects[track_id], dtype=np.float32)
        displacement = np.linalg.norm(points - points[0], axis=1)
        return float(np.max(displacement)) >= self.confirmation_displacement

    def _is_stationary(self, track_id):
        if self.hits[track_id] < self.stationary_frames:
            return False
        points = np.asarray(
            list(self.objects[track_id])[-self.stationary_frames :],
            dtype=np.float32,
        )
        return float(np.max(np.linalg.norm(points - points[0], axis=1))) <= self.stationary_radius

    def _cleanup(self):
        oldest_frame = self.tracker.frame_id - max(
            self.history_size,
            self.tracker.max_time_lost + 2,
        )
        for track_id, last_frame in list(self.last_seen_frame.items()):
            if last_frame >= oldest_frame:
                continue
            self.objects.pop(track_id, None)
            self.scores.pop(track_id, None)
            self.hits.pop(track_id, None)
            self.colors.pop(track_id, None)
            self.last_seen_frame.pop(track_id, None)
            self.rejected.discard(track_id)


def _normalize_detections(detections):
    detections = np.asarray(detections, dtype=np.float32)
    if detections.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    return detections.reshape(-1, 5)


def _scale_boxes(boxes, scale, frame_width, frame_height):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes.copy()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = (boxes[:, 2:] - boxes[:, :2]) * scale
    boxes[:, :2] = centers - sizes / 2
    boxes[:, 2:] = centers + sizes / 2
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height - 1)
    return boxes


def _nearest_detection(track_box, detections, max_distance=28.0):
    if len(detections) == 0:
        return None
    track_center = _center(track_box)
    centers = np.asarray([_center(box) for box in detections[:, :4]])
    distances = np.linalg.norm(centers - track_center, axis=1)
    index = int(np.argmin(distances))
    return index if distances[index] <= max_distance else None


def _center(box):
    return np.asarray(
        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
        dtype=np.float32,
    )
