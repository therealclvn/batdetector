import cv2
import numpy as np


class TiledYoloDetector:
    """Runs YOLO on overlapping tiles and merges duplicate detections."""

    def __init__(
        self,
        model,
        confidence=0.06,
        image_size=640,
        grid_size=2,
        overlap=0.25,
        nms_iou=0.45,
    ):
        self.model = model
        self.confidence = confidence
        self.image_size = image_size
        self.grid_size = grid_size
        self.overlap = overlap
        self.nms_iou = nms_iou

    def predict(self, frame):
        tiles = self._make_tiles(frame)
        images = [tile["image"] for tile in tiles]
        results = self.model.predict(
            images,
            conf=self.confidence,
            iou=0.7,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []
        for tile, result in zip(tiles, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            xyxy[:, [0, 2]] += tile["x"]
            xyxy[:, [1, 3]] += tile["y"]
            detections.extend(
                np.column_stack([xyxy, scores]).astype(np.float32)
            )

        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return non_maximum_suppression(
            np.asarray(detections, dtype=np.float32),
            self.nms_iou,
        )

    def _make_tiles(self, frame):
        height, width = frame.shape[:2]
        if self.grid_size <= 1:
            return [{"image": frame, "x": 0, "y": 0}]

        tile_width = int(np.ceil(width / (self.grid_size - self.overlap)))
        tile_height = int(np.ceil(height / (self.grid_size - self.overlap)))
        x_starts = _tile_starts(width, tile_width, self.grid_size)
        y_starts = _tile_starts(height, tile_height, self.grid_size)

        return [
            {
                "image": frame[y : y + tile_height, x : x + tile_width],
                "x": x,
                "y": y,
            }
            for y in y_starts
            for x in x_starts
        ]


def non_maximum_suppression(detections, iou_threshold):
    if len(detections) == 0:
        return detections

    boxes = detections[:, :4]
    scores = detections[:, 4]
    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        overlaps = _box_iou(boxes[current], boxes[remaining])
        order = remaining[overlaps <= iou_threshold]

    return detections[keep]


def motion_support_mask(previous_gray, current_frame, detections, threshold=8.0):
    """Returns detections supported by local temporal change.

    The mask is diagnostic for now: high-confidence detections are still kept,
    while weak detections can use this signal during candidate validation.
    """
    if previous_gray is None or len(detections) == 0:
        return np.ones(len(detections), dtype=bool)

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(previous_gray, current_gray)
    support = []

    for x1, y1, x2, y2 in detections[:, :4].astype(int):
        padding = 4
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(current_gray.shape[1], x2 + padding)
        y2 = min(current_gray.shape[0], y2 + padding)
        patch = difference[y1:y2, x1:x2]
        support.append(bool(patch.size and np.percentile(patch, 85) >= threshold))

    return np.asarray(support, dtype=bool)


def _tile_starts(full_size, tile_size, grid_size):
    if tile_size >= full_size:
        return [0]
    return [
        int(round(position))
        for position in np.linspace(0, full_size - tile_size, grid_size)
    ]


def _box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + areas - intersection
    return intersection / np.maximum(union, 1e-6)
