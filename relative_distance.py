from dataclasses import dataclass
from math import sqrt


FAR_AREA_MAX = 80.0
MID_AREA_MAX = 220.0


@dataclass(frozen=True)
class RelativeDistanceDescription:
    bbox_width: float
    bbox_height: float
    bbox_area: float
    relative_distance_score: float
    distance_bin: str


def describe_bbox_distance(bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    score = 1.0 / sqrt(area) if area > 0 else 0.0

    if area <= FAR_AREA_MAX:
        distance_bin = "far"
    elif area <= MID_AREA_MAX:
        distance_bin = "mid"
    else:
        distance_bin = "near"

    return RelativeDistanceDescription(
        bbox_width=round(width, 2),
        bbox_height=round(height, 2),
        bbox_area=round(area, 2),
        relative_distance_score=round(score, 4),
        distance_bin=distance_bin,
    )


def detection_distance_label(detection):
    return describe_bbox_distance(detection[:4]).distance_bin
