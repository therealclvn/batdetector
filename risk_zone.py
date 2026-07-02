from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np


DEFAULT_RADIUS_X_RATIO = 0.168
DEFAULT_RADIUS_Y_RATIO = 0.14


@dataclass(frozen=True)
class DangerZone:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    rotation_degrees: float = 0.0

    @classmethod
    def from_frame(cls, frame_width, frame_height):
        return cls(
            center_x=frame_width / 2,
            center_y=frame_height / 2,
            radius_x=frame_width * DEFAULT_RADIUS_X_RATIO,
            radius_y=frame_height * DEFAULT_RADIUS_Y_RATIO,
        )

    def contains_point(self, point):
        x, y = [float(value) for value in point[:2]]
        dx = x - self.center_x
        dy = y - self.center_y
        angle = radians(-self.rotation_degrees)
        rotated_x = (dx * cos(angle)) - (dy * sin(angle))
        rotated_y = (dx * sin(angle)) + (dy * cos(angle))

        normalized = (
            (rotated_x / max(self.radius_x, 1e-6)) ** 2
            + (rotated_y / max(self.radius_y, 1e-6)) ** 2
        )
        return normalized <= 1.0


@dataclass(frozen=True)
class DangerZoneStats:
    danger_count: int
    confirmed_total: int
    impact_risk_percent: float


class DangerZoneTracker:
    def __init__(self, zone):
        self.zone = zone
        self.danger_track_ids = set()
        self.inside_track_ids = set()

    def update(self, confirmed_track_ids, current_bboxes):
        self.inside_track_ids = set()

        for track_id in confirmed_track_ids:
            if track_id not in current_bboxes:
                continue

            center = _bbox_center(current_bboxes[track_id])
            if self.zone.contains_point(center):
                self.inside_track_ids.add(track_id)
                self.danger_track_ids.add(track_id)

        return self.inside_track_ids

    def is_inside(self, track_id):
        return track_id in self.inside_track_ids

    def is_dangerous(self, track_id):
        return track_id in self.danger_track_ids

    def stats(self, confirmed_total):
        confirmed_total = max(0, int(confirmed_total))
        danger_count = min(len(self.danger_track_ids), confirmed_total)
        impact_risk = 0.0
        if confirmed_total:
            impact_risk = round((danger_count / confirmed_total) * 100, 2)

        return DangerZoneStats(
            danger_count=danger_count,
            confirmed_total=confirmed_total,
            impact_risk_percent=impact_risk,
        )


def parse_danger_zone(value):
    if not value:
        return None

    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (4, 5):
        raise ValueError("危險區格式需為 x,y,radius_x,radius_y 或 x,y,radius_x,radius_y,rotation")

    numbers = [float(part) for part in parts]
    rotation = numbers[4] if len(numbers) == 5 else 0.0
    return DangerZone(
        center_x=numbers[0],
        center_y=numbers[1],
        radius_x=numbers[2],
        radius_y=numbers[3],
        rotation_degrees=rotation,
    )


def _bbox_center(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        dtype=np.float32,
    )
