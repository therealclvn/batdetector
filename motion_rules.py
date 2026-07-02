from dataclasses import dataclass

import numpy as np


@dataclass
class MotionContext:
    detection: np.ndarray
    centroid: np.ndarray
    predicted_centroid: np.ndarray
    last_centroid: np.ndarray
    recent_velocity: np.ndarray
    previous_bbox: np.ndarray | None
    disappeared: int
    score: float


@dataclass
class MotionDecision:
    accepted: bool
    cost: float


def bbox_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0

    union = bbox_area(box_a) + bbox_area(box_b) - intersection
    return intersection / max(union, 1.0)


def cosine_similarity(vec_a, vec_b):
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 1.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def area_change_penalty(previous_bbox, current_bbox):
    if previous_bbox is None:
        return 0.0
    previous_area = max(bbox_area(previous_bbox), 1.0)
    current_area = max(bbox_area(current_bbox), 1.0)
    return abs(np.log(current_area / previous_area))


class ShortStepMotionRule:
    """For bats that move only a little between adjacent frames."""

    def __init__(self, max_step=34, max_fast_track_speed=24):
        self.max_step = max_step
        self.max_fast_track_speed = max_fast_track_speed

    def evaluate(self, context):
        current_bbox = context.detection[:4]
        step = float(np.linalg.norm(context.centroid - context.last_centroid))
        predicted_error = float(np.linalg.norm(context.centroid - context.predicted_centroid))
        recent_speed = float(np.linalg.norm(context.recent_velocity))
        iou = bbox_iou(context.previous_bbox, current_bbox)

        if step > self.max_step and iou < 0.12:
            return MotionDecision(False, np.inf)
        if recent_speed > self.max_fast_track_speed and step < recent_speed * 0.18:
            return MotionDecision(False, np.inf)

        size_penalty = area_change_penalty(context.previous_bbox, current_bbox)
        if size_penalty > 1.2 and iou < 0.08:
            return MotionDecision(False, np.inf)

        score_penalty = max(0.0, 0.14 - context.score) * 1.5
        cost = (predicted_error / max(self.max_step, 1)) - (0.45 * iou)
        cost += 0.20 * size_penalty + score_penalty
        return MotionDecision(True, cost)


class LongStepMotionRule:
    """For bats that jump farther between frames because they fly fast."""

    def __init__(self, min_step=18, max_step=120):
        self.min_step = min_step
        self.max_step = max_step

    def evaluate(self, context):
        current_bbox = context.detection[:4]
        step_vec = context.centroid - context.last_centroid
        step = float(np.linalg.norm(step_vec))
        predicted_error = float(np.linalg.norm(context.centroid - context.predicted_centroid))
        recent_speed = float(np.linalg.norm(context.recent_velocity))
        direction = cosine_similarity(context.recent_velocity, step_vec)
        iou = bbox_iou(context.previous_bbox, current_bbox)

        if step < self.min_step and iou < 0.10:
            return MotionDecision(False, np.inf)
        if step > self.max_step and iou < 0.04:
            return MotionDecision(False, np.inf)

        required_direction = 0.20 if context.disappeared else -0.10
        if recent_speed > 8 and direction < required_direction and iou < 0.10:
            return MotionDecision(False, np.inf)

        if context.disappeared and predicted_error > self.max_step * 0.8 and iou < 0.08:
            return MotionDecision(False, np.inf)

        size_penalty = area_change_penalty(context.previous_bbox, current_bbox)
        if size_penalty > 1.4 and iou < 0.08:
            return MotionDecision(False, np.inf)

        direction_penalty = max(0.0, 1.0 - direction) * 0.25
        score_penalty = max(0.0, 0.14 - context.score) * 1.8
        missed_penalty = context.disappeared * 0.30
        cost = (predicted_error / max(self.max_step, 1)) - (0.35 * iou)
        cost += direction_penalty + (0.20 * size_penalty) + score_penalty + missed_penalty
        return MotionDecision(True, cost)


class MotionMatcher:
    def __init__(self, short_rule=None, long_rule=None, max_cost=1.20):
        self.short_rule = short_rule or ShortStepMotionRule()
        self.long_rule = long_rule or LongStepMotionRule()
        self.max_cost = max_cost

    def evaluate(self, context):
        decisions = [
            self.short_rule.evaluate(context),
            self.long_rule.evaluate(context),
        ]
        accepted = [decision for decision in decisions if decision.accepted]
        if not accepted:
            return MotionDecision(False, np.inf)

        best = min(accepted, key=lambda decision: decision.cost)
        if best.cost > self.max_cost:
            return MotionDecision(False, np.inf)
        return best
