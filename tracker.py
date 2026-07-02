from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from motion_rules import MotionContext, MotionMatcher


class InstantFadeTracker:
    def __init__(
        self,
        max_disappeared=8,
        min_center_distance=6,
        history_size=45,
        min_hits=2,
        min_confirmation_displacement=0,
        min_new_track_score=0.08,
        low_track_score=0.05,
        max_reconnect_disappeared=3,
        max_prediction_points=4,
        motion_matcher=None,
        stationary_radius=8,
        stationary_max_hits=5,
    ):
        self.next_id = 1
        self.confirmed_total = 0
        self.objects = {}
        self.current_bboxes = {}
        self.last_bboxes = {}
        self.kalman_filters = {}
        self.disappeared = {}
        self.hits = {}
        self.det_scores = {}
        self.confirmed = {}
        self.stationary_anchors = {}
        self.stationary_hits = {}
        self.colors = {}
        self.max_disappeared = max_disappeared
        self.min_center_distance = min_center_distance
        self.history_size = history_size
        self.min_hits = min_hits
        self.min_confirmation_displacement = min_confirmation_displacement
        self.min_new_track_score = min_new_track_score
        self.low_track_score = low_track_score
        self.max_reconnect_disappeared = max_reconnect_disappeared
        self.max_prediction_points = max_prediction_points
        self.motion_matcher = motion_matcher or MotionMatcher()
        self.stationary_radius = stationary_radius
        self.stationary_max_hits = stationary_max_hits
        self.active_ids_this_frame = set()

    def _create_kf(self, centroid):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([centroid[0], centroid[1], 0, 0])
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.Q = np.eye(4) * 0.5
        kf.R *= 1.0
        return kf

    @staticmethod
    def _centroid(rect):
        return (rect[:2] + rect[2:]) / 2

    def _normalize_detections(self, detections):
        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)
        if detections.shape[1] == 4:
            scores = np.ones((len(detections), 1), dtype=np.float32)
            return np.hstack([detections, scores])
        return detections[:, :5]

    def _dedupe_detections(self, detections):
        cleaned = []
        input_centroids = []

        detections = self._normalize_detections(detections)
        order = np.argsort(-detections[:, 4]) if len(detections) else []
        for detection in detections[order]:
            rect = detection[:4]
            centroid = self._centroid(rect)
            if any(
                np.linalg.norm(centroid - self._centroid(existing[:4]))
                < self.min_center_distance
                for existing in cleaned
            ):
                continue

            cleaned.append(detection)
            input_centroids.append(centroid.astype(int))

        if not input_centroids:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 2), dtype=int)
        return np.asarray(cleaned, dtype=np.float32), np.asarray(input_centroids, dtype=int)

    def _recent_velocity(self, tid, points=4):
        history = list(self.objects[tid])
        if len(history) < 2:
            return np.zeros(2, dtype=np.float32)

        recent = np.asarray(history[-points:], dtype=np.float32)
        return recent[-1] - recent[0]

    def _association_cost(self, tid, detection, centroid, predicted_centroid):
        if self.disappeared[tid] > self.max_reconnect_disappeared:
            return np.inf

        score = float(detection[4])
        if score < self.low_track_score:
            return np.inf

        context = MotionContext(
            detection=detection,
            centroid=centroid.astype(np.float32),
            predicted_centroid=predicted_centroid.astype(np.float32),
            last_centroid=np.asarray(self.objects[tid][-1], dtype=np.float32),
            recent_velocity=self._recent_velocity(tid),
            previous_bbox=self.last_bboxes.get(tid),
            disappeared=self.disappeared[tid],
            score=score,
        )
        decision = self.motion_matcher.evaluate(context)
        return decision.cost if decision.accepted else np.inf

    def _append_prediction(self, tid):
        if self.disappeared[tid] >= self.max_prediction_points:
            return

        predicted = self.kalman_filters[tid].x[:2].reshape(2)
        self.objects[tid].append((int(predicted[0]), int(predicted[1])))

    def _mark_disappeared(self, track_ids, already_predicted=False):
        for tid in list(track_ids):
            if tid not in self.disappeared:
                continue
            if not already_predicted:
                self.kalman_filters[tid].predict()
            self._append_prediction(tid)
            self.disappeared[tid] += 1
            if self.disappeared[tid] > self.max_disappeared:
                self.deregister(tid)

    def _record_hit(self, tid):
        self.hits[tid] += 1
        history = np.asarray(self.objects[tid], dtype=np.float32)
        displacement = 0.0
        if len(history) >= 2:
            displacement = float(
                np.max(np.linalg.norm(history - history[0], axis=1))
            )
        if (
            not self.confirmed[tid]
            and self.hits[tid] >= self.min_hits
            and displacement >= self.min_confirmation_displacement
        ):
            self.confirmed[tid] = True
            self.confirmed_total += 1

    def _record_motion(self, tid, centroid):
        centroid = np.asarray(centroid, dtype=np.float32)
        anchor = self.stationary_anchors[tid]
        distance = float(np.linalg.norm(centroid - anchor))
        if distance <= self.stationary_radius:
            self.stationary_hits[tid] += 1
        else:
            self.stationary_anchors[tid] = centroid
            self.stationary_hits[tid] = 1

    def _is_stationary_false_positive(self, tid):
        return self.stationary_hits.get(tid, 0) >= self.stationary_max_hits

    def is_confirmed(self, tid):
        return self.confirmed.get(tid, False)

    def _associate(self, object_ids, detection_indices, detections, centroids, predictions):
        if not object_ids or not detection_indices:
            return [], set(object_ids), set(detection_indices)

        costs = np.full((len(object_ids), len(detection_indices)), np.inf)
        for row, tid in enumerate(object_ids):
            for column, detection_index in enumerate(detection_indices):
                costs[row, column] = self._association_cost(
                    tid,
                    detections[detection_index],
                    centroids[detection_index],
                    predictions[tid],
                )

        finite_costs = np.where(np.isfinite(costs), costs, 1e6)
        rows, columns = linear_sum_assignment(finite_costs)
        matches = []
        used_rows = set()
        used_columns = set()

        for row, column in zip(rows, columns):
            if not np.isfinite(costs[row, column]):
                continue
            if costs[row, column] > self.motion_matcher.max_cost:
                continue
            matches.append((object_ids[row], detection_indices[column]))
            used_rows.add(row)
            used_columns.add(column)

        unmatched_tracks = {
            object_ids[row]
            for row in range(len(object_ids))
            if row not in used_rows
        }
        unmatched_detections = {
            detection_indices[column]
            for column in range(len(detection_indices))
            if column not in used_columns
        }
        return matches, unmatched_tracks, unmatched_detections

    def _apply_match(self, tid, detection, centroid):
        rect = detection[:4]
        score = float(detection[4])
        self.kalman_filters[tid].update(centroid)
        self.objects[tid].append(centroid)
        self.current_bboxes[tid] = rect
        self.last_bboxes[tid] = rect
        self.det_scores[tid] = score

        self._record_motion(tid, centroid)
        if self._is_stationary_false_positive(tid):
            self.deregister(tid, false_positive=True)
            return False

        self._record_hit(tid)
        self.disappeared[tid] = 0
        self.active_ids_this_frame.add(tid)
        return True

    def update(self, detections, frame=None):
        self.active_ids_this_frame.clear()
        self.current_bboxes.clear()

        cleaned_detections, input_centroids = self._dedupe_detections(detections)

        if len(input_centroids) == 0:
            self._mark_disappeared(self.disappeared.keys())
            return self.objects, self.colors, set(), self.current_bboxes

        if len(self.objects) == 0:
            for i, pt in enumerate(input_centroids):
                if cleaned_detections[i, 4] < self.min_new_track_score:
                    continue
                tid = self.register(pt, cleaned_detections[i])
                self.active_ids_this_frame.add(tid)
        else:
            object_ids = list(self.objects.keys())
            for tid in object_ids:
                self.kalman_filters[tid].predict()
            predictions = {
                tid: self.kalman_filters[tid].x[:2].reshape(2)
                for tid in object_ids
            }
            scores = cleaned_detections[:, 4]
            high_indices = np.flatnonzero(scores >= self.min_new_track_score).tolist()
            low_indices = np.flatnonzero(
                (scores >= self.low_track_score)
                & (scores < self.min_new_track_score)
            ).tolist()

            high_matches, unmatched_tracks, unmatched_high = self._associate(
                object_ids,
                high_indices,
                cleaned_detections,
                input_centroids,
                predictions,
            )
            for tid, detection_index in high_matches:
                self._apply_match(
                    tid,
                    cleaned_detections[detection_index],
                    input_centroids[detection_index],
                )

            remaining_tracks = [
                tid for tid in object_ids
                if tid in unmatched_tracks and tid in self.objects
            ]
            low_matches, missing_ids, _ = self._associate(
                remaining_tracks,
                low_indices,
                cleaned_detections,
                input_centroids,
                predictions,
            )
            for tid, detection_index in low_matches:
                self._apply_match(
                    tid,
                    cleaned_detections[detection_index],
                    input_centroids[detection_index],
                )

            self._mark_disappeared(
                [tid for tid in missing_ids if tid in self.objects],
                already_predicted=True,
            )
            for detection_index in unmatched_high:
                tid = self.register(
                    input_centroids[detection_index],
                    cleaned_detections[detection_index],
                )
                self.active_ids_this_frame.add(tid)

        return self.objects, self.colors, self.active_ids_this_frame, self.current_bboxes

    def register(self, centroid, detection):
        bbox = detection[:4]
        score = float(detection[4])
        self.objects[self.next_id] = deque(maxlen=self.history_size)
        self.objects[self.next_id].append(centroid)

        self.current_bboxes[self.next_id] = bbox
        self.last_bboxes[self.next_id] = bbox

        self.kalman_filters[self.next_id] = self._create_kf(centroid)
        self.disappeared[self.next_id] = 0
        self.hits[self.next_id] = 1
        self.det_scores[self.next_id] = score
        self.confirmed[self.next_id] = (
            self.min_hits <= 1
            and self.min_confirmation_displacement <= 0
        )
        self.stationary_anchors[self.next_id] = np.asarray(centroid, dtype=np.float32)
        self.stationary_hits[self.next_id] = 1
        if self.confirmed[self.next_id]:
            self.confirmed_total += 1
        rng = np.random.default_rng(self.next_id + 123)
        self.colors[self.next_id] = tuple(rng.integers(100, 255, 3).tolist())
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, tid, false_positive=False):
        if false_positive and self.confirmed.get(tid, False):
            self.confirmed_total = max(0, self.confirmed_total - 1)

        for data in [
            self.objects,
            self.kalman_filters,
            self.disappeared,
            self.hits,
            self.det_scores,
            self.confirmed,
            self.stationary_anchors,
            self.stationary_hits,
            self.colors,
            self.current_bboxes,
            self.last_bboxes,
        ]:
            if tid in data:
                del data[tid]
