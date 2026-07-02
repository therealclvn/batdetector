import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from detection_pipeline import non_maximum_suppression
from excel_report import BatDetectionReport
from main import (
    BLADE_CONFIDENCE,
    HIGH_CONFIDENCE,
    build_counter_lines,
    create_optional_blade_detector,
    create_tracker,
    draw_blade_boxes,
    filter_detection_boxes,
    format_track_label,
)
from relative_distance import describe_bbox_distance, detection_distance_label
from risk_zone import DangerZone, DangerZoneStats, DangerZoneTracker


class DetectionPipelineTests(unittest.TestCase):
    def test_non_maximum_suppression_keeps_best_overlapping_box(self):
        detections = np.asarray(
            [
                [10, 10, 20, 20, 0.40],
                [11, 11, 21, 21, 0.90],
                [100, 100, 110, 110, 0.30],
            ],
            dtype=np.float32,
        )

        kept = non_maximum_suppression(detections, iou_threshold=0.5)

        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(float(kept[0, 4]), 0.90)
        self.assertTrue(np.any(np.all(kept[:, :4] == [100, 100, 110, 110], axis=1)))


class MainFilteringTests(unittest.TestCase):
    def test_filter_detection_boxes_rejects_invalid_box_shapes(self):
        detections = np.asarray(
            [
                [0, 0, 3, 2, 0.90],
                [0, 0, 20, 20, 0.80],
                [0, 0, 120, 20, 0.70],
                [0, 0, 20, 200, 0.60],
            ],
            dtype=np.float32,
        )

        filtered = filter_detection_boxes(detections)

        self.assertEqual(filtered.tolist(), [[0.0, 0.0, 20.0, 20.0, 0.800000011920929]])


class BladeDetectionTests(unittest.TestCase):
    def test_optional_blade_detector_skips_missing_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "blade_best.pt"
            detector = create_optional_blade_detector(
                missing_path,
                load_model_fn=lambda path: self.fail(f"unexpected model load: {path}"),
                warn_on_missing=False,
            )

        self.assertIsNone(detector)

    def test_optional_blade_detector_uses_existing_model_path(self):
        loaded_paths = []

        class DummyModel:
            def predict(self, *args, **kwargs):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "blade_best.pt"
            model_path.write_bytes(b"fake model")

            detector = create_optional_blade_detector(
                model_path,
                load_model_fn=lambda path: loaded_paths.append(path) or DummyModel(),
            )

        self.assertIsNotNone(detector)
        self.assertEqual(loaded_paths, [model_path.resolve()])
        self.assertEqual(detector.confidence, BLADE_CONFIDENCE)
        self.assertEqual(detector.grid_size, 1)

    def test_draw_blade_boxes_marks_frame(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detections = np.asarray([[10, 10, 30, 30, 0.91]], dtype=np.float32)

        draw_blade_boxes(frame, detections)

        self.assertGreater(int(frame.sum()), 0)


class DangerZoneTests(unittest.TestCase):
    def test_default_danger_zone_is_scaled_to_seventy_percent(self):
        zone = DangerZone.from_frame(frame_width=1000, frame_height=500)

        self.assertEqual(zone.center_x, 500)
        self.assertEqual(zone.center_y, 250)
        self.assertAlmostEqual(zone.radius_x, 168)
        self.assertAlmostEqual(zone.radius_y, 70)

    def test_danger_zone_tracks_unique_confirmed_ids_that_enter_ellipse(self):
        zone = DangerZone(center_x=100, center_y=100, radius_x=50, radius_y=25)
        tracker = DangerZoneTracker(zone)

        tracker.update(
            confirmed_track_ids={1, 2},
            current_bboxes={
                1: np.asarray([90, 90, 110, 110], dtype=np.float32),
                2: np.asarray([180, 90, 200, 110], dtype=np.float32),
            },
        )
        tracker.update(
            confirmed_track_ids={1, 2},
            current_bboxes={
                1: np.asarray([95, 95, 115, 115], dtype=np.float32),
                2: np.asarray([90, 90, 110, 110], dtype=np.float32),
            },
        )

        stats = tracker.stats(confirmed_total=3)

        self.assertTrue(zone.contains_point((100, 100)))
        self.assertFalse(zone.contains_point((180, 100)))
        self.assertEqual(tracker.danger_track_ids, {1, 2})
        self.assertEqual(stats.danger_count, 2)
        self.assertEqual(stats.confirmed_total, 3)
        self.assertAlmostEqual(stats.impact_risk_percent, 66.67)

    def test_counter_lines_show_only_impact_percent_without_blade_or_ratio(self):
        lines = build_counter_lines(
            candidate_count=4,
            current_count=2,
            total_count=5,
            blade_count=3,
            danger_stats=DangerZoneStats(
                danger_count=2,
                confirmed_total=5,
                impact_risk_percent=40.0,
            ),
        )
        labels = [line[0] for line in lines]

        self.assertEqual(
            labels,
            [
                "Candidates: 4",
                "Active Bats: 2",
                "Confirmed Flights: 5",
                "Impact Risk: 40.0%",
            ],
        )
        self.assertFalse(any("Wind Blades" in label for label in labels))
        self.assertFalse(any("Danger Bats" in label for label in labels))


class TrackerFactoryTests(unittest.TestCase):
    def test_motion_tracker_is_not_the_hybrid_low_confidence_update_mode(self):
        hybrid_tracker = create_tracker("hybrid", frame_rate=9)
        motion_tracker = create_tracker("motion", frame_rate=9)

        self.assertLess(hybrid_tracker.low_track_score, hybrid_tracker.min_new_track_score)
        self.assertEqual(motion_tracker.min_new_track_score, HIGH_CONFIDENCE)
        self.assertEqual(motion_tracker.low_track_score, motion_tracker.min_new_track_score)


class RelativeDistanceTests(unittest.TestCase):
    def test_describe_bbox_distance_uses_area_as_relative_distance_proxy(self):
        description = describe_bbox_distance([10, 20, 30, 40])

        self.assertEqual(description.bbox_width, 20.0)
        self.assertEqual(description.bbox_height, 20.0)
        self.assertEqual(description.bbox_area, 400.0)
        self.assertAlmostEqual(description.relative_distance_score, 0.05)
        self.assertEqual(description.distance_bin, "near")

    def test_describe_bbox_distance_bins_small_boxes_as_far(self):
        description = describe_bbox_distance([0, 0, 6, 8])

        self.assertEqual(description.bbox_area, 48.0)
        self.assertEqual(description.distance_bin, "far")
        self.assertEqual(detection_distance_label([0, 0, 6, 8, 0.75]), "far")

    def test_confirmed_track_label_includes_relative_distance_bin(self):
        self.assertEqual(format_track_label(3, [10, 20, 30, 40]), "ID:3 near")


class ExcelReportTests(unittest.TestCase):
    def test_report_save_writes_a_valid_two_sheet_xlsx(self):
        report = BatDetectionReport()
        report.add_frame(
            42,
            np.asarray([[10, 20, 30, 40, 0.75]], dtype=np.float32),
            {7: np.asarray([10, 20, 30, 40], dtype=np.float32)},
            danger_track_ids={7},
            confirmed_total=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.xlsx"
            saved_path = report.save(output_path)

            with ZipFile(saved_path) as archive:
                names = set(archive.namelist())
                self.assertIn("xl/worksheets/sheet1.xml", names)
                self.assertIn("xl/worksheets/sheet2.xml", names)
                detail_xml = archive.read("xl/worksheets/sheet2.xml").decode("utf-8")

        self.assertIn("<v>42</v>", detail_xml)
        self.assertIn("<v>7</v>", detail_xml)
        self.assertIn("bbox_width", detail_xml)
        self.assertIn("bbox_height", detail_xml)
        self.assertIn("bbox_area", detail_xml)
        self.assertIn("relative_distance_score", detail_xml)
        self.assertIn("distance_bin", detail_xml)
        self.assertIn("inside_danger_zone", detail_xml)
        self.assertIn("ever_dangerous", detail_xml)
        self.assertIn("danger_entry_rate_percent", detail_xml)
        self.assertIn("<v>400.0</v>", detail_xml)
        self.assertIn("near", detail_xml)
        self.assertIn("yes", detail_xml)
        self.assertIn("<v>10</v>", detail_xml)


if __name__ == "__main__":
    unittest.main()
