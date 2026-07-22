import tempfile
import unittest
from pathlib import Path

from baseline.personal_baseline import PersonalBaseline


class PersonalBaselineTests(unittest.TestCase):
    def test_records_features_and_builds_windowed_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "baseline.csv"
            baseline = PersonalBaseline(window_seconds=60.0, storage_path=output_path)

            baseline.add_frame_features({"body_center_x": 0.1, "body_center_y": 0.2}, timestamp=0.0)
            baseline.add_frame_features({"body_center_x": 0.3, "body_center_y": 0.4}, timestamp=10.0)

            summary = baseline.get_summary()

            self.assertEqual(summary["sample_count"], 2)
            self.assertIn("body_center_x", summary["features"])
            self.assertAlmostEqual(summary["features"]["body_center_x"]["mean"], 0.2)
            self.assertAlmostEqual(summary["features"]["body_center_x"]["std"], 0.1)
            self.assertAlmostEqual(summary["features"]["body_center_x"]["moving_average"], 0.2)
            self.assertTrue(output_path.exists())

    def test_drops_frames_outside_the_rolling_window(self) -> None:
        baseline = PersonalBaseline(window_seconds=30.0)

        baseline.add_frame_features({"body_center_x": 1.0}, timestamp=0.0)
        baseline.add_frame_features({"body_center_x": 2.0}, timestamp=10.0)
        baseline.add_frame_features({"body_center_x": 3.0}, timestamp=40.0)

        summary = baseline.get_summary()

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["features"]["body_center_x"]["mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
