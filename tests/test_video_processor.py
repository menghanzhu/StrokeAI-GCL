import unittest
from unittest.mock import MagicMock, patch

from camera.video_processor import VideoProcessor


class VideoProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests verify temporal calculation logic only.
        # Prevent real MediaPipe initialization during unit tests.
        self.pose_detector_patcher = patch(
            "camera.video_processor.PoseDetector",
            autospec=True,
        )
        mock_pose_detector_class = self.pose_detector_patcher.start()
        mock_pose_detector_class.return_value = MagicMock()

        self.processor = VideoProcessor()

    def tearDown(self) -> None:
        close_method = getattr(self.processor, "close", None)
        if callable(close_method):
            close_method()

        self.pose_detector_patcher.stop()

    def test_speed_calculation_across_sequential_frames(self) -> None:
        first_row = {
            "body_center_x": 0.50,
            "body_center_y": 0.50,
            "body_tilt_angle": 5.0,
            "head_tilt_angle": 4.0,
            "shoulder_symmetry": 0.02,
            "hip_symmetry": 0.03,
        }

        second_row = {
            "body_center_x": 0.62,
            "body_center_y": 0.51,
            "body_tilt_angle": 7.0,
            "head_tilt_angle": 5.0,
            "shoulder_symmetry": 0.05,
            "hip_symmetry": 0.04,
        }

        self.processor.update_temporal_metrics(
            first_row,
            timestamp=100.0,
            pose_detected=True,
        )

        result = self.processor.update_temporal_metrics(
            second_row,
            timestamp=101.0,
            pose_detected=True,
        )

        self.assertGreater(result["movement_speed"], 0.0)
        self.assertGreater(result["smoothed_speed"], 0.0)

    def test_no_pose_handling(self) -> None:
        result = self.processor.update_temporal_metrics(
            {},
            timestamp=100.0,
            pose_detected=False,
        )

        self.assertEqual(
            result["monitoring_status"],
            "No person detected",
        )
        self.assertEqual(
            result["risk_state"],
            "Not available",
        )
        self.assertEqual(
            result["movement_speed"],
            0.0,
        )

    def test_calibration_state(self) -> None:
        for index in range(8):
            self.processor.update_temporal_metrics(
                {
                    "body_center_x": 0.50 + index * 0.001,
                    "body_center_y": 0.50,
                    "body_tilt_angle": 5.0,
                    "head_tilt_angle": 4.0,
                    "shoulder_symmetry": 0.02,
                    "hip_symmetry": 0.03,
                },
                timestamp=float(index),
                pose_detected=True,
            )

        result = self.processor.update_temporal_metrics(
            {
                "body_center_x": 0.51,
                "body_center_y": 0.50,
                "body_tilt_angle": 5.0,
                "head_tilt_angle": 4.0,
                "shoulder_symmetry": 0.02,
                "hip_symmetry": 0.03,
            },
            timestamp=9.0,
            pose_detected=True,
        )

        self.assertEqual(
            result["monitoring_status"],
            "Calibrating",
        )
        self.assertEqual(
            result["risk_state"],
            "Calibrating",
        )

    def test_anomaly_scoring(self) -> None:
        # Build enough stable frames to finish calibration.
        for index in range(10):
            self.processor.update_temporal_metrics(
                {
                    "body_center_x": 0.50,
                    "body_center_y": 0.50,
                    "body_tilt_angle": 5.0,
                    "head_tilt_angle": 4.0,
                    "shoulder_symmetry": 0.02,
                    "hip_symmetry": 0.03,
                },
                timestamp=100.0 + index,
                pose_detected=True,
            )

        # Add one abnormal frame after calibration.
        result = self.processor.update_temporal_metrics(
            {
                "body_center_x": 0.50,
                "body_center_y": 0.80,
                "body_tilt_angle": 35.0,
                "head_tilt_angle": 20.0,
                "shoulder_symmetry": 0.30,
                "hip_symmetry": 0.25,
            },
            timestamp=111.0,
            pose_detected=True,
        )

        self.assertGreater(result["risk_score"], 0.0)
        self.assertEqual(
            result["risk_state"],
            "Possible anomaly",
        )
        self.assertIn(
            "large body tilt change",
            result["reasons"],
        )
        self.assertIn(
            "large left/right asymmetry",
            result["reasons"],
        )

    def test_history_buffer_limit(self) -> None:
        processor = VideoProcessor(max_history=300)

        try:
            for index in range(310):
                processor.append_history_entry(
                    {
                        "timestamp": float(index),
                        "movement_speed": 0.1,
                        "smoothed_speed": 0.1,
                        "body_tilt": 5.0,
                        "shoulder_symmetry": 0.02,
                        "hip_symmetry": 0.03,
                        "pose_confidence": 0.9,
                        "monitoring_status": "Monitoring",
                        "risk_score": 10.0,
                        "risk_state": "Monitoring",
                        "reasons": [],
                    }
                )

            self.assertLessEqual(
                len(processor.history_records),
                300,
            )
        finally:
            close_method = getattr(processor, "close", None)
            if callable(close_method):
                close_method()


if __name__ == "__main__":
    unittest.main()