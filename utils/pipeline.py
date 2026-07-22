from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from anomaly.anomaly_detector import AnomalyDetector
from baseline.baseline_model import BaselineModel
from baseline.personal_baseline import PersonalBaseline
from camera.camera import CameraManager
from features.feature_extractor import FeatureExtractor
from pose.pose_detector import PoseDetector
from utils.config import DEFAULT_FRAME_INTERVAL_SECONDS


logger = logging.getLogger("strokeai.pipeline")


class StrokeAIPipeline:
    """Coordinate camera, pose, features, baseline and anomaly analysis."""

    MIN_BASELINE_SAMPLES = 10

    def __init__(self) -> None:
        self.camera = CameraManager()
        self.pose_detector = PoseDetector()
        self.feature_extractor = FeatureExtractor()

        self.baseline_model = BaselineModel()
        self.personal_baseline = PersonalBaseline(window_seconds=60.0)
        self.anomaly_detector = AnomalyDetector()

        self.history: list[dict[str, float]] = []
        self.last_landmarks: list[Any] | None = None
        self.last_frame: Any | None = None
        self.last_annotated_frame: Any | None = None
        self.last_metrics: dict[str, Any] | None = None

    def process_frame(self) -> dict[str, Any]:
        """Capture and analyze one frame."""
        frame = self.camera.read_frame()

        if frame is None:
            return {
                "status": "error",
                "message": "Unable to read a frame from the camera.",
            }

        annotated_frame, landmarks = self.pose_detector.detect(frame)

        if landmarks is None:
            self.last_frame = frame
            self.last_annotated_frame = annotated_frame

            return {
                "status": "no_pose",
                "message": (
                    "Pose landmarks were not detected. "
                    "Adjust lighting or positioning."
                ),
                "frame": frame,
                "annotated_frame": annotated_frame,
                "baseline_summary": self.personal_baseline.get_summary(),
            }

        feature_row = self.feature_extractor.extract_features(
            landmarks,
            previous_landmarks=self.last_landmarks,
            frame_interval_seconds=DEFAULT_FRAME_INTERVAL_SECONDS,
        )

        self.last_landmarks = landmarks
        self.last_frame = frame
        self.last_annotated_frame = annotated_frame

        timestamp = time.monotonic()

        # Save the current user's numerical features into the rolling baseline.
        self.personal_baseline.add_frame_features(
            feature_row,
            timestamp=timestamp,
        )

        baseline_summary = self.personal_baseline.get_summary()
        sample_count = baseline_summary["sample_count"]

        self.history.append(feature_row)

        # Do not show a fake risk result before enough samples exist.
        if sample_count < self.MIN_BASELINE_SAMPLES:
            metrics = {
                "walking_speed": feature_row.get("walking_speed", 0.0),
                "body_tilt_angle": feature_row.get("body_tilt_angle", 0.0),
                "risk_score": None,
                "risk_level": "Calibrating",
                "reason": (
                    f"Building personal baseline "
                    f"({sample_count}/{self.MIN_BASELINE_SAMPLES} samples)."
                ),
            }

            self.last_metrics = metrics

            return {
                "status": "calibrating",
                "frame": frame,
                "annotated_frame": annotated_frame,
                "features": feature_row,
                "metrics": metrics,
                "history": self.history,
                "baseline_summary": baseline_summary,
            }

        baseline_frame = pd.DataFrame(self.history[:-1])
        current_frame = pd.DataFrame([feature_row])

        if baseline_frame.empty:
            baseline_frame = current_frame.copy()

        self.baseline_model.fit(baseline_frame)
        self.anomaly_detector.fit(baseline_frame)

        risk_result = self.anomaly_detector.analyze_behavior(
            baseline_frame,
            current_frame,
        )

        metrics = {
            "walking_speed": feature_row.get("walking_speed", 0.0),
            "body_tilt_angle": feature_row.get("body_tilt_angle", 0.0),
            "risk_score": risk_result.get("risk_score"),
            "risk_level": risk_result.get("risk_level", "Monitoring"),
            "reason": risk_result.get("reason", "No significant deviation detected."),
        }

        self.last_metrics = metrics

        return {
            "status": "ok",
            "frame": frame,
            "annotated_frame": annotated_frame,
            "features": feature_row,
            "metrics": metrics,
            "history": self.history,
            "risk_result": risk_result,
            "baseline_summary": baseline_summary,
        }

    def release_camera(self) -> None:
        """Stop the camera device if it is active."""
        self.camera.release()