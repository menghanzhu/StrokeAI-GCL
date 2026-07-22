from __future__ import annotations

import logging
import threading
import time
from typing import Any

import av
import numpy as np
from streamlit_webrtc import VideoTransformerBase

from features.feature_extractor import FeatureExtractor
from pose.pose_detector import PoseDetector


logger = logging.getLogger("strokeai.video")


class VideoProcessor(VideoTransformerBase):
    """Process WebRTC frames and calculate real-time movement metrics."""

    MIN_CALIBRATION_FRAMES = 10

    def __init__(self, max_history: int = 300) -> None:
        self.max_history = max_history

        self.pose_detector = PoseDetector()
        self.feature_extractor = FeatureExtractor()

        self._lock = threading.RLock()

        self._last_inference_time = 0.0

        # Analyze approximately five frames per second.
        self._inference_interval = 0.20

        self._latest_frame: np.ndarray | None = None
        self._latest_metrics: dict[str, Any] | None = None

        self.temporal_buffer: list[dict[str, float]] = []
        self.history_records: list[dict[str, Any]] = []

        self._last_history_timestamp = 0.0
        self._valid_pose_frames = 0
        self._previous_landmarks: list[Any] | None = None

    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    @property
    def latest_metrics(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_metrics is None:
                return None

            return dict(self._latest_metrics)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one browser webcam frame."""
        frame_array = frame.to_ndarray(format="bgr24")
        current_time = time.monotonic()

        annotated_frame = frame_array
        metrics: dict[str, Any] | None = None

        if (
            current_time - self._last_inference_time
            >= self._inference_interval
        ):
            previous_inference_time = self._last_inference_time
            self._last_inference_time = current_time

            annotated_frame, landmarks = (
                self.pose_detector.detect(frame_array)
            )

            if landmarks is None:
                metrics = self.update_temporal_metrics(
                    {},
                    timestamp=current_time,
                    pose_detected=False,
                )

            else:
                if previous_inference_time:
                    frame_interval_seconds = max(
                        0.01,
                        current_time - previous_inference_time,
                    )
                else:
                    frame_interval_seconds = (
                        self._inference_interval
                    )

                feature_row = (
                    self.feature_extractor.extract_features(
                        landmarks,
                        previous_landmarks=(
                            self._previous_landmarks
                        ),
                        frame_interval_seconds=(
                            frame_interval_seconds
                        ),
                    )
                )

                self._previous_landmarks = landmarks

                metrics = self.update_temporal_metrics(
                    feature_row,
                    timestamp=current_time,
                    pose_detected=True,
                    landmarks=landmarks,
                )

        with self._lock:
            if metrics is not None:
                self._latest_metrics = metrics

            self._latest_frame = annotated_frame

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24",
        )

    def update_temporal_metrics(
        self,
        frame_features: dict[str, float],
        timestamp: float,
        pose_detected: bool,
        landmarks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Update temporal history and calculate movement metrics."""
        with self._lock:
            if not pose_detected:
                metrics = {
                    "movement_speed": 0.0,
                    "smoothed_speed": 0.0,
                    "body_center_displacement": 0.0,
                    "posture_change": 0.0,
                    "body_tilt": 0.0,
                    "shoulder_symmetry": 0.0,
                    "hip_symmetry": 0.0,
                    "pose_confidence": 0.0,
                    "monitoring_status": "No person detected",
                    "risk_state": "Not available",
                    "risk_score": 0.0,
                    "reasons": [],
                }

                self._latest_metrics = metrics
                return metrics

            record = self._build_temporal_record(
                frame_features=frame_features,
                timestamp=timestamp,
                landmarks=landmarks,
            )

            self.temporal_buffer.append(record)
            self._trim_buffer()

            self._valid_pose_frames = len(
                self.temporal_buffer
            )

            if len(self.temporal_buffer) < 2:
                movement_speed = 0.0
                smoothed_speed = 0.0
                body_center_displacement = 0.0
                posture_change = 0.0

            else:
                recent_window = self.temporal_buffer[
                    -min(10, len(self.temporal_buffer)) :
                ]

                movement_speed = (
                    self._calculate_movement_speed(
                        recent_window
                    )
                )

                smoothed_speed = (
                    self._calculate_smoothed_speed(
                        movement_speed,
                        recent_window,
                    )
                )

                body_center_displacement = (
                    self._calculate_body_center_displacement(
                        recent_window
                    )
                )

                posture_change = (
                    self._calculate_posture_change(
                        recent_window
                    )
                )

            recent_window = self.temporal_buffer[
                -min(10, len(self.temporal_buffer)) :
            ]

            shoulder_symmetry = self._calculate_symmetry(
                recent_window,
                "shoulder",
            )

            hip_symmetry = self._calculate_symmetry(
                recent_window,
                "hip",
            )

            current_shoulder_symmetry = (
                self._current_symmetry(
                    record,
                    "shoulder",
                )
            )

            current_hip_symmetry = (
                self._current_symmetry(
                    record,
                    "hip",
                )
            )

            current_asymmetry = max(
                current_shoulder_symmetry,
                current_hip_symmetry,
            )

            record["movement_speed"] = movement_speed
            record["smoothed_speed"] = smoothed_speed
            record["body_center_displacement"] = (
                body_center_displacement
            )
            record["posture_change"] = posture_change

            if len(self.temporal_buffer) >= 2:
                previous_tilt = self.temporal_buffer[-2].get(
                    "body_tilt",
                    record["body_tilt"],
                )

                body_tilt_change = abs(
                    record["body_tilt"] - previous_tilt
                )

            else:
                body_tilt_change = 0.0

            baseline_speed = self._baseline_speed()

            reasons: list[str] = []
            risk_score = 0.0

            if body_tilt_change > 20.0:
                reasons.append(
                    "large body tilt change"
                )
                risk_score += 30.0

            if record["body_center_y"] > 0.65:
                reasons.append(
                    "sudden body-center vertical drop"
                )
                risk_score += 25.0

            if record["body_center_y"] > 0.55:
                reasons.append(
                    "prolonged low body position"
                )
                risk_score += 20.0

            if current_asymmetry > 0.2:
                reasons.append(
                    "large left/right asymmetry"
                )
                risk_score += 20.0

            if (
                baseline_speed > 0.0
                and smoothed_speed < baseline_speed * 0.4
            ):
                reasons.append(
                    "unusually low movement relative "
                    "to the session baseline"
                )
                risk_score += 20.0

            risk_score = float(
                min(100.0, max(0.0, risk_score))
            )

            if (
                self._valid_pose_frames
                < self.MIN_CALIBRATION_FRAMES
            ):
                monitoring_status = "Calibrating"
                risk_state = "Calibrating"
                risk_score = 0.0
                reasons = []

            elif risk_score >= 45.0:
                monitoring_status = "Possible anomaly"
                risk_state = "Possible anomaly"

            else:
                monitoring_status = "Monitoring"
                risk_state = "Monitoring"

            if (
                timestamp - self._last_history_timestamp
                >= 1.0
                or not self.history_records
            ):
                history_entry = {
                    "timestamp": timestamp,
                    "movement_speed": movement_speed,
                    "smoothed_speed": smoothed_speed,
                    "body_center_displacement": (
                        body_center_displacement
                    ),
                    "posture_change": posture_change,
                    "body_tilt": record["body_tilt"],
                    "shoulder_symmetry": (
                        shoulder_symmetry
                    ),
                    "hip_symmetry": hip_symmetry,
                    "current_asymmetry": (
                        current_asymmetry
                    ),
                    "pose_confidence": 0.95,
                    "monitoring_status": (
                        monitoring_status
                    ),
                    "risk_score": risk_score,
                    "risk_state": risk_state,
                    "reasons": list(reasons),
                }

                self.append_history_entry(
                    history_entry
                )

                self._last_history_timestamp = (
                    timestamp
                )

            metrics = {
                "movement_speed": movement_speed,
                "smoothed_speed": smoothed_speed,
                "body_center_displacement": (
                    body_center_displacement
                ),
                "posture_change": posture_change,
                "body_tilt": record["body_tilt"],
                "shoulder_symmetry": (
                    shoulder_symmetry
                ),
                "hip_symmetry": hip_symmetry,
                "current_asymmetry": (
                    current_asymmetry
                ),
                "pose_confidence": 0.95,
                "monitoring_status": monitoring_status,
                "risk_state": risk_state,
                "risk_score": risk_score,
                "reasons": reasons,
            }

            self._latest_metrics = metrics
            return metrics

    def append_history_entry(
        self,
        entry: dict[str, Any],
    ) -> None:
        """Append one summarized record."""
        with self._lock:
            self.history_records.append(entry)

            if len(self.history_records) > self.max_history:
                self.history_records = (
                    self.history_records[
                        -self.max_history :
                    ]
                )

    def _build_temporal_record(
        self,
        frame_features: dict[str, float],
        timestamp: float,
        landmarks: list[Any] | None = None,
    ) -> dict[str, float]:
        record: dict[str, float] = {
            "timestamp": float(timestamp),
            "body_center_x": float(
                frame_features.get(
                    "body_center_x",
                    0.0,
                )
            ),
            "body_center_y": float(
                frame_features.get(
                    "body_center_y",
                    0.0,
                )
            ),
            "body_tilt": float(
                frame_features.get(
                    "body_tilt_angle",
                    0.0,
                )
            ),
            "head_tilt": float(
                frame_features.get(
                    "head_tilt_angle",
                    0.0,
                )
            ),
            "shoulder_symmetry": float(
                frame_features.get(
                    "shoulder_symmetry",
                    0.0,
                )
            ),
            "hip_symmetry": float(
                frame_features.get(
                    "hip_symmetry",
                    0.0,
                )
            ),
            "knee_angle": float(
                frame_features.get(
                    "knee_angle",
                    0.0,
                )
            ),
            "stride_length": float(
                frame_features.get(
                    "stride_length",
                    0.0,
                )
            ),
        }

        if landmarks is not None:
            record.update(
                self._extract_landmark_positions(
                    landmarks
                )
            )

        return record

    def _extract_landmark_positions(
        self,
        landmarks: list[Any],
    ) -> dict[str, float]:
        positions: dict[str, float] = {}

        landmark_indexes = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }

        for name, index in landmark_indexes.items():
            if index >= len(landmarks):
                continue

            landmark = landmarks[index]

            if landmark is None:
                continue

            positions[f"{name}_x"] = float(
                getattr(landmark, "x", 0.0)
            )

            positions[f"{name}_y"] = float(
                getattr(landmark, "y", 0.0)
            )

        return positions

    def _calculate_movement_speed(
        self,
        buffer: list[dict[str, float]],
    ) -> float:
        if len(buffer) < 2:
            return 0.0

        speeds: list[float] = []

        for previous_record, current_record in zip(
            buffer[:-1],
            buffer[1:],
        ):
            dt = max(
                current_record["timestamp"]
                - previous_record["timestamp"],
                1e-3,
            )

            displacement = float(
                np.hypot(
                    current_record["body_center_x"]
                    - previous_record[
                        "body_center_x"
                    ],
                    current_record["body_center_y"]
                    - previous_record[
                        "body_center_y"
                    ],
                )
            )

            speeds.append(displacement / dt)

        if not speeds:
            return 0.0

        return float(
            sum(speeds) / len(speeds)
        )

    def _calculate_smoothed_speed(
        self,
        movement_speed: float,
        buffer: list[dict[str, float]],
    ) -> float:
        if len(buffer) < 2:
            return movement_speed

        previous_speed = float(
            buffer[-2].get(
                "smoothed_speed",
                movement_speed,
            )
        )

        return float(
            0.5 * movement_speed
            + 0.5 * previous_speed
        )

    def _calculate_body_center_displacement(
        self,
        buffer: list[dict[str, float]],
    ) -> float:
        if len(buffer) < 2:
            return 0.0

        first_record = buffer[0]
        latest_record = buffer[-1]

        return float(
            np.hypot(
                latest_record["body_center_x"]
                - first_record["body_center_x"],
                latest_record["body_center_y"]
                - first_record["body_center_y"],
            )
        )

    def _calculate_posture_change(
        self,
        buffer: list[dict[str, float]],
    ) -> float:
        if len(buffer) < 2:
            return 0.0

        changes = [
            abs(
                current_record["body_tilt"]
                - previous_record["body_tilt"]
            )
            for previous_record, current_record
            in zip(
                buffer[:-1],
                buffer[1:],
            )
        ]

        if not changes:
            return 0.0

        return float(
            sum(changes) / len(changes)
        )

    def _current_symmetry(
        self,
        record: dict[str, float],
        metric_name: str,
    ) -> float:
        if metric_name == "shoulder":
            left_x = record.get(
                "left_shoulder_x"
            )
            right_x = record.get(
                "right_shoulder_x"
            )

            if (
                left_x is not None
                and right_x is not None
            ):
                return float(
                    abs(left_x - right_x)
                )

            return float(
                abs(
                    record.get(
                        "shoulder_symmetry",
                        0.0,
                    )
                )
            )

        if metric_name == "hip":
            left_x = record.get("left_hip_x")
            right_x = record.get(
                "right_hip_x"
            )

            if (
                left_x is not None
                and right_x is not None
            ):
                return float(
                    abs(left_x - right_x)
                )

            return float(
                abs(
                    record.get(
                        "hip_symmetry",
                        0.0,
                    )
                )
            )

        return 0.0

    def _calculate_symmetry(
        self,
        buffer: list[dict[str, float]],
        metric_name: str,
    ) -> float:
        if not buffer:
            return 0.0

        values = [
            self._current_symmetry(
                record,
                metric_name,
            )
            for record in buffer
        ]

        if not values:
            return 0.0

        return float(
            sum(values) / len(values)
        )

    def _baseline_speed(self) -> float:
        with self._lock:
            if len(self.history_records) < 2:
                return 0.0

            speeds = [
                float(
                    entry.get(
                        "smoothed_speed",
                        0.0,
                    )
                )
                for entry
                in self.history_records[-5:]
            ]

            if not speeds:
                return 0.0

            return float(
                sum(speeds) / len(speeds)
            )

    def _trim_buffer(self) -> None:
        if len(self.temporal_buffer) > self.max_history:
            self.temporal_buffer = (
                self.temporal_buffer[
                    -self.max_history :
                ]
            )

    def close(self) -> None:
        """Release MediaPipe resources."""
        pose_detector = getattr(
            self,
            "pose_detector",
            None,
        )

        if pose_detector is None:
            return

        close_method = getattr(
            pose_detector,
            "close",
            None,
        )

        if callable(close_method):
            close_method()
            return

        pose_model = getattr(
            pose_detector,
            "pose",
            None,
        )

        model_close_method = getattr(
            pose_model,
            "close",
            None,
        )

        if callable(model_close_method):
            model_close_method()