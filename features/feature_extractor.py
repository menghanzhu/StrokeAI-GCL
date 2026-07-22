from __future__ import annotations

import math
from typing import Any

import pandas as pd

from feature_schema import get_feature_columns
from utils.config import DEFAULT_FEATURE_COLUMNS


class FeatureExtractor:
    """Transforms MediaPipe pose landmarks into posture and gait features."""

    def __init__(self) -> None:
        self.previous_landmarks: list[Any] | None = None

    def _get_landmark(self, landmarks: list[Any], index: int) -> tuple[float, float] | None:
        """Return the x/y position of a landmark if it exists."""
        if not landmarks or index >= len(landmarks):
            return None

        landmark = landmarks[index]
        return (float(getattr(landmark, "x", 0.0)), float(getattr(landmark, "y", 0.0)))

    def _get_center(self, landmarks: list[Any], indices: list[int]) -> tuple[float, float] | None:
        """Compute the average position of a set of landmarks."""
        points = [self._get_landmark(landmarks, index) for index in indices]
        valid_points = [point for point in points if point is not None]
        if not valid_points:
            return None

        mean_x = sum(point[0] for point in valid_points) / len(valid_points)
        mean_y = sum(point[1] for point in valid_points) / len(valid_points)
        return (mean_x, mean_y)

    def _distance(self, point_a: tuple[float, float] | None, point_b: tuple[float, float] | None) -> float:
        """Compute Euclidean distance between two points."""
        if point_a is None or point_b is None:
            return 0.0
        return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

    def _angle_degrees(self, point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
        """Calculate the angle at point B using points A-B-C."""
        vector_ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
        vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])

        dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
        magnitude_product = math.hypot(*vector_ba) * math.hypot(*vector_bc)
        if magnitude_product == 0:
            return 0.0

        cosine_theta = max(-1.0, min(1.0, dot_product / magnitude_product))
        return math.degrees(math.acos(cosine_theta))

    def _line_angle(self, point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        """Calculate the angle of a line relative to the positive x-axis."""
        delta_x = point_b[0] - point_a[0]
        delta_y = point_b[1] - point_a[1]
        if delta_x == 0 and delta_y == 0:
            return 0.0
        return math.degrees(math.atan2(delta_y, delta_x))

    def extract_features(
        self,
        landmarks: list[Any],
        previous_landmarks: list[Any] | None = None,
        frame_interval_seconds: float = 1 / 30,
    ) -> dict[str, float]:
        """Calculate posture and gait-related features for a single frame."""
        if not landmarks:
            return self._empty_feature_row()

        left_shoulder = self._get_landmark(landmarks, 11)
        right_shoulder = self._get_landmark(landmarks, 12)
        left_hip = self._get_landmark(landmarks, 23)
        right_hip = self._get_landmark(landmarks, 24)
        left_knee = self._get_landmark(landmarks, 25)
        right_knee = self._get_landmark(landmarks, 26)
        left_ankle = self._get_landmark(landmarks, 27)
        right_ankle = self._get_landmark(landmarks, 28)
        nose = self._get_landmark(landmarks, 0)

        shoulder_center = self._get_center(landmarks, [11, 12])
        hip_center = self._get_center(landmarks, [23, 24])
        body_center = self._get_center(landmarks, [11, 12, 23, 24, 25, 26, 27, 28])

        body_tilt_angle = self._line_angle(shoulder_center, hip_center) if shoulder_center and hip_center else 0.0
        head_tilt_angle = self._line_angle(shoulder_center, nose) if nose and shoulder_center else 0.0

        shoulder_symmetry = abs(left_shoulder[0] - right_shoulder[0]) if left_shoulder and right_shoulder else 0.0
        hip_symmetry = abs(left_hip[0] - right_hip[0]) if left_hip and right_hip else 0.0

        left_knee_angle = self._angle_degrees(left_hip, left_knee, left_ankle) if left_hip and left_knee and left_ankle else 0.0
        right_knee_angle = self._angle_degrees(right_hip, right_knee, right_ankle) if right_hip and right_knee and right_ankle else 0.0
        knee_angle = (left_knee_angle + right_knee_angle) / 2.0 if (left_knee_angle or right_knee_angle) else 0.0

        stride_length = self._distance(left_ankle, right_ankle)

        walking_speed = 0.0
        if previous_landmarks is not None:
            previous_body_center = self._get_center(previous_landmarks, [11, 12, 23, 24, 25, 26, 27, 28])
            walking_speed = self._distance(body_center, previous_body_center) / max(frame_interval_seconds, 1e-6)

        return {
            "body_center_x": body_center[0] if body_center else 0.0,
            "body_center_y": body_center[1] if body_center else 0.0,
            "body_tilt_angle": body_tilt_angle,
            "head_tilt_angle": head_tilt_angle,
            "shoulder_symmetry": shoulder_symmetry,
            "hip_symmetry": hip_symmetry,
            "knee_angle": knee_angle,
            "stride_length": stride_length,
            "walking_speed": walking_speed,
        }

    def feature_columns(self) -> list[str]:
        """Expose the canonical list of feature names for downstream modeling."""
        return get_feature_columns() if DEFAULT_FEATURE_COLUMNS != get_feature_columns() else DEFAULT_FEATURE_COLUMNS

    def extract_features_dataframe(
        self,
        frames: list[list[Any]],
        frame_interval_seconds: float = 1 / 30,
    ) -> pd.DataFrame:
        """Convert a sequence of landmark frames into a pandas DataFrame."""
        records: list[dict[str, float]] = []
        previous_landmarks: list[Any] | None = None

        for frame in frames:
            features = self.extract_features(
                frame,
                previous_landmarks=previous_landmarks,
                frame_interval_seconds=frame_interval_seconds,
            )
            records.append(features)
            previous_landmarks = frame

        return pd.DataFrame(records)

    def save_features_csv(self, features_df: pd.DataFrame, output_path: str) -> str:
        """Persist a feature DataFrame to a CSV file."""
        features_df.to_csv(output_path, index=False)
        return output_path

    def _empty_feature_row(self) -> dict[str, float]:
        """Return a zero-filled feature row for missing landmarks."""
        return {column: 0.0 for column in self.feature_columns()}
