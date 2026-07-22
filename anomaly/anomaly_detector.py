from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.config import DEFAULT_CONTAMINATION, DEFAULT_HISTORY_LIMIT, DEFAULT_RANDOM_STATE


logger = logging.getLogger("strokeai.anomaly")


class AnomalyDetector:
    """Evaluate a person's current movement behavior with simple rule-based checks."""

    def __init__(self, contamination: float = DEFAULT_CONTAMINATION, random_state: int = DEFAULT_RANDOM_STATE, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.history_limit = history_limit
        self.history: list[pd.DataFrame] = []
        self.model: IsolationForest | None = None
        self.feature_columns: list[str] | None = None
        self.last_result: dict[str, Any] | None = None

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select numeric columns and ensure the input is suitable for modeling."""
        if data.empty:
            raise ValueError("Input data cannot be empty.")

        numeric_frame = data.select_dtypes(include=["number"]).copy()
        if numeric_frame.empty:
            raise ValueError("No numeric features were found in the input data.")

        if self.feature_columns is None:
            self.feature_columns = list(numeric_frame.columns)
        else:
            missing = [column for column in self.feature_columns if column not in numeric_frame.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            numeric_frame = numeric_frame[self.feature_columns]

        return numeric_frame.fillna(0.0)

    def fit(self, baseline_data: pd.DataFrame) -> None:
        """Train the isolation forest on baseline movement data."""
        features = self._prepare_features(baseline_data)
        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.model.fit(features)

    def predict(self, current_data: pd.DataFrame) -> list[int]:
        """Predict whether each row is normal or anomalous."""
        if self.model is None:
            raise RuntimeError("The detector must be trained before prediction.")

        features = self._prepare_features(current_data)
        return self.model.predict(features).tolist()

    def analyze_behavior(self, baseline_data: pd.DataFrame, current_data: pd.DataFrame) -> dict[str, Any]:
        """Apply a lightweight rule-based anomaly check to the latest movement metrics."""
        current_values = current_data.iloc[-1].to_dict() if not current_data.empty else {}
        baseline_values = baseline_data.iloc[-1].to_dict() if not baseline_data.empty else {}

        reasons: list[str] = []
        risk_score = 0.0

        body_tilt_change = float(current_values.get("body_tilt", 0.0) - baseline_values.get("body_tilt", 0.0))
        if abs(body_tilt_change) > 20.0:
            reasons.append("large posture change")
            risk_score += 35.0

        body_center_y = float(current_values.get("body_center_y", 0.0))
        if body_center_y > 0.65:
            reasons.append("sudden drop")
            risk_score += 30.0

        movement_speed = float(current_values.get("movement_speed", 0.0))
        baseline_speed = float(baseline_values.get("smoothed_speed", 0.0))
        if baseline_speed > 0.0 and movement_speed < baseline_speed * 0.4:
            reasons.append("abnormal movement pattern")
            risk_score += 25.0

        shoulder_symmetry = float(current_values.get("shoulder_symmetry", 0.0))
        hip_symmetry = float(current_values.get("hip_symmetry", 0.0))
        if max(shoulder_symmetry, hip_symmetry) > 0.2:
            reasons.append("asymmetry")
            risk_score += 20.0

        risk_score = min(100.0, max(0.0, risk_score))
        if risk_score >= 45.0:
            monitoring_status = "Possible anomaly"
            risk_level = "Possible anomaly"
        elif risk_score > 0.0:
            monitoring_status = "Monitoring"
            risk_level = "Monitoring"
        else:
            monitoring_status = "Calibrating"
            risk_level = "Calibrating"

        if not reasons:
            reason = "No significant posture or movement change detected."
        else:
            reason = "Detected: " + ", ".join(reasons)

        self.last_result = {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "reason": reason,
            "monitoring_status": monitoring_status,
            "reasons": reasons,
        }
        logger.debug("Rule-based anomaly evaluation completed: %s", self.last_result)
        return self.last_result

    def _build_reason(self, baseline_data: pd.DataFrame, current_data: pd.DataFrame, predictions: list[int]) -> str:
        """Generate a concise explanation of the detected anomaly pattern."""
        if not predictions or all(prediction == 1 for prediction in predictions):
            return "No meaningful deviation from the baseline profile was detected."

        baseline_features = self._prepare_features(baseline_data)
        current_features = self._prepare_features(current_data)

        baseline_mean = baseline_features.mean()
        baseline_std = baseline_features.std().fillna(0.0)
        standardized = (current_features.mean() - baseline_mean) / baseline_std.replace(0.0, 1.0)

        top_features = standardized.abs().sort_values(ascending=False)
        top_names = [feature for feature in top_features.index[:3] if top_features[feature] > 1.0]

        if top_names:
            feature_summary = ", ".join(top_names)
            return f"The latest behavior deviates most in {feature_summary} compared with the baseline profile."

        return "The latest behavior shows a notable departure from the baseline profile."

    def fit_if_needed(self, feature_frame: pd.DataFrame) -> None:
        """Keep the original lightweight interface working for single-frame use."""
        if not self.history:
            self.history.append(feature_frame.copy())
            self.fit(feature_frame)

    def update_history(self, feature_frame: pd.DataFrame) -> None:
        """Store recent frames for later review."""
        self.history.append(feature_frame.copy())
        if len(self.history) > self.history_limit:
            self.history.pop(0)
