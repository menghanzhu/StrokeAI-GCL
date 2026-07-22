from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.config import DEFAULT_CONTAMINATION, DEFAULT_RANDOM_STATE


logger = logging.getLogger("strokeai.baseline")


class BaselineModel:
    """A reusable baseline modeling component for a single person's weekly movement data."""

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns or []
        self.model = IsolationForest(contamination=DEFAULT_CONTAMINATION, random_state=DEFAULT_RANDOM_STATE)
        self.is_fitted = False
        self.profile: dict[str, Any] | None = None

    def load_weekly_data(self, csv_paths: list[str | Path]) -> pd.DataFrame:
        """Load and concatenate movement data from multiple CSV files."""
        if not csv_paths:
            raise ValueError("At least one CSV file is required.")

        frames: list[pd.DataFrame] = []
        for path in csv_paths:
            data = pd.read_csv(path)
            if self.feature_columns:
                missing = [column for column in self.feature_columns if column not in data.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
                data = data[self.feature_columns]
            frames.append(data)

        return pd.concat(frames, ignore_index=True)

    def calculate_profile(self, data: pd.DataFrame) -> dict[str, Any]:
        """Compute mean, standard deviation, moving average, and trend for each feature."""
        if data.empty:
            raise ValueError("Input data cannot be empty.")

        if not self.feature_columns:
            self.feature_columns = list(data.columns)

        summary: dict[str, Any] = {}
        for column in self.feature_columns:
            series = pd.to_numeric(data[column], errors="coerce").dropna()
            if series.empty:
                continue

            moving_average = series.rolling(window=3, min_periods=1).mean()
            trend = series.diff().fillna(0.0)
            std_value = float(series.std(ddof=0)) if len(series) > 1 else 0.0
            summary[column] = {
                "mean": float(series.mean()),
                "std": std_value,
                "moving_average": moving_average.tolist(),
                "trend": trend.tolist(),
            }

        self.profile = summary
        logger.debug("Baseline profile calculated for %d features", len(summary))
        return summary

    def fit(self, features: pd.DataFrame) -> None:
        """Fit an isolation forest for anomaly detection using the baseline features."""
        if len(features) < 2:
            self.is_fitted = True
            return
        self.model.fit(features)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Predict whether a sample is anomalous relative to the fitted baseline."""
        if not self.is_fitted:
            raise RuntimeError("Baseline model must be fitted before predicting.")
        if len(features) < 2:
            return [1] * len(features)
        return self.model.predict(features).tolist()

    def visualize_profile(self, output_path: str | Path | None = None) -> None:
        """Plot the baseline profile for each feature using matplotlib."""
        if self.profile is None:
            raise ValueError("No baseline profile has been calculated yet.")

        figure, axes = plt.subplots(len(self.profile), 1, figsize=(10, 3 * len(self.profile)), squeeze=False)
        axes = axes.flatten()

        for index, (feature_name, stats) in enumerate(self.profile.items()):
            axis = axes[index]
            moving_average = stats["moving_average"]
            axis.plot(moving_average, marker="o", label="Moving Average")
            axis.axhline(stats["mean"], color="red", linestyle="--", label="Mean")
            axis.set_title(f"Baseline profile for {feature_name}")
            axis.set_ylabel(feature_name)
            axis.grid(True, alpha=0.3)
            axis.legend()

        figure.tight_layout()
        if output_path is not None:
            figure.savefig(output_path, dpi=150)
        plt.close(figure)
