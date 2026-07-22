from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("strokeai.baseline")


class PersonalBaseline:
    """Build a personal baseline from a user's own monitoring session.

    Every frame's extracted features are appended to an in-memory rolling history.
    The baseline is computed from the latest window of the session, defaulting to
    the most recent 60 seconds. It reports mean, standard deviation, and moving
    average per feature.
    """

    def __init__(self, window_seconds: float = 60.0, storage_path: str | Path | None = None) -> None:
        self.window_seconds = window_seconds
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._history: list[dict[str, Any]] = []

    def add_frame_features(self, features: dict[str, float], *, timestamp: float | None = None) -> None:
        """Store one frame worth of feature values for the personal baseline."""
        if not features:
            return

        payload = dict(features)
        payload["timestamp"] = timestamp if timestamp is not None else len(self._history)
        self._history.append(payload)
        self._prune_history()
        self._persist()

    def get_summary(self) -> dict[str, Any]:
        """Return the rolling baseline summary for the most recent window."""
        if not self._history:
            return {"sample_count": 0, "features": {}}

        frame = pd.DataFrame(self._history)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        feature_columns = [column for column in frame.columns if column != "timestamp"]

        summary: dict[str, Any] = {
            "sample_count": len(frame),
            "window_seconds": self.window_seconds,
            "features": {},
        }

        for column in feature_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0)) if values.size > 1 else 0.0
            moving_average = float(np.mean(values))
            summary["features"][column] = {
                "mean": mean_value,
                "std": std_value,
                "moving_average": moving_average,
            }

        return summary

    def _prune_history(self) -> None:
        if len(self._history) < 2:
            return

        timestamps = [entry["timestamp"] for entry in self._history if "timestamp" in entry]
        if not timestamps:
            return

        latest_timestamp = max(timestamps)
        oldest_allowed = latest_timestamp - self.window_seconds
        self._history = [
            entry for entry in self._history
            if entry.get("timestamp", latest_timestamp) >= oldest_allowed
        ]

    def _persist(self) -> None:
        if self.storage_path is None:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window_seconds": self.window_seconds,
            "history": self._history,
        }
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
