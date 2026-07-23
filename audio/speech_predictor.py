from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from audio.speech_feature_extractor import SpeechFeatureExtractor


class SpeechPredictor:
    """Load the trained TORGO model and predict speech-pattern differences."""

    def __init__(
        self,
        model_path: str | Path = (
            "models/speech/speech_model.joblib"
        ),
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Speech model not found: {self.model_path}"
            )

        package = joblib.load(self.model_path)

        self.model = package["model"]
        self.feature_columns: list[str] = list(
            package["feature_columns"]
        )
        self.selected_model_name = str(
            package.get(
                "selected_model_name",
                "unknown",
            )
        )
        self.positive_class = str(
            package.get(
                "positive_class",
                "dysarthria",
            )
        )

        self.feature_extractor = SpeechFeatureExtractor()

    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        """Predict a dysarthria-related speech-pattern score."""
        features = self.feature_extractor.extract(
            samples,
            sample_rate,
        )

        input_row = {
            column: float(features.get(column, 0.0))
            for column in self.feature_columns
        }

        frame = pd.DataFrame(
            [input_row],
            columns=self.feature_columns,
        )

        probabilities = self.model.predict_proba(frame)[0]

        classes = list(self.model.classes_)
        positive_index = classes.index(1)

        difference_probability = float(
            probabilities[positive_index]
        )

        predicted_label = int(
            self.model.predict(frame)[0]
        )

        if difference_probability >= 0.75:
            result_level = "Higher speech-pattern difference"
        elif difference_probability >= 0.45:
            result_level = "Moderate speech-pattern difference"
        else:
            result_level = "Lower speech-pattern difference"

        return {
            "predicted_label": predicted_label,
            "difference_probability": difference_probability,
            "difference_score": difference_probability * 100.0,
            "result_level": result_level,
            "model_name": self.selected_model_name,
            "positive_class": self.positive_class,
            "features": features,
            "important_notice": (
                "This model detects speech patterns related to "
                "dysarthria in the TORGO dataset. It does not "
                "diagnose stroke."
            ),
        }