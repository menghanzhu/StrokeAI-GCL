from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("strokeai.train_speech_model")

DEFAULT_FEATURE_PATH = Path(
    "data/speech/processed/torgo_speech_features.csv"
)

DEFAULT_MODEL_PATH = Path(
    "models/speech/speech_model.joblib"
)

DEFAULT_METADATA_PATH = Path(
    "models/speech/speech_model_metadata.json"
)

DEFAULT_RESULTS_PATH = Path(
    "models/speech/speech_model_results.json"
)


# These columns are identifiers, labels, text, or redundant metadata.
# They must not be passed to the classifier.
EXCLUDED_COLUMNS = {
    "row_index",
    "transcription",
    "speech_status",
    "label",
    "gender",
    "reported_duration",
    "decoded_duration",
    "sample_rate",
    "sample_count",
}


def configure_logging() -> None:
    """Configure terminal logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare speech classifiers using "
            "the extracted TORGO feature CSV."
        )
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to the speech feature CSV.",
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path used to save the selected model.",
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path used to save model metadata.",
    )

    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path used to save evaluation results.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of rows used for testing.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def load_training_data(
    feature_path: Path,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load and validate the extracted feature table."""
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {feature_path}"
        )

    frame = pd.read_csv(feature_path)

    if len(frame) < 100:
        raise ValueError(
            "The feature CSV contains fewer than 100 rows. "
            "It may still be the small development test file. "
            "Generate the full TORGO feature CSV first."
        )

    if "label" not in frame.columns:
        raise ValueError(
            "The feature CSV does not contain the required "
            "'label' column."
        )

    labels = pd.to_numeric(
        frame["label"],
        errors="coerce",
    )

    valid_mask = labels.isin([0, 1])

    frame = frame.loc[valid_mask].copy()
    labels = labels.loc[valid_mask].astype(int)

    feature_columns = [
        column
        for column in frame.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric speech feature columns were found."
        )

    features = frame[feature_columns].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    LOGGER.info(
        "Loaded %s valid rows and %s feature columns.",
        f"{len(features):,}",
        len(feature_columns),
    )

    LOGGER.info(
        "Label counts: %s",
        labels.value_counts().sort_index().to_dict(),
    )

    return features, labels, feature_columns


def build_candidate_models(
    random_state: int,
) -> dict[str, Pipeline]:
    """Create the models that will be compared."""
    logistic_regression = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def evaluate_model(
    model: ClassifierMixin,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, Any]:
    """Calculate classification metrics."""
    predictions = model.predict(features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        roc_auc = float(
            roc_auc_score(labels, probabilities)
        )
    else:
        probabilities = None
        roc_auc = 0.0

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "healthy",
            "dysarthria",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def print_results(
    model_name: str,
    results: dict[str, Any],
) -> None:
    """Print readable model results."""
    matrix = results["confusion_matrix"]

    print("\n" + "=" * 64)
    print(f"Model: {model_name}")
    print("=" * 64)

    print(
        f"Accuracy:  {results['accuracy']:.4f}"
    )
    print(
        f"Precision: {results['precision']:.4f}"
    )
    print(
        f"Recall:    {results['recall']:.4f}"
    )
    print(
        f"F1 score:  {results['f1']:.4f}"
    )
    print(
        f"ROC-AUC:   {results['roc_auc']:.4f}"
    )

    print("\nConfusion matrix:")
    print("                 Predicted")
    print("                 Healthy  Dysarthria")
    print(
        "Actual Healthy   "
        f"{matrix[0][0]:7d}  {matrix[0][1]:10d}"
    )
    print(
        "Actual Dysarthria "
        f"{matrix[1][0]:7d}  {matrix[1][1]:10d}"
    )


def select_best_model(
    results: dict[str, dict[str, Any]],
) -> str:
    """Select the model using ROC-AUC, then recall and F1."""
    return max(
        results,
        key=lambda name: (
            results[name]["roc_auc"],
            results[name]["recall"],
            results[name]["f1"],
        ),
    )


def save_json(
    value: dict[str, Any],
    path: Path,
) -> None:
    """Save JSON with consistent formatting."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def train_speech_model(
    feature_path: Path,
    model_output_path: Path,
    metadata_output_path: Path,
    results_output_path: Path,
    test_size: float,
    random_state: int,
) -> None:
    """Train, compare, refit and save the speech model."""
    features, labels, feature_columns = (
        load_training_data(feature_path)
    )

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    LOGGER.info(
        "Training rows: %s",
        f"{len(train_features):,}",
    )
    LOGGER.info(
        "Testing rows: %s",
        f"{len(test_features):,}",
    )

    candidate_models = build_candidate_models(
        random_state=random_state
    )

    trained_models: dict[str, Pipeline] = {}
    evaluation_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_name, model in candidate_models.items():
        LOGGER.info(
            "Training %s...",
            model_name,
        )

        model.fit(
            train_features,
            train_labels,
        )

        results = evaluate_model(
            model,
            test_features,
            test_labels,
        )

        trained_models[model_name] = model
        evaluation_results[model_name] = results

        print_results(
            model_name,
            results,
        )

    selected_model_name = select_best_model(
        evaluation_results
    )

    LOGGER.info(
        "Selected model: %s",
        selected_model_name,
    )

    # Refit the selected model using all available rows.
    final_model = build_candidate_models(
        random_state=random_state
    )[selected_model_name]

    final_model.fit(
        features,
        labels,
    )

    model_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": final_model,
        "feature_columns": feature_columns,
        "positive_class": "dysarthria",
        "negative_class": "healthy",
        "label_mapping": {
            "healthy": 0,
            "dysarthria": 1,
        },
        "selected_model_name": selected_model_name,
    }

    joblib.dump(
        model_package,
        model_output_path,
    )

    metadata = {
        "model_name": selected_model_name,
        "training_row_count": int(len(features)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "label_mapping": {
            "healthy": 0,
            "dysarthria": 1,
        },
        "selected_test_results": (
            evaluation_results[
                selected_model_name
            ]
        ),
        "important_limitations": [
            (
                "The model distinguishes TORGO healthy and "
                "dysarthric speech patterns; it does not "
                "diagnose stroke."
            ),
            (
                "The current dataset columns do not include "
                "speaker identifiers, so this is an "
                "utterance-level split. The same speaker may "
                "appear in both training and test data."
            ),
            (
                "Reported performance may therefore be higher "
                "than performance on completely unseen speakers."
            ),
        ],
    }

    save_json(
        metadata,
        metadata_output_path,
    )

    all_results = {
        "selected_model": selected_model_name,
        "models": evaluation_results,
    }

    save_json(
        all_results,
        results_output_path,
    )

    print("\n" + "=" * 64)
    print("Training complete")
    print("=" * 64)
    print(
        f"Selected model: {selected_model_name}"
    )
    print(
        f"Saved model: {model_output_path}"
    )
    print(
        f"Saved metadata: {metadata_output_path}"
    )
    print(
        f"Saved results: {results_output_path}"
    )


def main() -> None:
    """Run model training."""
    configure_logging()
    arguments = parse_arguments()

    train_speech_model(
        feature_path=arguments.features,
        model_output_path=(
            arguments.model_output
        ),
        metadata_output_path=(
            arguments.metadata_output
        ),
        results_output_path=(
            arguments.results_output
        ),
        test_size=arguments.test_size,
        random_state=arguments.random_state,
    )


if __name__ == "__main__":
    main()