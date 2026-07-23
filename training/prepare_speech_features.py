from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from audio.speech_feature_extractor import SpeechFeatureExtractor
from datasets.speech_dataset_loader import SpeechDatasetLoader


LOGGER = logging.getLogger("strokeai.prepare_speech_features")

DEFAULT_DATASET_DIRECTORY = Path(
    "data/speech/TORGO-database/data"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/speech/processed/torgo_speech_features.csv"
)
DEFAULT_SUMMARY_PATH = Path(
    "data/speech/processed/torgo_speech_features_summary.json"
)


def configure_logging() -> None:
    """Configure readable terminal logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract timing, energy, spectral and MFCC features "
            "from the local TORGO Parquet dataset."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
        help=(
            "Directory containing the TORGO Parquet shards."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output JSON summary path.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional row limit. Use a small value such as 100 "
            "for a quick test before processing the full dataset."
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help=(
            "Write a temporary checkpoint after this many "
            "successfully processed rows."
        ),
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help=(
            "Skip rows before this global row index. "
            "Useful for debugging."
        ),
    )

    return parser.parse_args()


def sanitize_text(value: Any) -> str:
    """Convert an optional value to safe text."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def sanitize_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(result):
        return default

    return result


def normalize_label(value: Any) -> str:
    """Normalize the TORGO speech-status label."""
    label = sanitize_text(value).lower()

    label_aliases = {
        "healthy": "healthy",
        "control": "healthy",
        "normal": "healthy",
        "dysarthria": "dysarthria",
        "dysarthric": "dysarthria",
    }

    return label_aliases.get(label, label)


def create_metadata_record(
    row: dict[str, Any],
    row_index: int,
    decoded_duration: float,
    sample_rate: int,
    sample_count: int,
) -> dict[str, Any]:
    """Build the non-acoustic portion of one output record."""
    speech_status = normalize_label(
        row.get("speech_status")
    )

    label = (
        0
        if speech_status == "healthy"
        else 1
        if speech_status == "dysarthria"
        else -1
    )

    return {
        "row_index": int(row_index),
        "transcription": sanitize_text(
            row.get("transcription")
        ),
        "speech_status": speech_status,
        "label": int(label),
        "gender": sanitize_text(
            row.get("gender")
        ).lower(),
        "reported_duration": sanitize_float(
            row.get("duration")
        ),
        "decoded_duration": sanitize_float(
            decoded_duration
        ),
        "sample_rate": int(sample_rate),
        "sample_count": int(sample_count),
    }


def validate_feature_record(
    record: dict[str, Any],
) -> None:
    """Raise an error if a numeric feature is invalid."""
    ignored_columns = {
        "transcription",
        "speech_status",
        "gender",
        "error",
    }

    for key, value in record.items():
        if key in ignored_columns:
            continue

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(
                    f"Feature {key!r} is not finite."
                )


def write_checkpoint(
    records: list[dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    """Write current records to a temporary CSV file."""
    if not records:
        return

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_frame = pd.DataFrame(records)

    checkpoint_frame.to_csv(
        checkpoint_path,
        index=False,
    )


def build_summary(
    frame: pd.DataFrame,
    total_attempted: int,
    total_failed: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a JSON-serializable preparation summary."""
    label_counts: dict[str, int] = {}

    if "speech_status" in frame.columns:
        label_counts = {
            str(key): int(value)
            for key, value in (
                frame["speech_status"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        }

    gender_counts: dict[str, int] = {}

    if "gender" in frame.columns:
        gender_counts = {
            str(key): int(value)
            for key, value in (
                frame["gender"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        }

    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
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
    ]

    return {
        "total_attempted": int(total_attempted),
        "total_successful": int(len(frame)),
        "total_failed": int(total_failed),
        "feature_count": int(
            len(feature_columns)
        ),
        "output_column_count": int(
            len(frame.columns)
        ),
        "speech_status_counts": label_counts,
        "gender_counts": gender_counts,
        "elapsed_seconds": float(elapsed_seconds),
        "feature_columns": feature_columns,
    }


def prepare_speech_features(
    dataset_directory: Path,
    output_path: Path,
    summary_output_path: Path,
    max_rows: int | None,
    checkpoint_every: int,
    start_row: int,
) -> pd.DataFrame:
    """Extract speech features and save them as CSV."""
    loader = SpeechDatasetLoader(
        dataset_directory=dataset_directory
    )
    extractor = SpeechFeatureExtractor()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = output_path.with_name(
        output_path.stem
        + "_checkpoint.csv"
    )

    LOGGER.info(
        "Loading TORGO dataset from %s",
        dataset_directory,
    )

    dataset = loader.load_dataframe(
        max_rows=max_rows,
    )

    if dataset.empty:
        raise RuntimeError(
            "The speech dataset contained no rows."
        )

    total_rows = len(dataset)

    LOGGER.info(
        "Loaded %s rows.",
        f"{total_rows:,}",
    )

    if start_row < 0:
        raise ValueError(
            "--start-row cannot be negative."
        )

    if start_row >= total_rows:
        raise ValueError(
            "--start-row is greater than or equal "
            "to the number of available rows."
        )

    records: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    selected_rows = dataset.iloc[start_row:]

    progress = tqdm(
        selected_rows.iterrows(),
        total=len(selected_rows),
        desc="Extracting speech features",
        unit="audio",
    )

    for dataframe_index, row in progress:
        global_row_index = int(
            dataframe_index
        )

        try:
            audio_value = row["audio"]

            samples, sample_rate = (
                loader.decode_audio(
                    audio_value
                )
            )

            samples_array = np.asarray(
                samples,
                dtype=np.float32,
            )

            decoded_duration = (
                len(samples_array) / sample_rate
                if sample_rate > 0
                else 0.0
            )

            features = extractor.extract(
                samples_array,
                sample_rate,
            )

            metadata = create_metadata_record(
                row=row.to_dict(),
                row_index=global_row_index,
                decoded_duration=(
                    decoded_duration
                ),
                sample_rate=sample_rate,
                sample_count=len(
                    samples_array
                ),
            )

            record = {
                **metadata,
                **features,
            }

            validate_feature_record(
                record
            )

            records.append(record)

        except Exception as exc:
            LOGGER.warning(
                "Skipping row %s because feature "
                "extraction failed: %s",
                global_row_index,
                exc,
            )

            failed_rows.append(
                {
                    "row_index": (
                        global_row_index
                    ),
                    "error": str(exc),
                }
            )

        successful_count = len(records)

        progress.set_postfix(
            successful=successful_count,
            failed=len(failed_rows),
        )

        if (
            checkpoint_every > 0
            and successful_count > 0
            and successful_count
            % checkpoint_every
            == 0
        ):
            write_checkpoint(
                records,
                checkpoint_path,
            )

            gc.collect()

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if not records:
        raise RuntimeError(
            "No speech feature rows were generated."
        )

    output_frame = pd.DataFrame(
        records
    )

    output_frame = output_frame.sort_values(
        "row_index"
    ).reset_index(drop=True)

    output_frame.to_csv(
        output_path,
        index=False,
    )

    if failed_rows:
        failure_path = output_path.with_name(
            output_path.stem
            + "_failures.csv"
        )

        pd.DataFrame(
            failed_rows
        ).to_csv(
            failure_path,
            index=False,
        )

        LOGGER.warning(
            "Saved %s failed rows to %s",
            len(failed_rows),
            failure_path,
        )

    summary = build_summary(
        frame=output_frame,
        total_attempted=len(
            selected_rows
        ),
        total_failed=len(
            failed_rows
        ),
        elapsed_seconds=(
            elapsed_seconds
        ),
    )

    summary_output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    LOGGER.info(
        "Feature extraction complete."
    )
    LOGGER.info(
        "Saved %s rows to %s",
        f"{len(output_frame):,}",
        output_path,
    )
    LOGGER.info(
        "Summary saved to %s",
        summary_output_path,
    )
    LOGGER.info(
        "Elapsed time: %.1f seconds",
        elapsed_seconds,
    )

    return output_frame


def main() -> None:
    """Run the feature-preparation command."""
    configure_logging()
    arguments = parse_arguments()

    prepare_speech_features(
        dataset_directory=(
            arguments.dataset_dir
        ),
        output_path=arguments.output,
        summary_output_path=(
            arguments.summary_output
        ),
        max_rows=arguments.max_rows,
        checkpoint_every=(
            arguments.checkpoint_every
        ),
        start_row=arguments.start_row,
    )


if __name__ == "__main__":
    main()