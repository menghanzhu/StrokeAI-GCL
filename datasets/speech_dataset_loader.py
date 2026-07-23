from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import soundfile as sf


logger = logging.getLogger("strokeai.speech_dataset")


class SpeechDatasetLoader:
    """Load the local Hugging Face TORGO Parquet dataset.

    The dataset stores each WAV recording inside the ``audio`` column as a
    dictionary containing raw bytes. This loader reads the Parquet shards,
    validates their schema, and exposes rows or decoded audio samples without
    requiring temporary WAV files.
    """

    REQUIRED_COLUMNS = {
        "audio",
        "transcription",
        "speech_status",
        "gender",
        "duration",
    }

    def __init__(
        self,
        dataset_directory: str | Path = (
            "data/speech/TORGO-database/data"
        ),
    ) -> None:
        self.dataset_directory = Path(dataset_directory)

    def find_parquet_files(self) -> list[Path]:
        """Return the dataset's Parquet shards in deterministic order."""
        if not self.dataset_directory.exists():
            raise FileNotFoundError(
                "Speech dataset directory not found: "
                f"{self.dataset_directory}"
            )

        parquet_files = sorted(
            self.dataset_directory.glob("*.parquet")
        )

        if not parquet_files:
            raise FileNotFoundError(
                "No Parquet files were found in: "
                f"{self.dataset_directory}"
            )

        return parquet_files

    def load_dataframe(
        self,
        columns: list[str] | None = None,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Load all Parquet shards into one DataFrame.

        Parameters
        ----------
        columns:
            Optional subset of columns to load.
        max_rows:
            Optional row limit, useful for development and tests.
        """
        parquet_files = self.find_parquet_files()

        frames: list[pd.DataFrame] = []
        remaining = max_rows

        for parquet_file in parquet_files:
            frame = pd.read_parquet(
                parquet_file,
                columns=columns,
            )

            if columns is None:
                self._validate_columns(frame)

            if remaining is not None:
                if remaining <= 0:
                    break

                frame = frame.head(remaining)
                remaining -= len(frame)

            frames.append(frame)

        if not frames:
            selected_columns = (
                columns
                if columns is not None
                else sorted(self.REQUIRED_COLUMNS)
            )
            return pd.DataFrame(columns=selected_columns)

        dataset = pd.concat(
            frames,
            ignore_index=True,
        )

        if max_rows is not None:
            dataset = dataset.head(max_rows)

        return dataset

    def iter_rows(
        self,
        max_rows: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield normalized dataset records one at a time."""
        dataset = self.load_dataframe(
            max_rows=max_rows,
        )

        for row_index, row in dataset.iterrows():
            yield {
                "row_index": int(row_index),
                "audio": row["audio"],
                "transcription": str(
                    row.get("transcription", "")
                    or ""
                ),
                "speech_status": str(
                    row.get("speech_status", "")
                    or ""
                ).strip().lower(),
                "gender": str(
                    row.get("gender", "")
                    or ""
                ).strip().lower(),
                "duration": float(
                    row.get("duration", 0.0)
                    or 0.0
                ),
            }

    def decode_audio(
        self,
        audio_value: Any,
    ) -> tuple[Any, int]:
        """Decode one audio value into samples and sample rate.

        Returns
        -------
        samples:
            A NumPy array produced by ``soundfile``.
        sample_rate:
            Audio sample rate in hertz.
        """
        audio_bytes = self._extract_audio_bytes(
            audio_value
        )

        with io.BytesIO(audio_bytes) as audio_buffer:
            samples, sample_rate = sf.read(
                audio_buffer,
                dtype="float32",
                always_2d=False,
            )

        return samples, int(sample_rate)

    def get_dataset_summary(self) -> dict[str, Any]:
        """Return high-level label and duration information."""
        dataset = self.load_dataframe(
            columns=[
                "speech_status",
                "gender",
                "duration",
            ]
        )

        status_counts = (
            dataset["speech_status"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )

        gender_counts = (
            dataset["gender"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )

        durations = pd.to_numeric(
            dataset["duration"],
            errors="coerce",
        ).dropna()

        return {
            "sample_count": int(len(dataset)),
            "speech_status_counts": {
                str(key): int(value)
                for key, value in status_counts.items()
            },
            "gender_counts": {
                str(key): int(value)
                for key, value in gender_counts.items()
            },
            "mean_duration_seconds": (
                float(durations.mean())
                if not durations.empty
                else 0.0
            ),
            "minimum_duration_seconds": (
                float(durations.min())
                if not durations.empty
                else 0.0
            ),
            "maximum_duration_seconds": (
                float(durations.max())
                if not durations.empty
                else 0.0
            ),
        }

    def _validate_columns(
        self,
        frame: pd.DataFrame,
    ) -> None:
        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(frame.columns)
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                "Speech dataset is missing required columns: "
                f"{missing_text}"
            )

    @staticmethod
    def _extract_audio_bytes(
        audio_value: Any,
    ) -> bytes:
        if isinstance(audio_value, dict):
            audio_bytes = audio_value.get("bytes")

            if isinstance(audio_bytes, bytes):
                return audio_bytes

            if isinstance(audio_bytes, bytearray):
                return bytes(audio_bytes)

            raise ValueError(
                "Audio dictionary does not contain valid bytes."
            )

        if isinstance(audio_value, bytes):
            return audio_value

        if isinstance(audio_value, bytearray):
            return bytes(audio_value)

        raise TypeError(
            "Unsupported audio value type: "
            f"{type(audio_value).__name__}"
        )