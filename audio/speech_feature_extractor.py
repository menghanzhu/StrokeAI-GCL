from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np


@dataclass(frozen=True)
class SpeechFeatureConfig:
    target_sample_rate: int = 16000
    frame_length: int = 1024
    hop_length: int = 256
    top_db: float = 30.0
    minimum_pause_seconds: float = 0.20
    number_of_mfccs: int = 13


class SpeechFeatureExtractor:
    """Extract timing, energy and spectral features from speech audio."""

    def __init__(
        self,
        config: SpeechFeatureConfig | None = None,
    ) -> None:
        self.config = config or SpeechFeatureConfig()

    def extract(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Extract a fixed-length feature dictionary."""
        mono_samples = self._prepare_audio(
            samples,
            sample_rate,
        )

        if mono_samples.size == 0:
            return self.empty_features()

        target_rate = self.config.target_sample_rate
        total_duration = mono_samples.size / target_rate

        intervals = librosa.effects.split(
            mono_samples,
            top_db=self.config.top_db,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
        )

        voiced_duration = self._calculate_voiced_duration(
            intervals,
            target_rate,
        )

        speech_start_delay = self._calculate_start_delay(
            intervals,
            target_rate,
            total_duration,
        )

        speech_end_silence = self._calculate_end_silence(
            intervals,
            target_rate,
            total_duration,
        )

        pause_durations = self._calculate_pause_durations(
            intervals,
            target_rate,
        )

        meaningful_pauses = [
            pause
            for pause in pause_durations
            if pause >= self.config.minimum_pause_seconds
        ]

        rms = librosa.feature.rms(
            y=mono_samples,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
        )[0]

        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            mono_samples,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
        )[0]

        spectral_centroid = librosa.feature.spectral_centroid(
            y=mono_samples,
            sr=target_rate,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length,
        )[0]

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=mono_samples,
            sr=target_rate,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length,
        )[0]

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=mono_samples,
            sr=target_rate,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length,
        )[0]

        mfcc = librosa.feature.mfcc(
            y=mono_samples,
            sr=target_rate,
            n_mfcc=self.config.number_of_mfccs,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length,
        )

        features: dict[str, float] = {
            "total_duration": float(total_duration),
            "voiced_duration": float(voiced_duration),
            "voiced_ratio": float(
                voiced_duration / total_duration
                if total_duration > 0.0
                else 0.0
            ),
            "speech_start_delay": float(speech_start_delay),
            "speech_end_silence": float(speech_end_silence),
            "pause_count": float(len(meaningful_pauses)),
            "mean_pause_duration": float(
                np.mean(meaningful_pauses)
                if meaningful_pauses
                else 0.0
            ),
            "longest_pause": float(
                max(meaningful_pauses)
                if meaningful_pauses
                else 0.0
            ),
            "rms_mean": self._safe_mean(rms),
            "rms_std": self._safe_std(rms),
            "zero_crossing_rate_mean": self._safe_mean(
                zero_crossing_rate
            ),
            "zero_crossing_rate_std": self._safe_std(
                zero_crossing_rate
            ),
            "spectral_centroid_mean": self._safe_mean(
                spectral_centroid
            ),
            "spectral_centroid_std": self._safe_std(
                spectral_centroid
            ),
            "spectral_bandwidth_mean": self._safe_mean(
                spectral_bandwidth
            ),
            "spectral_bandwidth_std": self._safe_std(
                spectral_bandwidth
            ),
            "spectral_rolloff_mean": self._safe_mean(
                spectral_rolloff
            ),
            "spectral_rolloff_std": self._safe_std(
                spectral_rolloff
            ),
        }

        for index in range(
            self.config.number_of_mfccs
        ):
            coefficient = mfcc[index]

            features[
                f"mfcc_{index + 1}_mean"
            ] = self._safe_mean(coefficient)

            features[
                f"mfcc_{index + 1}_std"
            ] = self._safe_std(coefficient)

        return features

    def empty_features(self) -> dict[str, float]:
        """Return zero-valued features with a stable schema."""
        features: dict[str, float] = {
            "total_duration": 0.0,
            "voiced_duration": 0.0,
            "voiced_ratio": 0.0,
            "speech_start_delay": 0.0,
            "speech_end_silence": 0.0,
            "pause_count": 0.0,
            "mean_pause_duration": 0.0,
            "longest_pause": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "zero_crossing_rate_std": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_centroid_std": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "spectral_bandwidth_std": 0.0,
            "spectral_rolloff_mean": 0.0,
            "spectral_rolloff_std": 0.0,
        }

        for index in range(
            self.config.number_of_mfccs
        ):
            features[f"mfcc_{index + 1}_mean"] = 0.0
            features[f"mfcc_{index + 1}_std"] = 0.0

        return features

    def _prepare_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        audio = np.asarray(
            samples,
            dtype=np.float32,
        )

        if audio.ndim == 2:
            audio = np.mean(
                audio,
                axis=1,
            )

        if audio.ndim != 1:
            audio = audio.reshape(-1)

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if audio.size == 0:
            return audio

        if sample_rate != self.config.target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.config.target_sample_rate,
            )

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 1.0:
            audio = audio / peak

        return audio.astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _calculate_voiced_duration(
        intervals: np.ndarray,
        sample_rate: int,
    ) -> float:
        if intervals.size == 0:
            return 0.0

        voiced_samples = sum(
            int(end) - int(start)
            for start, end in intervals
        )

        return voiced_samples / sample_rate

    @staticmethod
    def _calculate_start_delay(
        intervals: np.ndarray,
        sample_rate: int,
        total_duration: float,
    ) -> float:
        if intervals.size == 0:
            return total_duration

        first_start = int(
            intervals[0][0]
        )

        return first_start / sample_rate

    @staticmethod
    def _calculate_end_silence(
        intervals: np.ndarray,
        sample_rate: int,
        total_duration: float,
    ) -> float:
        if intervals.size == 0:
            return total_duration

        final_end = int(
            intervals[-1][1]
        )

        final_time = final_end / sample_rate

        return max(
            0.0,
            total_duration - final_time,
        )

    @staticmethod
    def _calculate_pause_durations(
        intervals: np.ndarray,
        sample_rate: int,
    ) -> list[float]:
        if len(intervals) < 2:
            return []

        pause_durations: list[float] = []

        for previous_interval, current_interval in zip(
            intervals[:-1],
            intervals[1:],
        ):
            previous_end = int(
                previous_interval[1]
            )
            current_start = int(
                current_interval[0]
            )

            pause_samples = max(
                0,
                current_start - previous_end,
            )

            pause_durations.append(
                pause_samples / sample_rate
            )

        return pause_durations

    @staticmethod
    def _safe_mean(
        values: np.ndarray,
    ) -> float:
        if values.size == 0:
            return 0.0

        value = float(
            np.nanmean(values)
        )

        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _safe_std(
        values: np.ndarray,
    ) -> float:
        if values.size == 0:
            return 0.0

        value = float(
            np.nanstd(values)
        )

        return value if np.isfinite(value) else 0.0