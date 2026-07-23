from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import av
import numpy as np

from audio.speech_predictor import SpeechPredictor


logger = logging.getLogger("strokeai.realtime_audio")


class RealtimeAudioProcessor:
    """Collect microphone audio and reject silence before prediction."""

    def __init__(
        self,
        recording_seconds: float = 6.0,
        minimum_recording_seconds: float = 2.0,
        maximum_buffer_seconds: float = 12.0,
        minimum_voiced_seconds: float = 0.9,
        minimum_voiced_ratio: float = 0.12,
        minimum_rms: float = 0.003,
        minimum_peak_amplitude: float = 0.02,
    ) -> None:
        self.recording_seconds = float(recording_seconds)
        self.minimum_recording_seconds = float(
            minimum_recording_seconds
        )
        self.maximum_buffer_seconds = float(
            maximum_buffer_seconds
        )
        self.minimum_voiced_seconds = float(
            minimum_voiced_seconds
        )
        self.minimum_voiced_ratio = float(
            minimum_voiced_ratio
        )
        self.minimum_rms = float(minimum_rms)
        self.minimum_peak_amplitude = float(
            minimum_peak_amplitude
        )

        self.predictor = SpeechPredictor()
        self._lock = threading.RLock()
        self._sample_chunks: deque[np.ndarray] = deque()
        self._sample_rate: int | None = None
        self._buffer_sample_count = 0
        self._recording_active = False
        self._recording_started_at: float | None = None
        self._recording_finished_at: float | None = None
        self._latest_result: dict[str, Any] | None = None
        self._latest_error: str | None = None
        self._analysis_in_progress = False
        self._last_volume = 0.0
        self._speech_detected = False

    @property
    def recording_active(self) -> bool:
        with self._lock:
            return self._recording_active

    @property
    def analysis_in_progress(self) -> bool:
        with self._lock:
            return self._analysis_in_progress

    @property
    def latest_result(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_result is None:
                return None
            return dict(self._latest_result)

    @property
    def latest_error(self) -> str | None:
        with self._lock:
            return self._latest_error

    @property
    def current_volume(self) -> float:
        with self._lock:
            return float(self._last_volume)

    @property
    def speech_detected(self) -> bool:
        with self._lock:
            return bool(self._speech_detected)

    @property
    def recorded_duration(self) -> float:
        with self._lock:
            if self._sample_rate is None or self._sample_rate <= 0:
                return 0.0
            return float(
                self._buffer_sample_count / self._sample_rate
            )

    def start_recording(self) -> None:
        """Clear previous audio and start a new recording."""
        with self._lock:
            self._sample_chunks.clear()
            self._sample_rate = None
            self._buffer_sample_count = 0
            self._recording_active = True
            self._recording_started_at = time.monotonic()
            self._recording_finished_at = None
            self._latest_result = None
            self._latest_error = None
            self._analysis_in_progress = False
            self._last_volume = 0.0
            self._speech_detected = False

    def stop_recording(self) -> None:
        """Stop recording and analyse the buffered audio."""
        with self._lock:
            if not self._recording_active:
                return
            self._recording_active = False
            self._recording_finished_at = time.monotonic()

        self._analyse_buffer()

    def reset(self) -> None:
        """Reset all recording and prediction state."""
        with self._lock:
            self._sample_chunks.clear()
            self._sample_rate = None
            self._buffer_sample_count = 0
            self._recording_active = False
            self._recording_started_at = None
            self._recording_finished_at = None
            self._latest_result = None
            self._latest_error = None
            self._analysis_in_progress = False
            self._last_volume = 0.0
            self._speech_detected = False

    def process_frame(
        self,
        frame: av.AudioFrame,
    ) -> av.AudioFrame:
        """Receive one WebRTC audio frame and store it only while recording."""
        samples = frame.to_ndarray()
        mono_samples = self._to_mono_float32(samples)
        sample_rate = int(
            frame.sample_rate
            or self._sample_rate
            or 48000
        )
        rms = self._calculate_rms(mono_samples)

        with self._lock:
            self._last_volume = rms

            if self._recording_active:
                self._sample_rate = sample_rate
                self._sample_chunks.append(
                    mono_samples.copy()
                )
                self._buffer_sample_count += len(
                    mono_samples
                )
                self._trim_buffer_locked()

                duration = (
                    self._buffer_sample_count
                    / sample_rate
                )

                if duration >= self.recording_seconds:
                    self._recording_active = False
                    self._recording_finished_at = (
                        time.monotonic()
                    )
                    should_analyse = True
                else:
                    should_analyse = False
            else:
                should_analyse = False

        if should_analyse:
            self._analyse_buffer()

        return frame

    def _analyse_buffer(self) -> None:
        """Reject silence first, then run the trained model."""
        with self._lock:
            if self._analysis_in_progress:
                return

            if self._sample_rate is None:
                self._latest_error = (
                    "No microphone audio was received."
                )
                return

            if not self._sample_chunks:
                self._latest_error = (
                    "No audio samples were recorded."
                )
                return

            samples = np.concatenate(
                list(self._sample_chunks)
            ).astype(np.float32, copy=False)
            sample_rate = int(self._sample_rate)
            duration = len(samples) / sample_rate

            if duration < self.minimum_recording_seconds:
                self._latest_error = (
                    "The recording was too short. "
                    "Please speak for a little longer."
                )
                return

            self._analysis_in_progress = True
            self._latest_error = None

        try:
            diagnostics = self._measure_speech_activity(
                samples,
                sample_rate,
            )

            valid_speech = (
                diagnostics["voiced_duration"] >= 0.4
                and diagnostics["peak_amplitude"] >= 0.005
            )

            if not valid_speech:
                result: dict[str, Any] = {
                    "valid_speech": False,
                    "speech_detected": False,
                    "result_level": "No valid speech detected",
                    "difference_score": None,
                    "recorded_duration": float(duration),
                    "sample_rate": sample_rate,
                    **diagnostics,
                }
            else:
                result = self.predictor.predict(
                    samples,
                    sample_rate,
                )
                result.update(
                    {
                        "valid_speech": True,
                        "speech_detected": True,
                        "recorded_duration": float(duration),
                        "sample_rate": sample_rate,
                        **diagnostics,
                    }
                )

            with self._lock:
                self._latest_result = result
                self._speech_detected = valid_speech

        except Exception as exc:
            logger.exception(
                "Speech prediction failed: %s",
                exc,
            )
            with self._lock:
                self._latest_error = str(exc)

        finally:
            with self._lock:
                self._analysis_in_progress = False

    @staticmethod
    def _measure_speech_activity(
        samples: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Measure speech activity over short frames."""
        if samples.size == 0 or sample_rate <= 0:
            return {
                "voiced_duration": 0.0,
                "voiced_ratio": 0.0,
                "rms_mean": 0.0,
                "peak_amplitude": 0.0,
                "activity_threshold": 0.0,
            }

        frame_length = max(1, int(sample_rate * 0.02))
        frame_count = int(np.ceil(len(samples) / frame_length))
        padded_length = frame_count * frame_length
        padded = np.pad(
            samples,
            (0, padded_length - len(samples)),
        )
        frames = padded.reshape(frame_count, frame_length)

        frame_rms = np.sqrt(
            np.mean(
                np.square(frames, dtype=np.float64),
                axis=1,
            )
        )

        noise_floor = float(
            np.percentile(frame_rms, 20)
        )
        activity_threshold = max(
            0.003,
            noise_floor * 2.0,
        )

        voiced_mask = frame_rms >= activity_threshold
        voiced_frame_count = int(np.sum(voiced_mask))
        voiced_duration = (
            voiced_frame_count * frame_length / sample_rate
        )
        voiced_ratio = (
            voiced_frame_count / frame_count
            if frame_count
            else 0.0
        )

        return {
            "voiced_duration": float(voiced_duration),
            "voiced_ratio": float(voiced_ratio),
            "rms_mean": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            samples,
                            dtype=np.float64,
                        )
                    )
                )
            ),
            "peak_amplitude": float(
                np.max(np.abs(samples))
            ),
            "activity_threshold": float(
                activity_threshold
            ),
        }

    def _trim_buffer_locked(self) -> None:
        """Keep the in-memory buffer bounded."""
        if self._sample_rate is None:
            return

        maximum_samples = int(
            self.maximum_buffer_seconds
            * self._sample_rate
        )

        while (
            self._sample_chunks
            and self._buffer_sample_count
            > maximum_samples
        ):
            removed = self._sample_chunks.popleft()
            self._buffer_sample_count -= len(removed)

    @staticmethod
    def _to_mono_float32(
        samples: np.ndarray,
    ) -> np.ndarray:
        """Convert PyAV audio data to normalized mono float32."""
        audio = np.asarray(samples)

        if audio.ndim == 2:
            if audio.shape[0] <= 8:
                audio = np.mean(audio, axis=0)
            else:
                audio = np.mean(audio, axis=1)

        audio = audio.reshape(-1)
        original_dtype = audio.dtype
        audio = audio.astype(np.float32, copy=False)

        if np.issubdtype(original_dtype, np.integer):
            dtype_info = np.iinfo(original_dtype)
            scale = float(
                max(
                    abs(dtype_info.min),
                    dtype_info.max,
                )
            )
            if scale > 0.0:
                audio = audio / scale

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _calculate_rms(
        samples: np.ndarray,
    ) -> float:
        if samples.size == 0:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        samples,
                        dtype=np.float64,
                    )
                )
            )
        )
