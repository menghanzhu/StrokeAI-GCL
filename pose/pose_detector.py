from __future__ import annotations

import logging
from typing import Any, Iterator

import cv2
import mediapipe as mp
import numpy as np


logger = logging.getLogger("strokeai.pose")


class PoseDetector:
    """A reusable MediaPipe-based pose detector for full-body analysis."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self.pose = None
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.initialization_error: Exception | None = None

        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self.initialization_error = exc
            logger.warning("MediaPipe pose initialization failed; continuing without pose inference: %s", exc)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[Any] | None]:
        """Process a single frame and return the annotated image plus landmark objects."""
        if frame is None:
            raise ValueError("Frame cannot be None.")

        annotated_frame = frame.copy()
        if self.pose is None:
            return annotated_frame, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is not None:
            self.drawing_utils.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style(),
            )
            return annotated_frame, list(results.pose_landmarks.landmark)

        return annotated_frame, None

    def run_webcam(
        self,
        camera_index: int = 0,
        display: bool = True,
        window_name: str = "StrokeAI Pose",
    ) -> Iterator[tuple[np.ndarray, list[Any] | None]]:
        """Read frames from the webcam, detect full-body pose, and optionally display them."""
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError("Unable to open the webcam. Check camera availability.")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                annotated_frame, landmarks = self.detect(frame)

                if display:
                    cv2.imshow(window_name, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                yield annotated_frame, landmarks
        finally:
            capture.release()
            if display:
                cv2.destroyAllWindows()

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.pose is not None:
            self.pose.close()
        logger.debug("Pose detector closed.")
