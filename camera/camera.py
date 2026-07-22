from __future__ import annotations

import logging

import cv2
import numpy as np


logger = logging.getLogger("strokeai.camera")


class CameraManager:
    """Simple wrapper around OpenCV video capture for frame acquisition."""

    def __init__(self, source: int = 0) -> None:
        self.source = source
        self.capture = cv2.VideoCapture(source)
        self.available = self.capture.isOpened()
        if not self.available:
            logger.warning("Camera source %s is unavailable; the app will continue in a degraded mode.", source)

    def read_frame(self) -> np.ndarray | None:
        if not self.available:
            return None

        success, frame = self.capture.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.available = False
