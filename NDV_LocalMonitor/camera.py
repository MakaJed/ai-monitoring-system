from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraManager:
    def __init__(self, preferred_resolution: Tuple[int, int] = (640, 480)) -> None:
        self._lock = threading.Lock()
        self._use_picamera2 = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._resolution = preferred_resolution
        self._init_camera()

    def _init_camera(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore

            picam2 = Picamera2()
            preview_config = picam2.create_preview_configuration(main={"size": self._resolution})
            picam2.configure(preview_config)
            picam2.start()
            self._picam2 = picam2
            self._use_picamera2 = True
        except Exception:
            # Fallback to OpenCV webcam if available
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
                self._cap = cap
            else:
                self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        if self._use_picamera2:
            try:
                return self._picam2.capture_array()
            except Exception:
                return None
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self._use_picamera2:
            try:
                self._picam2.stop()
            except Exception:
                pass
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


