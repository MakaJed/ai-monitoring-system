from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np


class TFLiteYOLO:
    def __init__(self, model_path: str, labels: List[str]) -> None:
        self._labels = labels
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._input_size = (320, 320)
        self._init_interpreter(model_path)

    def _init_interpreter(self, model_path: str) -> None:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
        except Exception:
            try:
                import tensorflow.lite as tflite  # type: ignore
            except Exception:
                tflite = None

        if tflite is None:
            self._interpreter = None
            return

        try:
            self._interpreter = tflite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            ishape = self._input_details[0]["shape"]
            self._input_size = (int(ishape[2]), int(ishape[1]))
        except Exception:
            self._interpreter = None

    def predict(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._interpreter is None:
            return []

        h, w = frame_bgr.shape[:2]
        resized = self._preprocess(frame_bgr)

        try:
            self._interpreter.set_tensor(self._input_details[0]["index"], resized)
            self._interpreter.invoke()
            outputs = [
                self._interpreter.get_tensor(od["index"]) for od in self._output_details
            ]
            detections = self._postprocess(outputs, original_size=(w, h))
            return detections
        except Exception:
            return []

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = self._letterbox(frame_bgr, new_shape=self._input_size)
        # Convert BGR to RGB and normalize to [0,1]
        rgb = resized[..., ::-1].astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(rgb, axis=0)

    def _letterbox(self, image: np.ndarray, new_shape=(320, 320)) -> np.ndarray:
        h, w = image.shape[:2]
        new_w, new_h = new_shape
        scale = min(new_w / w, new_h / h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        top = (new_h - resized.shape[0]) // 2
        left = (new_w - resized.shape[1]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        return canvas

    def _postprocess(self, outputs: List[np.ndarray], original_size) -> List[Dict[str, Any]]:
        # This is a generic YOLO-like postprocess stub. Actual details depend on your model.
        # We'll try to parse common TFLite export format: boxes, scores, classes, count
        w, h = original_size
        detections: List[Dict[str, Any]] = []

        try:
            if len(outputs) >= 4:
                boxes, classes, scores, count = outputs[0], outputs[1], outputs[2], int(outputs[3][0])
            else:
                # Unknown format
                return []

            for i in range(count):
                score = float(scores[0][i])
                if score < 0.4:
                    continue
                cls_idx = int(classes[0][i])
                label = self._labels[cls_idx] if 0 <= cls_idx < len(self._labels) else str(cls_idx)
                # boxes are typically [ymin, xmin, ymax, xmax] normalized
                ymin, xmin, ymax, xmax = boxes[0][i]
                x1 = max(0, int(xmin * w))
                y1 = max(0, int(ymin * h))
                x2 = min(w - 1, int(xmax * w))
                y2 = min(h - 1, int(ymax * h))
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "score": score,
                })
        except Exception:
            return []

        return detections

try:
    import cv2  # noqa: E402
except Exception:
    # Define a minimal shim for environments without OpenCV import available at build time
    class cv2:  # type: ignore
        @staticmethod
        def resize(img, size):
            return img


