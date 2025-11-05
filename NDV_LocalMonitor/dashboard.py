from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Generator, List, Dict, Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template


@dataclass
class Managers:
    sensor_manager: Any
    camera_manager: Any
    yolo: Any
    alert_logger: Any
    summary_manager: Any


def encode_frame_to_mjpeg_bgr(frame_bgr: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return encoded.tobytes()


def draw_detections(frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
        label = det.get("label", "")
        score = det.get("score", 0.0)
        color = (0, 255, 0) if label == "normal(eyes)" else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}" if score else label
        cv2.putText(frame_bgr, text, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def create_app(
    sensor_manager,
    camera_manager,
    yolo,
    alert_logger,
    summary_manager,
) -> Flask:
    managers = Managers(
        sensor_manager=sensor_manager,
        camera_manager=camera_manager,
        yolo=yolo,
        alert_logger=alert_logger,
        summary_manager=summary_manager,
    )

    app = Flask(__name__, static_folder="static", template_folder="templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    def video_stream_generator() -> Generator[bytes, None, None]:
        while True:
            frame_bgr = managers.camera_manager.get_frame()
            if frame_bgr is None:
                frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_bgr, "Camera Offline", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            detections = managers.yolo.predict(frame_bgr)
            if detections:
                # Log alerts for non-normal labels
                for det in detections:
                    if det.get("label") and det["label"].lower() != "normal(eyes)".lower():
                        managers.alert_logger.log_alert(
                            source="yolo",
                            message=f"Detection: {det['label']}",
                        )
                managers.summary_manager.add_detections(detections)

            draw_detections(frame_bgr, detections)

            # Optionally, overlay small thermal hotspot indicator
            thermal = managers.sensor_manager.read_thermal_frame()
            if thermal is None:
                pass
            else:
                # Normalize thermal frame and upscale to 320x240
                tmin, tmax = float(np.min(thermal)), float(np.max(thermal))
                span = max(1e-3, tmax - tmin)
                normalized = ((thermal - tmin) / span * 255.0).astype(np.uint8)
                thermal_u8 = cv2.resize(normalized, (320, 240), interpolation=cv2.INTER_CUBIC)
                thermal_color = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_INFERNO)
                # Hotspot detection relative to ambient
                ambient = managers.sensor_manager.read_environment().get("temperature_c")
                ambient_est = float(ambient) if ambient is not None else float(np.mean(thermal))
                delta = 2.0  # HOTSPOT_DELTA
                hotspot_mask = (thermal > (ambient_est + delta)).astype(np.uint8) * 255
                hotspot_mask = cv2.resize(hotspot_mask, (320, 240), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hotspots = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 8:  # ignore tiny noise
                        continue
                    x, y, w0, h0 = cv2.boundingRect(cnt)
                    cv2.rectangle(thermal_color, (x, y), (x + w0, y + h0), (0, 0, 255), 1)
                    hotspots += 1
                # Update thermal stats for summary
                managers.summary_manager.update_thermal(avg_temp=float(np.mean(thermal)), hotspots=hotspots)
                if hotspots > 0:
                    managers.alert_logger.log_alert(source="thermal", message=f"Hotspots detected: {hotspots}")
                # Place thermal preview at top-right corner
                h, w, _ = frame_bgr.shape
                th, tw, _ = thermal_color.shape
                y0, x0 = 10, w - tw - 10
                roi = frame_bgr[y0 : y0 + th, x0 : x0 + tw]
                if roi.shape[:2] == thermal_color.shape[:2]:
                    overlay = cv2.addWeighted(roi, 0.6, thermal_color, 0.4, 0)
                    frame_bgr[y0 : y0 + th, x0 : x0 + tw] = overlay

            # Update environmental summary periodically
            env = managers.sensor_manager.read_environment()
            managers.summary_manager.update_environment(env)

            jpeg = encode_frame_to_mjpeg_bgr(frame_bgr)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

            time.sleep(0.02)  # ~50 FPS cap; actual capture rate is lower

    @app.route("/video_feed")
    def video_feed():
        return Response(video_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def thermal_stream_generator() -> Generator[bytes, None, None]:
        while True:
            thermal = managers.sensor_manager.read_thermal_frame()
            if thermal is None:
                # Produce placeholder image
                placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Thermal Offline", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                frame = placeholder
            else:
                tmin, tmax = float(np.min(thermal)), float(np.max(thermal))
                span = max(1e-3, tmax - tmin)
                normalized = ((thermal - tmin) / span * 255.0).astype(np.uint8)
                thermal_u8 = cv2.resize(normalized, (320, 240), interpolation=cv2.INTER_CUBIC)
                frame = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_INFERNO)

            jpeg = encode_frame_to_mjpeg_bgr(frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.12)  # ~8 Hz for MLX90640

    @app.route("/thermal_feed")
    def thermal_feed():
        return Response(thermal_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/sensors")
    def api_sensors():
        env = managers.sensor_manager.read_environment()
        status = managers.sensor_manager.status()
        return jsonify({"environment": env, "status": status})

    @app.route("/api/alerts")
    def api_alerts():
        return jsonify({"alerts": managers.alert_logger.tail(100)})

    @app.route("/api/summary")
    def api_summary():
        return jsonify(managers.summary_manager.read_summary())

    return app


