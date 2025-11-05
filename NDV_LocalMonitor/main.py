import os
from pathlib import Path

from .dashboard import create_app
from .sensors import SensorManager
from .camera import CameraManager
from .yolo import TFLiteYOLO
from .utils import AlertLogger, SummaryManager


LABELS = [
    "Chicken",
]


def ensure_directories() -> None:
    base = Path(__file__).resolve().parent
    for sub in ("models", "logs", "data"):
        Path(base / sub).mkdir(parents=True, exist_ok=True)


def build_app():
    ensure_directories()

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "chickenmodel.tflite"

    sensor_manager = SensorManager()
    camera_manager = CameraManager(preferred_resolution=(640, 480))
    yolo_model = TFLiteYOLO(model_path=str(model_path), labels=LABELS)
    alert_logger = AlertLogger(log_file=str(base_dir / "logs" / "alerts.log"))
    summary_manager = SummaryManager(
        summary_file=str(base_dir / "data" / "daily_summary.json")
    )

    app = create_app(
        sensor_manager=sensor_manager,
        camera_manager=camera_manager,
        yolo=yolo_model,
        alert_logger=alert_logger,
        summary_manager=summary_manager,
    )

    # Start background summary scheduler
    summary_manager.start_background_writer()

    return app


if __name__ == "__main__":
    flask_app = build_app()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    flask_app.run(host=host, port=port, debug=False, threaded=True)


