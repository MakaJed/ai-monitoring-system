import os
import sys
import time
import json
import math
import threading
from datetime import datetime, timedelta
from collections import deque
import sqlite3
import queue

from flask import Flask, Response, jsonify, render_template, stream_with_context, request


# Optional dependencies and hardware-specific libraries
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # OpenCV is optional on non-Pi dev machines

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
    from io import BytesIO
except Exception:  # pragma: no cover
    Image = None
    BytesIO = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# TensorFlow Lite runtime (model is optional until deployed on Pi)
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:  # pragma: no cover
    Interpreter = None

# Optional plotting for daily PNG summaries
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None
# Adafruit sensor libs (available on Pi)
try:
    import board  # type: ignore
    import busio  # type: ignore
except Exception:  # pragma: no cover
    board = None
    busio = None

try:
    import adafruit_bme680  # type: ignore
except Exception:  # pragma: no cover
    adafruit_bme680 = None

try:
    import adafruit_mlx90640  # type: ignore
except Exception:  # pragma: no cover
    adafruit_mlx90640 = None

# Try legacy picamera (for legacy camera stack) first, then picamera2 (for libcamera)
try:
    import picamera  # type: ignore  # Legacy camera
    from picamera import PiCamera  # type: ignore
    print("[Import] Legacy picamera library imported successfully")
except ImportError as ie:
    print(f"[Import] Legacy picamera import failed: {ie}")
    picamera = None  # type: ignore
    PiCamera = None  # type: ignore
except Exception as e:
    print(f"[Import] Legacy picamera import error: {e}")
    picamera = None  # type: ignore
    PiCamera = None  # type: ignore

try:
    from picamera2 import Picamera2  # type: ignore  # New libcamera
    print("[Import] Picamera2 library imported successfully")
except ImportError as ie:
    print(f"[Import] Picamera2 import failed: {ie}")
    print("[Import] Try: pip3 install picamera2  or  sudo apt install python3-picamera2")
    Picamera2 = None
except Exception as e:
    print(f"[Import] Picamera2 import error: {e}")
    Picamera2 = None


app = Flask(__name__, template_folder="templates", static_folder="static")
# Reduce static caching so updated UI/JS is always pulled after deploy
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.context_processor
def _inject_cache_bust():
    try:
        return {"cache_bust": int(time.time())}
    except Exception:
        return {"cache_bust": 0}

@app.after_request
def _no_cache(response):  # type: ignore[override]
    try:
        # For HTML/JSON, prevent browser from using stale cache after updates
        ct = response.headers.get("Content-Type", "")
        if any(x in ct for x in ("text/html", "application/json")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
    except Exception:
        pass
    return response
def add_alert(text: str) -> None:
    try:
        ts = datetime.utcnow().isoformat()
        with alert_seq_lock:
            global alert_seq
            alert_seq += 1
            aid = alert_seq
        payload = {"id": aid, "timestamp": ts, "text": text, "read": False}
        with alerts_lock:
            alerts_queue.append(text)
            alerts_log.append(payload)
        # update overlay banner
        global last_alert_time, last_alert_message
        last_alert_time = time.time()
        last_alert_message = text
    except Exception:
        pass



# Paths and configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LOG_DIR = "/home/pi/NDV_Monitoring/logs"
LOG_DIR = DEFAULT_LOG_DIR if os.path.isdir("/home/pi") else os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE = os.getenv("MODEL_FILE", "chickenmodel.tflite")
LABELS_FILE = os.getenv("LABELS_FILE", "chicken.labels.txt")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)
LABELS_PATH = os.path.join(MODELS_DIR, LABELS_FILE)

# Data directory for images and SQLite
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "logs")
DB_PATH = os.path.join(DATA_DIR, "logs.db")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
SUMMARIES_DIR = os.path.join(DATA_DIR, "summaries")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)


# Global state
camera_lock = threading.Lock()
camera_capture = None  # type: ignore
picam2 = None  # type: ignore
camera_provider = None  # type: ignore
latest_frame_bgr = None  # type: ignore

thermal_lock = threading.Lock()
thermal_sensor = None  # type: ignore
thermal_shape = (24, 32)  # MLX90640 default

bme680_lock = threading.Lock()
bme680_sensor = None  # type: ignore

# Shared I2C bus for sensors
i2c_lock = threading.Lock()
shared_i2c = None  # type: ignore

detector_lock = threading.Lock()
tflite_interpreter = None  # type: ignore
model_input_shape = None
labels = []

alerts_lock = threading.Lock()
alerts_queue = deque(maxlen=200)  # recent alerts kept in memory
# Structured alerts log (id, ts, text, read)
alerts_log = deque(maxlen=500)
alert_seq_lock = threading.Lock()
alert_seq = 0

# Unified stream sizing
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

# Detection and alert configuration
HOTSPOT_DELTA_C = 3.0  # hotspot threshold above ambient average
CALIBRATION_OFFSET_X = 0  # pixels to shift boxes horizontally (placeholder)
CALIBRATION_OFFSET_Y = 0  # pixels to shift boxes vertically (placeholder)
OVERLAY_ALERT_SECONDS = 5.0
DETECTION_OVERLAY_SECONDS = 3.0

# Recent alert state
last_alert_time = 0.0
last_alert_message = ""

# Sensor logging throttle
last_sensor_log_time = 0.0

# Camera provider state
camera_provider = None  # 'cv2' | 'picamera2' | None
picam2 = None

# Latest detections for overlay
latest_detections_lock = threading.Lock()
latest_detections = []  # list of dicts: {x1,y1,x2,y2,score,class}
latest_detections_ts = 0.0

# Symptoms model globals
SYMPTOMS_MODEL_FILE = os.getenv("SYMPTOMS_MODEL_FILE", "ndvsymptoms.tflite")
SYMPTOMS_LABELS_FILE = os.getenv("SYMPTOMS_LABELS_FILE", "symptoms.labels.txt")
SYMPTOMS_MODEL_PATH = os.path.join(MODELS_DIR, SYMPTOMS_MODEL_FILE)
SYMPTOMS_LABELS_PATH = os.path.join(MODELS_DIR, SYMPTOMS_LABELS_FILE)
sym_interpreter = None
sym_input_shape = None
sym_labels = []

# Symptoms worker state
sym_task_queue = queue.Queue(maxsize=100)
sym_results_lock = threading.Lock()
latest_symptom_overlays = []  # list of {x1,y1,x2,y2,score,label}
latest_symptoms_ts = 0.0
last_symptom_alert_ts = {}

# Settings cache (overrides env at runtime)
settings_lock = threading.Lock()
settings_cache = {}

# Camera configuration via environment
CAMERA_PROVIDER = os.getenv("CAMERA_PROVIDER", "picamera2").lower()  # picamera2|v4l2
CAMERA_DEVICE = os.getenv("CAMERA_DEVICE", "0")  # e.g., "0" or "/dev/video0"

# ------------------------------
# SQLite logging helpers
# ------------------------------

def init_db() -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT NOT NULL,
              bme_temp REAL,
              bme_humidity REAL,
              bme_pressure REAL,
              bme_gas REAL,
              thermal_hotspot_temp REAL,
              ambient_temp REAL,
              symptoms_detected TEXT,
              has_abnormality INTEGER,
              image_path_visible TEXT,
              image_path_thermal TEXT,
              notes TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              period TEXT NOT NULL,
              period_label TEXT NOT NULL,
              created_at TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_settings() -> dict:
    global settings_cache
    try:
        if os.path.isfile(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}
    with settings_lock:
        settings_cache = data
    return data


def save_settings(new_settings: dict) -> None:
    with settings_lock:
        settings_cache.update(new_settings or {})
        data = settings_cache.copy()
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_setting(key: str, default: str) -> str:
    with settings_lock:
        if key in settings_cache:
            return str(settings_cache[key])
    return os.getenv(key, default)


def insert_log(row: dict) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO logs (
              timestamp, bme_temp, bme_humidity, bme_pressure, bme_gas,
              thermal_hotspot_temp, ambient_temp, symptoms_detected, has_abnormality,
              image_path_visible, image_path_thermal, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("timestamp"),
                row.get("bme_temp"),
                row.get("bme_humidity"),
                row.get("bme_pressure"),
                row.get("bme_gas"),
                row.get("thermal_hotspot_temp"),
                row.get("ambient_temp"),
                row.get("symptoms_detected"),
                1 if row.get("has_abnormality") else 0,
                row.get("image_path_visible"),
                row.get("image_path_thermal"),
                row.get("notes"),
            ),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def is_abnormal(bme: dict | None, thermal: dict | None, symptoms: list[str]) -> tuple[bool, list[str]]:
    reasons = []
    if bme:
        t = bme.get("temperature_c")
        if t is not None and (t > 35 or t < 20):
            reasons.append("Temperature out of range")
        h = bme.get("humidity_pct")
        if h is not None and (h < 30 or h > 80):
            reasons.append("Humidity out of range")
        g = bme.get("gas_ohms")
        if g is not None and g < 1000:
            reasons.append("Poor air quality")
        p = bme.get("pressure_hpa")
        if p is not None and p < 990:
            reasons.append("Low pressure")
    if thermal:
        amb = thermal.get("ambient")
        hot = thermal.get("hotspot")
        if amb is not None and hot is not None and (hot > amb + 6):
            reasons.append("Thermal hotspot")
    if symptoms:
        reasons.append("NDV symptom(s)")
    return (len(reasons) > 0, reasons)


def save_images_and_build_paths(frame_bgr: "np.ndarray | None", thermal_colored: "np.ndarray | None", ts_iso: str) -> tuple[str | None, str | None]:
    if cv2 is None:
        return None, None
    date_dir = ts_iso.split("T")[0]
    out_dir = os.path.join(DATA_DIR, date_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    vis_path = None
    th_path = None
    try:
        if frame_bgr is not None:
            vp = os.path.join(out_dir, f"{ts_iso.replace(':', '-')}_visible.jpg")
            cv2.imwrite(vp, frame_bgr)
            vis_path = vp
    except Exception:
        vis_path = None
    try:
        if thermal_colored is not None:
            tp = os.path.join(out_dir, f"{ts_iso.replace(':', '-')}_thermal.jpg")
            cv2.imwrite(tp, thermal_colored)
            th_path = tp
    except Exception:
        th_path = None
    return vis_path, th_path


def cleanup_normal_logs_loop() -> None:
    # Configurable retention (reloaded each cycle from settings)
    normal_hours = float(get_setting("NORMAL_LOG_RETENTION_HOURS", "2"))
    abnormal_days = float(get_setting("ABNORMAL_LOG_RETENTION_DAYS", "0"))  # 0 = keep forever
    vacuum_every_hours = float(get_setting("LOG_VACUUM_EVERY_HOURS", "24"))
    last_vacuum = 0.0
    while True:
        try:
            # reload settings periodically
            try:
                load_settings()
                normal_hours = float(get_setting("NORMAL_LOG_RETENTION_HOURS", str(normal_hours)))
                abnormal_days = float(get_setting("ABNORMAL_LOG_RETENTION_DAYS", str(abnormal_days)))
                vacuum_every_hours = float(get_setting("LOG_VACUUM_EVERY_HOURS", str(vacuum_every_hours)))
            except Exception:
                pass
            now_ts = datetime.utcnow().timestamp()
            # Normal logs cutoff
            normal_cutoff_iso = datetime.utcfromtimestamp(now_ts - normal_hours * 3600).isoformat()
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            # Delete normal logs (and images)
            cur.execute(
                "SELECT id, image_path_visible, image_path_thermal FROM logs WHERE has_abnormality=0 AND timestamp < ?",
                (normal_cutoff_iso,),
            )
            rows = cur.fetchall()
            ids = [r[0] for r in rows]
            for _, vp, tp in rows:
                for p in (vp, tp):
                    if p and os.path.isfile(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            if ids:
                qmarks = ",".join(["?"] * len(ids))
                cur.execute(f"DELETE FROM logs WHERE id IN ({qmarks})", ids)
                conn.commit()

            # Optional: trim very old abnormal logs if a retention is set
            if abnormal_days > 0:
                abnormal_cutoff_iso = datetime.utcfromtimestamp(now_ts - abnormal_days * 86400).isoformat()
                cur.execute(
                    "SELECT id, image_path_visible, image_path_thermal FROM logs WHERE has_abnormality=1 AND timestamp < ?",
                    (abnormal_cutoff_iso,),
                )
                rows = cur.fetchall()
                ids = [r[0] for r in rows]
                for _, vp, tp in rows:
                    for p in (vp, tp):
                        if p and os.path.isfile(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                if ids:
                    qmarks = ",".join(["?"] * len(ids))
                    cur.execute(f"DELETE FROM logs WHERE id IN ({qmarks})", ids)
                    conn.commit()

            # Periodic VACUUM to compact DB
            if now_ts - last_vacuum >= vacuum_every_hours * 3600:
                try:
                    cur.execute("VACUUM")
                    conn.commit()
                except Exception:
                    pass
                last_vacuum = now_ts
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        # Sleep until next cycle (use the configured normal retention granularity)
        time.sleep(max(600.0, min(7200.0, normal_hours * 1800)))


def load_labels() -> None:
    global labels
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
    except Exception:
        labels = []


def load_symptoms_labels():
    global sym_labels
    try:
        with open(SYMPTOMS_LABELS_PATH, "r", encoding="utf-8") as f:
            sym_labels = [l.strip() for l in f if l.strip()]
    except Exception:
        sym_labels = []


def initialize_symptoms_detector():
    global sym_interpreter, sym_input_shape
    if Interpreter is None or not os.path.exists(SYMPTOMS_MODEL_PATH):
        return
    try:
        sym_interpreter = Interpreter(model_path=SYMPTOMS_MODEL_PATH, num_threads=2)
        sym_interpreter.allocate_tensors()
        sym_input_shape = sym_interpreter.get_input_details()[0]["shape"]
    except Exception:
        sym_interpreter = None
        sym_input_shape = None


def run_symptoms_on_roi(roi_bgr: "np.ndarray") -> list:
    SYM_CONF = float(get_setting("SYMPTOMS_CONF", "0.35"))
    if sym_interpreter is None or sym_input_shape is None or cv2 is None or np is None:
        return []
    try:
        _, in_h, in_w, _ = sym_input_shape
        resized = cv2.resize(roi_bgr, (in_w, in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_details = sym_interpreter.get_input_details()
        dtype = input_details[0]["dtype"]
        if dtype == np.float32:
            input_data = (rgb.astype(np.float32) / 255.0)[None, ...]
        else:
            input_data = rgb.astype(np.uint8)[None, ...]
        sym_interpreter.set_tensor(input_details[0]["index"], input_data)
        sym_interpreter.invoke()
        out_detail = sym_interpreter.get_output_details()[0]
        arr = sym_interpreter.get_tensor(out_detail["index"])[0]
        # Normalize layout to (anchors, C)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[-1], arr.shape[-2]) if arr.shape[1] < arr.shape[2] else arr.reshape(arr.shape[1], arr.shape[2])
        if arr.ndim != 2 or arr.shape[1] < 5:
            return []
        boxes = arr[:, :4]
        cls_logits = arr[:, 4:]
        conf = 1.0 / (1.0 + np.exp(-cls_logits))
        scores = conf.max(axis=1)
        clses = conf.argmax(axis=1)
        keep = scores >= SYM_CONF
        if not np.any(keep):
            return []
        boxes = boxes[keep]; scores = scores[keep]; clses = clses[keep]
        # Assume normalized boxes
        cx = boxes[:, 0] * in_w; cy = boxes[:, 1] * in_h; w = boxes[:, 2] * in_w; h = boxes[:, 3] * in_h
        x1 = np.clip((cx - w/2).astype(np.int32), 0, in_w - 1)
        y1 = np.clip((cy - h/2).astype(np.int32), 0, in_h - 1)
        x2 = np.clip((cx + w/2).astype(np.int32), 0, in_w - 1)
        y2 = np.clip((cy + h/2).astype(np.int32), 0, in_h - 1)
        # Map to ROI size
        roi_h, roi_w = roi_bgr.shape[:2]
        sx = roi_w / float(in_w); sy = roi_h / float(in_h)
        x1 = (x1 * sx).astype(np.int32); y1 = (y1 * sy).astype(np.int32)
        x2 = (x2 * sx).astype(np.int32); y2 = (y2 * sy).astype(np.int32)
        return [{"x1":int(x1[i]),"y1":int(y1[i]),"x2":int(x2[i]),"y2":int(y2[i]),"score":float(scores[i]),"class":int(clses[i])} for i in range(len(scores))]
    except Exception:
        return []


def symptoms_worker_loop():
    DEBOUNCE_SEC = float(os.getenv("SYMPTOMS_DEBOUNCE", "10"))
    while True:
        try:
            job = sym_task_queue.get()
            if job is None:
                continue
            frame_bgr, x1c, y1c, x2c, y2c = job
            roi = frame_bgr[y1c:y2c+1, x1c:x2c+1].copy()
            if roi.size == 0:
                continue
            dets = run_symptoms_on_roi(roi)
            overlays = []
            now_ts = time.time()
            for d in dets:
                gx1 = x1c + d["x1"]; gy1 = y1c + d["y1"]
                gx2 = x2c + d["x2"]; gy2 = y2c + d["y2"]
                label_idx = d["class"]
                label = sym_labels[label_idx] if 0 <= label_idx < len(sym_labels) else f"sym{label_idx}"
                overlays.append({"x1":gx1,"y1":gy1,"x2":gx2,"y2":gy2,"score":d["score"],"label":label})
                # Debounced alert key by label
                last_ts = last_symptom_alert_ts.get(label, 0.0)
                if now_ts - last_ts >= DEBOUNCE_SEC:
                    add_alert(f"Symptom detected: {label}")
                    last_symptom_alert_ts[label] = now_ts
            if overlays:
                with sym_results_lock:
                    global latest_symptom_overlays, latest_symptoms_ts
                    latest_symptom_overlays = overlays
                    latest_symptoms_ts = now_ts
        except Exception:
            time.sleep(0.1)


def _open_v4l2_device(dev):
    if cv2 is None:
        return None
    try:
        cap = cv2.VideoCapture(dev if isinstance(dev, int) else str(dev), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        ok, test = cap.read()
        if not ok or test is None:
            cap.release()
            return None
        return cap
    except Exception:
        return None


def initialize_camera_if_available() -> None:
    """Simple camera initialization - Picamera2 first, then OpenCV fallback."""
    global camera_capture, camera_provider, picam2
    
    # Reset state
    with camera_lock:
        camera_capture = None
        picam2 = None
        camera_provider = None
    
    # Prefer Picamera2 (matches working snippet - uses RGB888 format)
    if Picamera2 is not None and cv2 is not None:
        try:
            cam = Picamera2()
            config = cam.create_preview_configuration(main={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "RGB888"})
            cam.configure(config)
            cam.start()
            time.sleep(1.0)
            with camera_lock:
                picam2 = cam
                camera_provider = 'picamera2'
            print("[Camera] OK Picamera2 initialized successfully")
            return
        except Exception as e:
            print(f"[Camera] Picamera2 init failed: {e}")
            with camera_lock:
                picam2 = None
                camera_provider = None
    
    # Fallback: simple OpenCV VideoCapture(0)
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
                ok, test = cap.read()
                if ok and test is not None:
                    with camera_lock:
                        camera_capture = cap
                        camera_provider = 'cv2'
                    print("[Camera] OK OpenCV VideoCapture initialized successfully")
                    return
                cap.release()
        except Exception as e:
            print(f"[Camera] OpenCV init failed: {e}")
    
    print("[Camera] FAIL No camera available")


def read_camera_frame() -> tuple[bool, "np.ndarray | None"]:
    """Grab a single frame from the active camera."""
    if np is None or cv2 is None:
        return False, None

    with camera_lock:
        provider = camera_provider

    if provider == 'picamera2':
        with camera_lock:
            cam = picam2
        if cam is None:
            return False, None
        try:
            rgb = cam.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if frame.shape[1] != STREAM_WIDTH or frame.shape[0] != STREAM_HEIGHT:
                frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            return True, frame
        except Exception:
            return False, None

    if provider == 'cv2':
        with camera_lock:
            cap = camera_capture
        if cap is None:
            return False, None
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None
        try:
            if frame.shape[1] != STREAM_WIDTH or frame.shape[0] != STREAM_HEIGHT:
                frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
        except Exception:
            pass
        return True, frame

    return False, None


def release_camera() -> None:
    if cv2 is None:
        return
    with camera_lock:
        global camera_capture, picam2
        try:
            if camera_capture is not None:
                camera_capture.release()
            if picam2 is not None:
                try:
                    picam2.stop()
                except Exception:
                    pass
        finally:
            camera_capture = None
            picam2 = None


def _get_or_create_i2c():
    """Get or create shared I2C bus instance."""
    global shared_i2c
    if board is None or busio is None:
        return None
    with i2c_lock:
        if shared_i2c is None:
            try:
                print("[I2C] Creating shared I2C bus...")
                shared_i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
                print("[I2C] OK Shared I2C bus created")
            except Exception as e:
                print(f"[I2C] FAIL Could not create I2C bus: {e}")
                import traceback
                traceback.print_exc()
                return None
        return shared_i2c


def _init_thermal_raw_i2c(i2c):
    """Try to initialize thermal sensor with raw I2C access, bypassing calibration checks."""
    # This is a workaround for sensors with calibration issues
    # Note: The adafruit library doesn't allow bypassing the calibration check
    # So this function currently just returns None - the workaround would require
    # modifying the library or using raw I2C commands
    print("[Thermal] Raw I2C workaround not yet implemented - requires library modification")
    return None


def _init_thermal_sensor_with_timeout(i2c, timeout_seconds=10):
    """Initialize thermal sensor with timeout to prevent hanging."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Thermal sensor initialization timed out")
    
    # Set up timeout (only works on Unix)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            sensor = adafruit_mlx90640.MLX90640(i2c)
            signal.alarm(0)  # Cancel timeout
            signal.signal(signal.SIGALRM, old_handler)
            return sensor
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            signal.signal(signal.SIGALRM, old_handler)
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            signal.signal(signal.SIGALRM, old_handler)
            raise e
    else:
        # Windows or no SIGALRM - use threading timeout instead
        import threading
        result = [None]
        exception = [None]
        
        def init_sensor():
            try:
                result[0] = adafruit_mlx90640.MLX90640(i2c)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=init_sensor)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running - timeout
            raise TimeoutError("Thermal sensor initialization timed out")
        if exception[0]:
            raise exception[0]
        return result[0]


def initialize_thermal_if_available() -> None:
    global thermal_sensor
    print("[Thermal] Initializing MLX90640 thermal sensor...")
    
    if adafruit_mlx90640 is None:
        print("[Thermal] FAIL adafruit_mlx90640 module not available")
        return
    if board is None:
        print("[Thermal] FAIL board module not available")
        return
    if busio is None:
        print("[Thermal] FAIL busio module not available")
        return
    
    i2c = _get_or_create_i2c()
    if i2c is None:
        print("[Thermal] FAIL Could not get I2C bus")
        return
    
    try:
        with thermal_lock:
            if thermal_sensor is None:
                print("[Thermal] Creating MLX90640 sensor instance...")
                # Try multiple times - sometimes the sensor needs a few attempts
                for attempt in range(3):  # Reduced attempts since we have timeout
                    try:
                        # Try to create sensor instance with timeout
                        print(f"[Thermal] Attempt {attempt+1}/3 (with 10s timeout)...")
                        thermal_sensor = _init_thermal_sensor_with_timeout(i2c, timeout_seconds=10)
                        # 8 Hz refresh
                        thermal_sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                        print("[Thermal] Testing frame read...")
                        # Test reading a frame with retries
                        test_frame = [0.0] * (thermal_shape[0] * thermal_shape[1])
                        for read_attempt in range(3):
                            try:
                                thermal_sensor.getFrame(test_frame)
                                min_temp = min(test_frame)
                                max_temp = max(test_frame)
                                avg_temp = sum(test_frame) / len(test_frame)
                                # Validate frame has reasonable values
                                if -40 <= min_temp <= 125 and -40 <= max_temp <= 125:
                                    print(f"[Thermal] OK MLX90640 initialized successfully (test: min={min_temp:.1f}C, max={max_temp:.1f}C, avg={avg_temp:.1f}C)")
                                    return
                                else:
                                    print(f"[Thermal] Warning: Invalid temp range (min={min_temp:.1f}C, max={max_temp:.1f}C), retrying...")
                                    time.sleep(0.5)
                            except RuntimeError as read_err:
                                # RuntimeError during getFrame - try again
                                if read_attempt < 2:
                                    time.sleep(0.1)
                                    continue
                                else:
                                    raise read_err
                            except Exception as read_err:
                                if read_attempt < 2:
                                    time.sleep(0.5)
                                    continue
                                else:
                                    raise read_err
                        # If we get here, frame read succeeded but values were invalid
                        print("[Thermal] Warning: Frame read succeeded but values seem invalid, continuing anyway...")
                        return
                    except TimeoutError as e:
                        print(f"[Thermal] FAIL Initialization timed out after 10 seconds: {e}")
                        if attempt < 2:
                            print(f"[Thermal] Retrying ({attempt+1}/3)...")
                            time.sleep(2.0)
                            thermal_sensor = None
                            continue
                        else:
                            raise e
                    except RuntimeError as e:
                        error_str = str(e)
                        if "outlier pixels" in error_str:
                            print(f"[Thermal] Outlier pixels error: {e}")
                            print("[Thermal] Sensor detected at I2C 0x33 but has calibration issues.")
                            print("[Thermal] This is a known issue with some MLX90640 sensors.")
                            print("[Thermal] Possible solutions:")
                            print("[Thermal]   1. Sensor may need recalibration (contact manufacturer)")
                            print("[Thermal]   2. Try warming up sensor for 10-15 minutes")
                            print("[Thermal]   3. Check I2C connections and power supply")
                            print("[Thermal]   4. Sensor may have hardware defects - consider replacement")
                            if attempt < 2:
                                print(f"[Thermal] Retrying ({attempt+1}/3) after 5s delay...")
                                time.sleep(5.0)
                                thermal_sensor = None
                                continue
                            else:
                                # Final attempt - try raw I2C workaround
                                print("[Thermal] Attempting final workaround with raw I2C...")
                                try:
                                    raw_sensor = _init_thermal_raw_i2c(i2c)
                                    if raw_sensor is not None:
                                        thermal_sensor = raw_sensor
                                        thermal_sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                                        print("[Thermal] OK Sensor initialized with workaround (reduced accuracy expected)")
                                        return
                                    else:
                                        raise e
                                except Exception:
                                    raise e
                        else:
                            raise e
                    except Exception as e:
                        if attempt < 2:
                            print(f"[Thermal] Warning: Init attempt {attempt+1}/3 failed: {e}, retrying...")
                            time.sleep(2.0)
                            thermal_sensor = None
                            continue
                        else:
                            raise e
    except Exception as e:
        print(f"[Thermal] FAIL MLX90640 init failed after all retries: {e}")
        print("[Thermal] Note: 'More than 4 outlier pixels' indicates sensor calibration issues.")
        print("[Thermal] This may require sensor recalibration or hardware replacement.")
        print("[Thermal] System will continue without thermal imaging.")
        import traceback
        traceback.print_exc()
        with thermal_lock:
            thermal_sensor = None


def read_thermal_frame() -> tuple[bool, list[float] | None]:
    with thermal_lock:
        if thermal_sensor is None:
            return False, None
        try:
            frame = [0.0] * (thermal_shape[0] * thermal_shape[1])
            thermal_sensor.getFrame(frame)
            # Validate frame values (check for NaN/Inf and reasonable range)
            import math
            valid_values = [v for v in frame if math.isfinite(v) and -40 <= v <= 125]
            if len(valid_values) < len(frame) * 0.9:  # Less than 90% valid
                return False, None
            return True, frame
        except Exception as e:
            # Log error occasionally (not every frame)
            import time
            if not hasattr(read_thermal_frame, '_last_error_log') or time.time() - read_thermal_frame._last_error_log > 10:
                print(f"[Thermal] Read error: {e}")
                read_thermal_frame._last_error_log = time.time()
            return False, None


def initialize_bme680_if_available() -> None:
    global bme680_sensor
    print("[BME680] Initializing BME680 environmental sensor...")
    
    if adafruit_bme680 is None:
        print("[BME680] FAIL adafruit_bme680 module not available")
        return
    if board is None:
        print("[BME680] FAIL board module not available")
        return
    if busio is None:
        print("[BME680] FAIL busio module not available")
        return
    
    i2c = _get_or_create_i2c()
    if i2c is None:
        print("[BME680] FAIL Could not get I2C bus")
        return
    
    try:
        with bme680_lock:
            if bme680_sensor is None:
                # Try common I2C addresses: 0x76 and 0x77
                for addr in (0x76, 0x77):
                    try:
                        print(f"[BME680] Trying address 0x{addr:02X}...")
                        # Add small delay to avoid I2C conflicts
                        time.sleep(0.2)
                        sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=addr)
                        sensor.sea_level_pressure = 1013.25
                        # Test reading with retries (I2C can be flaky)
                        for read_retry in range(3):
                            try:
                                time.sleep(0.5)  # Let sensor stabilize
                                temp = sensor.temperature
                                humidity = sensor.relative_humidity
                                pressure = sensor.pressure
                                gas = sensor.gas
                                # Validate readings are reasonable
                                if -40 <= temp <= 85 and 0 <= humidity <= 100 and 300 <= pressure <= 1100:
                                    bme680_sensor = sensor
                                    print(f"[BME680] OK BME680 initialized at 0x{addr:02X} (test: temp={temp:.1f}C, humidity={humidity:.1f}%, pressure={pressure:.1f}hPa, gas={gas:.0f}ohms)")
                                    return
                                else:
                                    print(f"[BME680] Warning: Invalid readings (temp={temp}, humidity={humidity}, pressure={pressure}), retrying...")
                                    time.sleep(0.5)
                            except Exception as read_err:
                                if read_retry < 2:
                                    print(f"[BME680] Read error on attempt {read_retry+1}/3: {read_err}, retrying...")
                                    time.sleep(0.5)
                                    continue
                                else:
                                    raise read_err
                    except OSError as os_err:
                        # I2C I/O error - might be temporary, try again
                        if "Input/output error" in str(os_err):
                            print(f"[BME680] I2C I/O error at 0x{addr:02X}: {os_err}")
                            print("[BME680] This might be due to I2C bus conflicts. Retrying after delay...")
                            time.sleep(1.0)
                            # Try one more time
                            try:
                                sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=addr)
                                sensor.sea_level_pressure = 1013.25
                                time.sleep(0.8)
                                temp = sensor.temperature
                                humidity = sensor.relative_humidity
                                pressure = sensor.pressure
                                gas = sensor.gas
                                bme680_sensor = sensor
                                print(f"[BME680] OK BME680 initialized at 0x{addr:02X} after retry (test: temp={temp:.1f}C, humidity={humidity:.1f}%, pressure={pressure:.1f}hPa, gas={gas:.0f}ohms)")
                                return
                            except Exception:
                                print(f"[BME680] Retry also failed for 0x{addr:02X}")
                                continue
                        else:
                            print(f"[BME680] Address 0x{addr:02X} failed: {os_err}")
                            continue
                    except Exception as e:
                        print(f"[BME680] Address 0x{addr:02X} failed: {e}")
                        continue
                print("[BME680] FAIL BME680 not found at addresses 0x76 or 0x77")
    except Exception as e:
        print(f"[BME680] FAIL BME680 init failed: {e}")
        print("[BME680] Note: I2C errors may be due to bus conflicts or sensor disconnection")
        print("[BME680] Try: sudo i2cdetect -y 1  # Check if sensor appears")
        import traceback
        traceback.print_exc()
        with bme680_lock:
            bme680_sensor = None


def initialize_detector_if_available() -> None:
    global tflite_interpreter, model_input_shape
    if Interpreter is None:
        return
    if not os.path.exists(MODEL_PATH):
        return
    try:
        with detector_lock:
            if tflite_interpreter is None:
                tflite_interpreter = Interpreter(model_path=MODEL_PATH, num_threads=2)
                tflite_interpreter.allocate_tensors()
                input_details = tflite_interpreter.get_input_details()
                if input_details:
                    model_input_shape = input_details[0]["shape"]  # e.g., [1, h, w, c]
    except Exception:
        with detector_lock:
            tflite_interpreter = None
            model_input_shape = None


def ensure_initialized() -> None:
    print("\n" + "="*60)
    print("NDV Monitoring System - Initialization")
    print("="*60)
    load_labels()
    load_symptoms_labels()
    load_settings()
    # Initialize sensors with delays to avoid I2C conflicts
    # BME680 first (it was working before)
    print("[Init] Step 1: Initializing BME680...")
    initialize_bme680_if_available()
    time.sleep(1.0)  # Give I2C bus time to settle
    
    # Then thermal (uses same I2C bus, but different address)
    print("[Init] Step 2: Initializing thermal sensor...")
    initialize_thermal_if_available()
    time.sleep(0.5)
    
    # Then camera (independent hardware)
    print("[Init] Step 3: Initializing camera...")
    initialize_camera_if_available()
    # Models last
    initialize_detector_if_available()
    initialize_symptoms_detector()
    # Start symptoms worker thread
    symptoms_worker_thread = threading.Thread(target=symptoms_worker_loop, daemon=True)
    symptoms_worker_thread.start()
    
    # Print summary
    print("\n" + "="*60)
    print("Initialization Summary:")
    print("="*60)
    with bme680_lock:
        bme_status = "OK" if bme680_sensor is not None else "FAIL"
    with thermal_lock:
        thermal_status = "OK" if thermal_sensor is not None else "FAIL"
    with camera_lock:
        camera_status = "OK" if camera_provider is not None else "FAIL"
    with detector_lock:
        model_status = "OK" if tflite_interpreter is not None else "FAIL"
    
    print(f"  BME680:     {bme_status}")
    print(f"  Thermal:    {thermal_status}")
    print(f"  Camera:     {camera_status} ({camera_provider or 'none'})")
    print(f"  Model:      {model_status}")
    print(f"  Symptoms:   {'OK' if sym_interpreter is not None else 'FAIL'}")
    print("="*60 + "\n")


def log_csv(record: dict) -> None:
    # Logs saved to LOG_DIR/YYYY-MM-DD.csv
    day_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(LOG_DIR, f"{day_str}.csv")
    is_new = not os.path.exists(path)
    headers = [
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "gas_ohms",
        "pressure_hpa",
        "detections_json",
        "max_hotspot_c",
        "notes",
    ]
    try:
        with open(path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(",".join(headers) + "\n")
            row = [
                record.get("timestamp", ""),
                str(record.get("temperature_c", "")),
                str(record.get("humidity_pct", "")),
                str(record.get("gas_ohms", "")),
                str(record.get("pressure_hpa", "")),
                json.dumps(record.get("detections", [])),
                str(record.get("max_hotspot_c", "")),
                record.get("notes", ""),
            ]
            f.write(",".join(row) + "\n")
    except Exception:
        # Ignore logging errors on dev machines
        pass


def summarize_daily_logs() -> dict:
    summary = {
        "days": [],
        "temperature_c_avg": None,
        "humidity_pct_avg": None,
        "detections_per_class": {},
        "max_hotspot_c": None,
    }
    try:
        files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]
    except Exception:
        files = []

    total_temp = 0.0
    total_hum = 0.0
    count_temp = 0
    count_hum = 0
    global_max_hotspot = None

    for fname in sorted(files):
        path = os.path.join(LOG_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if len(parts) < 8:
                        continue
                    # Align with headers defined in log_csv
                    try:
                        temp = float(parts[1]) if parts[1] else None
                        hum = float(parts[2]) if parts[2] else None
                        det_json = parts[5]
                        max_hot = float(parts[6]) if parts[6] else None
                    except Exception:
                        temp = None
                        hum = None
                        det_json = "[]"
                        max_hot = None

                    if temp is not None:
                        total_temp += temp
                        count_temp += 1
                    if hum is not None:
                        total_hum += hum
                        count_hum += 1
                    if max_hot is not None:
                        if global_max_hotspot is None or max_hot > global_max_hotspot:
                            global_max_hotspot = max_hot
                    try:
                        dets = json.loads(det_json)
                        for d in dets:
                            cls = d.get("class", "unknown")
                            summary["detections_per_class"][cls] = summary["detections_per_class"].get(cls, 0) + 1
                    except Exception:
                        pass
            day = fname.replace(".csv", "")
            summary["days"].append(day)
        except Exception:
            continue

    if count_temp:
        summary["temperature_c_avg"] = total_temp / count_temp
    if count_hum:
        summary["humidity_pct_avg"] = total_hum / count_hum
    summary["max_hotspot_c"] = global_max_hotspot
    return summary


def compute_system_status() -> dict:
    uptime_seconds = None
    try:
        boot_time = psutil.boot_time() if psutil else None
        uptime_seconds = int(time.time() - boot_time) if boot_time else None
    except Exception:
        uptime_seconds = None
    try:
        cpu_percent = psutil.cpu_percent(interval=None) if psutil else None
        memory = psutil.virtual_memory() if psutil else None
        memory_percent = memory.percent if memory else None
        cpu_temp_c = None
        if psutil and hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            # Try typical keys; this varies by platform
            for key in ["cpu-thermal", "coretemp", "soc_thermal"]:
                if key in temps and temps[key]:
                    cpu_temp_c = temps[key][0].current
                    break
    except Exception:
        cpu_percent = None
        memory_percent = None
        cpu_temp_c = None

    return {
        "cpu_percent": cpu_percent,
        "cpu_temp_c": cpu_temp_c,
        "memory_percent": memory_percent,
        "uptime_seconds": uptime_seconds,
        "online": True,
    }


def format_mjpeg(frame_bgr: "np.ndarray") -> bytes:
    if cv2 is None:
        return b""
    # Force a consistent output dimension for both streams
    try:
        if frame_bgr.shape[1] != STREAM_WIDTH or frame_bgr.shape[0] != STREAM_HEIGHT:
            frame_bgr = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT))
    except Exception:
        pass
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes() if ok else b""


def nms_boxes(boxes, scores, iou_threshold=0.45):
    if not boxes:
        return []
    # boxes: [ [x1,y1,x2,y2], ... ] in pixels
    import numpy as _np
    idxs = _np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        i_box = _np.array(boxes[i], dtype=_np.float32)
        rest = _np.array([boxes[j] for j in idxs[1:]], dtype=_np.float32)
        xx1 = _np.maximum(i_box[0], rest[:, 0])
        yy1 = _np.maximum(i_box[1], rest[:, 1])
        xx2 = _np.minimum(i_box[2], rest[:, 2])
        yy2 = _np.minimum(i_box[3], rest[:, 3])
        inter_w = _np.maximum(0.0, xx2 - xx1)
        inter_h = _np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        area_i = (i_box[2] - i_box[0]) * (i_box[3] - i_box[1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        union = area_i + area_r - inter
        iou = inter / (union + 1e-6)
        idxs = idxs[1:][iou <= iou_threshold]
    return keep


def _sanitize_dets(boxes, scores, classes):
    """Drop rows with NaN/Inf or negative scores; return numpy arrays."""
    if np is None:
        return [], [], []
    try:
        boxes_np = np.asarray(boxes, dtype=np.float32)
        scores_np = np.asarray(scores, dtype=np.float32)
        classes_np = np.asarray(classes, dtype=np.int32)
        # Coerce invalid scores to -1 then filter
        scores_np = np.nan_to_num(scores_np, nan=-1.0, posinf=-1.0, neginf=-1.0)
        finite_mask = np.isfinite(boxes_np).all(axis=1) & (scores_np >= 0.0)
        boxes_np = boxes_np[finite_mask]
        scores_np = scores_np[finite_mask]
        classes_np = classes_np[finite_mask]
        return boxes_np, scores_np, classes_np
    except Exception:
        return [], [], []


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def letterbox_bgr(bgr, input_hw: tuple[int, int]):
    in_w, in_h = input_hw
    h, w = bgr.shape[:2]
    r = min(in_w / w, in_h / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (new_w, new_h))
    canvas = np.full((in_h, in_w, 3), 114, dtype=resized.dtype)
    pad_w = in_w - new_w; pad_h = in_h - new_h
    pad_left = pad_w // 2; pad_top = pad_h // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, (r, pad_left, pad_top), (w, h)

def _postprocess_yolo_rows(out, input_hw: tuple[int,int], map_info, orig_wh, conf_thres: float):
    out2 = out
    while out2.ndim > 2:
        out2 = out2[0]
    preds = out2.T if out2.shape[0] < out2.shape[1] else out2
    in_w, in_h = input_hw
    r, pad_left, pad_top = map_info
    orig_w, orig_h = orig_wh
    boxes, scores, classes = [], [], []
    for row in preds:
        c = row.shape[0]
        if c < 5:
            continue
        x, y, w, h = row[0:4]
        if c == 5:
            score = float(row[4]); cls_idx = 0
        else:
            tail = row[4:]
            if (tail.max() <= 1.0 + 1e-3) and (tail.min() >= -1e-3):
                cls_scores = tail; cls_idx = int(cls_scores.argmax()); score = float(cls_scores[cls_idx])
            else:
                obj = row[4]; cls_scores = row[5:]
                cls_idx = int(cls_scores.argmax()); score = float(obj * float(cls_scores[cls_idx]))
        if score < conf_thres:
            continue
        normalized = max(x, y, w, h) <= 2.0
        if normalized:
            x *= in_w; y *= in_h; w *= in_w; h *= in_h
        x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
        x1 = (x1 - pad_left) / r; x2 = (x2 - pad_left) / r
        y1 = (y1 - pad_top) / r;  y2 = (y2 - pad_top) / r
        x1 = max(0.0, min(x1, orig_w - 1)); y1 = max(0.0, min(y1, orig_h - 1))
        x2 = max(0.0, min(x2, orig_w - 1)); y2 = max(0.0, min(y2, orig_h - 1))
        boxes.append([x1, y1, x2, y2]); scores.append(score); classes.append(cls_idx)
    if not boxes:
        return [], [], []
    return boxes, scores, classes

def run_tflite_inference(frame_bgr: "np.ndarray") -> list:
    CONF = float(get_setting("DETECT_CONF", "0.35"))
    IOU = float(get_setting("DETECT_IOU", "0.45"))
    MIN_AREA = float(get_setting("MIN_AREA_RATIO", "0.003")) * (STREAM_WIDTH * STREAM_HEIGHT)
    MAX_AREA = float(get_setting("MAX_AREA_RATIO", "0.7"))   * (STREAM_WIDTH * STREAM_HEIGHT)
    TOPK = int(get_setting("PRE_NMS_TOPK", "100"))

    with detector_lock:
        if tflite_interpreter is None or model_input_shape is None or np is None or cv2 is None:
            return []
        in_det = tflite_interpreter.get_input_details()[0]
        out_det = tflite_interpreter.get_output_details()[0]
        _, in_h, in_w, _ = in_det["shape"]

    letter, map_info, orig_wh = letterbox_bgr(frame_bgr, (in_w, in_h))
    model_input = (letter.astype("float32") / 255.0)[None, ...]
    with detector_lock:
        tflite_interpreter.set_tensor(in_det["index"], model_input)
        tflite_interpreter.invoke()
        out_data = tflite_interpreter.get_tensor(out_det["index"])

    boxes, scores, classes = _postprocess_yolo_rows(out_data, (in_w, in_h), map_info, orig_wh, CONF)
    if not boxes:
        return []
    import numpy as _np
    boxes_np, scores_np, classes_np = _sanitize_dets(boxes, scores, classes)
    if boxes_np.size == 0:
        return []
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    keep_size = (areas >= MIN_AREA) & (areas <= MAX_AREA) & _np.isfinite(areas)
    if not _np.any(keep_size):
        return []
    boxes_np = boxes_np[keep_size]; scores_np = scores_np[keep_size]; classes_np = classes_np[keep_size]
    # Whitelist classes: Chicken only (primary model)
    try:
        if len(labels) > 1 and "Chicken" in labels:
            chi = labels.index("Chicken")
            m = (classes_np == chi)
            if not _np.any(m):
                return []
            boxes_np = boxes_np[m]; scores_np = scores_np[m]; classes_np = classes_np[m]
    except Exception:
        pass
    dets = []
    for cls in _np.unique(classes_np):
        idxs = _np.where(classes_np == cls)[0]
        kept = nms_boxes([[*boxes_np[i]] for i in idxs], [float(scores_np[i]) for i in idxs], iou_threshold=IOU)
        for k in kept[:TOPK]:
            i = idxs[k]
            x1, y1, x2, y2 = boxes_np[i]
            dets.append({"x1": int(round(x1)), "y1": int(round(y1)), "x2": int(round(x2)), "y2": int(round(y2)),
                         "score": float(scores_np[i]), "class": int(cls)})
    return dets


def generate_camera_stream():
    if cv2 is None:
        # Hardware/library unavailable: do not simulate per requirements
        return
    while True:
        ok, frame = read_camera_frame()
        if not ok or frame is None:
            time.sleep(0.1)
            continue
        # Overlay recent alert if any
        try:
            now_ts = time.time()
            if now_ts - last_alert_time <= OVERLAY_ALERT_SECONDS and last_alert_message:
                cv2.putText(
                    frame,
                    last_alert_message,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            # Draw latest detections if recent
            with latest_detections_lock:
                if now_ts - latest_detections_ts <= DETECTION_OVERLAY_SECONDS and latest_detections:
                    for d in latest_detections:
                        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                        cls_idx = d["class"]
                        label = labels[cls_idx] if 0 <= cls_idx < len(labels) else str(cls_idx)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Remove confidence from visible overlay per request
                        cv2.putText(frame, f"{label}", (x1, max(12, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception:
            pass
        jpeg = format_mjpeg(frame)
        if not jpeg:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")


def colormap_thermal(values: list[float]) -> "np.ndarray | None":
    if np is None or cv2 is None:
        return None
    try:
        arr = np.array(values, dtype=np.float32).reshape(thermal_shape)
        min_v = np.min(arr)
        max_v = np.max(arr)
        if max_v - min_v < 1e-6:
            max_v = min_v + 1e-6
        norm = ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        # Custom colormap: cooler (blue/green) -> warmer (orange/red)
        # Create lookup table: 0=blue, 85=cyan, 170=green, 212=yellow, 255=red
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:  # blue -> cyan: increase green from 0 to 255
                r, g, b = 0, int(i * 255 / 85), 255
            elif i < 170:  # cyan -> green: decrease blue from 255 to 0
                r, g, b = 0, 255, int(255 - (i - 85) * 255 / 85)
            elif i < 212:  # green -> yellow: increase red from 0 to 255
                r, g, b = int((i - 170) * 255 / 42), 255, 0
            else:  # yellow -> orange -> red: decrease green from 255 to 0
                r, g, b = 255, int(255 - (i - 212) * 255 / 43), 0
            lut[i, 0] = [b, g, r]  # BGR format
        colored = cv2.LUT(norm, lut)
        return colored
    except Exception:
        return None


def generate_thermal_stream():
    ok, frame_vals = read_thermal_frame()
    if not ok or frame_vals is None:
        return
    while True:
        ok, frame_vals = read_thermal_frame()
        if not ok or frame_vals is None:
            time.sleep(0.25)
            continue
        # Hotspot detection (no UI alert)
        try:
            ambient_avg = sum(frame_vals) / len(frame_vals)
            max_hotspot = max(frame_vals)
            if max_hotspot > ambient_avg + HOTSPOT_DELTA_C:
                log_csv({
                    "timestamp": datetime.utcnow().isoformat(),
                    "temperature_c": None,
                    "humidity_pct": None,
                    "gas_ohms": None,
                    "pressure_hpa": None,
                    "detections": [],
                    "max_hotspot_c": round(float(max_hotspot), 2),
                    "notes": "thermal_hotspot",
                })
        except Exception:
            pass

        colored = colormap_thermal(frame_vals)
        if colored is None:
            time.sleep(0.1)
            continue
        # Flip horizontally to match visible orientation
        try:
            colored = cv2.flip(colored, 1)
        except Exception:
            pass
        # Ensure consistent sizing
        try:
            if colored.shape[1] != STREAM_WIDTH or colored.shape[0] != STREAM_HEIGHT:
                colored = cv2.resize(colored, (STREAM_WIDTH, STREAM_HEIGHT))
        except Exception:
            pass

        # Helper: compute approximate chicken temperature from thermal grid under a box
        def box_temperature_c(frame_values_list, x1p, y1p, x2p, y2p):
            try:
                if np is None:
                    return None
                arr = np.array(frame_values_list, dtype=np.float32).reshape(thermal_shape)
                # Flip horizontally to match the flipped displayed image
                arr = arr[:, ::-1]
                # Map pixels to thermal grid indices
                col1 = max(0, min(thermal_shape[1]-1, int(x1p / max(1, STREAM_WIDTH) * thermal_shape[1])))
                col2 = max(0, min(thermal_shape[1]-1, int(x2p / max(1, STREAM_WIDTH) * thermal_shape[1])))
                row1 = max(0, min(thermal_shape[0]-1, int(y1p / max(1, STREAM_HEIGHT) * thermal_shape[0])))
                row2 = max(0, min(thermal_shape[0]-1, int(y2p / max(1, STREAM_HEIGHT) * thermal_shape[0])))
                c1, c2 = sorted((col1, col2))
                r1, r2 = sorted((row1, row2))
                if r2 < r1 or c2 < c1:
                    return None
                region = arr[r1:r2+1, c1:c2+1]
                if region.size == 0:
                    return None
                return float(np.max(region))
            except Exception:
                return None

        # Overlay latest detections from visible camera, adjusted by calibration offsets and flip
        try:
            now_ts = time.time()
            with latest_detections_lock:
                if now_ts - latest_detections_ts <= DETECTION_OVERLAY_SECONDS and latest_detections:
                    for d in latest_detections:
                        # Start from visible coords
                        vx1 = d["x1"] + CALIBRATION_OFFSET_X
                        vy1 = d["y1"] + CALIBRATION_OFFSET_Y
                        vx2 = d["x2"] + CALIBRATION_OFFSET_X
                        vy2 = d["y2"] + CALIBRATION_OFFSET_Y
                        # Mirror horizontally for flipped thermal
                        tx1 = STREAM_WIDTH - 1 - vx2
                        tx2 = STREAM_WIDTH - 1 - vx1
                        x1 = max(0, min(STREAM_WIDTH - 1, tx1))
                        y1 = max(0, min(STREAM_HEIGHT - 1, vy1))
                        x2 = max(0, min(STREAM_WIDTH - 1, tx2))
                        y2 = max(0, min(STREAM_HEIGHT - 1, vy2))
                        cls_idx = d.get("class", -1)
                        label = labels[cls_idx] if 0 <= cls_idx < len(labels) else str(cls_idx)
                        cv2.rectangle(colored, (x1, y1), (x2, y2), (255, 140, 0), 2)
                        # For thermal overlay: show chicken temperature (max in box), no confidence
                        temp_c = box_temperature_c(frame_vals, x1, y1, x2, y2)
                        if label.lower() == "chicken" and temp_c is not None:
                            cv2.putText(colored, f"{label} {temp_c:.1f}C", (x1, max(12, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(colored, f"{label}", (x1, max(12, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)
        except Exception:
            pass

        jpeg = format_mjpeg(colored)
        if not jpeg:
            time.sleep(0.1)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera_feed")
def camera_feed():
    with camera_lock:
        provider = camera_provider
    if provider not in ('cv2', 'picamera2'):
        return jsonify({"error": "cam offline"}), 503
    return Response(stream_with_context(generate_camera_stream()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/thermal_feed")
def thermal_feed():
    if thermal_sensor is None:
        return jsonify({"error": "sensor offline"}), 503
    return Response(stream_with_context(generate_thermal_stream()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/sensor_data")
def sensor_data():
    with bme680_lock:
        if bme680_sensor is None:
            return jsonify({"status": "offline"})
        try:
            data = {
                "status": "online",
                "temperature_c": round(float(bme680_sensor.temperature), 2),
                "humidity_pct": round(float(bme680_sensor.relative_humidity), 2),
                "gas_ohms": round(float(bme680_sensor.gas), 2),
                "pressure_hpa": round(float(bme680_sensor.pressure), 2),
            }
            # Throttled logging every 60s
            try:
                global last_sensor_log_time
                now_ts = time.time()
                if now_ts - last_sensor_log_time > 60.0:
                    last_sensor_log_time = now_ts
                    log_csv({
                        "timestamp": datetime.utcnow().isoformat(),
                        "temperature_c": data["temperature_c"],
                        "humidity_pct": data["humidity_pct"],
                        "gas_ohms": data["gas_ohms"],
                        "pressure_hpa": data["pressure_hpa"],
                        "detections": [],
                        "max_hotspot_c": None,
                        "notes": "sensor_poll",
                    })
            except Exception:
                pass
            return jsonify(data)
        except Exception:
            return jsonify({"status": "offline"})


@app.route("/detect", methods=["POST", "GET"])
def detect():
    # Returns detection results on latest frame. If unavailable, returns offline.
    with detector_lock:
        if tflite_interpreter is None or model_input_shape is None:
            return jsonify({"status": "offline", "reason": "model unavailable"}), 503

    ok, frame = read_camera_frame()
    if (not ok or frame is None) and np is not None:
        # Retry a few times to tolerate intermittent read failures
        for _ in range(4):
            time.sleep(0.15)
            ok, frame = read_camera_frame()
            if ok and frame is not None:
                break
    if not ok or frame is None or np is None:
        return jsonify({"status": "offline", "reason": "camera unavailable"}), 503

    dets = run_tflite_inference(frame)
    with latest_detections_lock:
        global latest_detections, latest_detections_ts
        latest_detections = dets
        latest_detections_ts = time.time()

    # Queue symptoms jobs for top-K chickens
    if dets and sym_interpreter is not None:
        try:
            order = sorted(range(len(dets)), key=lambda i: dets[i]["score"], reverse=True)[:int(os.getenv("SYMPTOMS_MAX_ROI", "3"))]
            for i in order:
                d = dets[i]
                x1 = max(0, d["x1"]); y1 = max(0, d["y1"]) ; x2 = min(STREAM_WIDTH-1, d["x2"]) ; y2 = min(STREAM_HEIGHT-1, d["y2"]) 
                if x2 <= x1 or y2 <= y1:
                    continue
                try:
                    sym_task_queue.put_nowait((frame.copy(), x1, y1, x2, y2))
                except Exception:
                    pass
        except Exception:
            pass

    # Build overlay on visible frame for saving
    vis_out = frame.copy()
    try:
        if cv2 is not None:
            for d in dets:
                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                score = d["score"]
                label = "Chicken"
                cv2.rectangle(vis_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_out, f"{label} {score:.2f}", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # Overlay latest symptom boxes if present
            with sym_results_lock:
                if latest_symptom_overlays and time.time() - latest_symptoms_ts <= DETECTION_OVERLAY_SECONDS:
                    for s in latest_symptom_overlays:
                        cv2.rectangle(vis_out, (s["x1"], s["y1"]), (s["x2"], s["y2"]), (0, 165, 255), 2)
                        cv2.putText(vis_out, f"{s['label']} {s['score']:.2f}", (s["x1"], max(12, s["y1"] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)
    except Exception:
        pass

    # Thermal snapshot and ambient/hotspot (unchanged)
    thermal_colored = None
    amb = None
    hot = None
    ok_t, frame_vals = read_thermal_frame()
    if ok_t and frame_vals is not None:
        try:
            amb = float(sum(frame_vals) / len(frame_vals))
            hot = float(max(frame_vals))
            col = colormap_thermal(frame_vals)
            if col is not None:
                # Overlay detections with calibration offset for saving
                if cv2 is not None and dets:
                    for d in dets:
                        x1 = max(0, min(STREAM_WIDTH - 1, d["x1"] + CALIBRATION_OFFSET_X))
                        y1 = max(0, min(STREAM_HEIGHT - 1, d["y1"] + CALIBRATION_OFFSET_Y))
                        x2 = max(0, min(STREAM_WIDTH - 1, d["x2"] + CALIBRATION_OFFSET_X))
                        y2 = max(0, min(STREAM_HEIGHT - 1, d["y2"] + CALIBRATION_OFFSET_Y))
                        cv2.rectangle(col, (x1, y1), (x2, y2), (255, 140, 0), 2)
                # Overlay symptoms (orange) if present
                with sym_results_lock:
                    if latest_symptom_overlays and time.time() - latest_symptoms_ts <= DETECTION_OVERLAY_SECONDS:
                        for s in latest_symptom_overlays:
                            x1 = max(0, min(STREAM_WIDTH - 1, s["x1"] + CALIBRATION_OFFSET_X))
                            y1 = max(0, min(STREAM_HEIGHT - 1, s["y1"] + CALIBRATION_OFFSET_Y))
                            x2 = max(0, min(STREAM_WIDTH - 1, s["x2"] + CALIBRATION_OFFSET_X))
                            y2 = max(0, min(STREAM_HEIGHT - 1, s["y2"] + CALIBRATION_OFFSET_Y))
                            cv2.rectangle(col, (x1, y1), (x2, y2), (0, 140, 255), 2)
                thermal_colored = col
        except Exception:
            pass

    # BME snapshot (unchanged)
    bme = None
    with bme680_lock:
        if bme680_sensor is not None:
            try:
                bme = {
                    "temperature_c": round(float(bme680_sensor.temperature), 2),
                    "humidity_pct": round(float(bme680_sensor.relative_humidity), 2),
                    "gas_ohms": round(float(bme680_sensor.gas), 2),
                    "pressure_hpa": round(float(bme680_sensor.pressure), 2),
                }
            except Exception:
                bme = None

    # Save images (unchanged)
    ts_iso = datetime.utcnow().isoformat()
    vis_path, th_path = save_images_and_build_paths(vis_out, thermal_colored, ts_iso)

    # Insert log row (unchanged)
    try:
        insert_log({
            "timestamp": ts_iso,
            "bme_temp": bme.get("temperature_c") if bme else None,
            "bme_humidity": bme.get("humidity_pct") if bme else None,
            "bme_pressure": bme.get("pressure_hpa") if bme else None,
            "bme_gas": bme.get("gas_ohms") if bme else None,
            "thermal_hotspot_temp": hot,
            "ambient_temp": amb,
            "symptoms_detected": ", ".join(sorted({s["label"] for s in latest_symptom_overlays})) if latest_symptom_overlays else "",
            "has_abnormality": bool(latest_symptom_overlays),
            "image_path_visible": vis_path,
            "image_path_thermal": th_path,
            "notes": "detect",
        })
    except Exception:
        pass

    return jsonify({
        "status": "online",
        "detections": dets,
        "symptoms": latest_symptom_overlays,
        "labels": labels,
        "symptom_labels": sym_labels,
        "saved_visible": vis_path,
        "saved_thermal": th_path,
        "abnormal": bool(latest_symptom_overlays),
    })


@app.route("/analytics")
def analytics():
    return jsonify(summarize_daily_logs())


@app.route("/status")
def status():
    return jsonify(compute_system_status())


@app.route("/notifications")
def notifications():
    with alerts_lock:
        items = list(alerts_log)
    return jsonify({"alerts": items})


@app.route("/api/alerts/read", methods=["POST"])  # mark read by id
def api_alerts_read():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get("ids", [])
        if not isinstance(ids, list):
            return jsonify({"ok": False, "error": "ids must be list"}), 400
        with alerts_lock:
            for a in list(alerts_log):
                if a["id"] in ids:
                    a["read"] = True
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/summaries/generate", methods=["POST"])  # manual trigger
def api_generate_summaries():
    try:
        # run once immediately for yesterday
        th = threading.Thread(target=daily_summaries_worker, daemon=True)
        th.start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/logs")
def logs_page():
    return render_template("logs.html")


@app.route("/settings", methods=["GET"])
def settings_page():
    return render_template("settings.html")


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify(load_settings())
    try:
        payload = request.get_json(silent=True) or {}
        # allow specific keys only
        allowed = {
            "DETECT_CONF", "DETECT_IOU", "PRE_NMS_TOPK", "MIN_AREA_RATIO", "MAX_AREA_RATIO",
            "SYMPTOMS_CONF", "SYMPTOMS_MAX_ROI",
            "NORMAL_LOG_RETENTION_HOURS", "ABNORMAL_LOG_RETENTION_DAYS", "LOG_VACUUM_EVERY_HOURS",
        }
        updates = {k: payload[k] for k in payload.keys() if k in allowed}
        save_settings(updates)
        return jsonify({"ok": True, "saved": updates})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/logs")
def api_logs():
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 20))
        offset = (page - 1) * page_size
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM logs")
        total = cur.fetchone()[0]
        cur.execute(
            """
            SELECT id, timestamp, bme_temp, bme_humidity, bme_pressure, bme_gas,
                   thermal_hotspot_temp, ambient_temp, symptoms_detected,
                   has_abnormality, image_path_visible, image_path_thermal, notes
            FROM logs
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (page_size, offset),
        )
        items = []
        def _status_level(t, h, g, p):
            try:
                level = "normal"
                # Temperature
                if t is not None:
                    if t > 33 or t < 20:
                        return "critical"
                    if 30 <= t <= 33:
                        level = "warning"
                # Humidity
                if h is not None:
                    if h > 85 or h < 50:
                        return "critical"
                    if h > 75:
                        level = "warning"
                # Gas (kOhm expected; stored may be ohms)
                if g is not None:
                    # If value looks like ohms, convert to kOhm for comparison
                    gv = float(g)
                    gk = gv / 1000.0 if gv > 1000 else gv
                    if gk < 50:
                        level = "critical"
                    elif gk < 100:
                        level = "warning"
                # Pressure
                if p is not None:
                    if p < 995:
                        level = "critical"
                    elif p < 1000 or p > 1020:
                        level = "warning"
                return level
            except Exception:
                return "normal"

        for r in cur.fetchall():
            items.append({
                "id": r[0],
                "timestamp": r[1],
                "bme_temp": r[2],
                "bme_humidity": r[3],
                "bme_pressure": r[4],
                "bme_gas": r[5],
                "thermal_hotspot_temp": r[6],
                "ambient_temp": r[7],
                "symptoms_detected": r[8] or "",
                "has_abnormality": bool(r[9]),
                "image_path_visible": r[10],
                "image_path_thermal": r[11],
                "notes": r[12] or "",
                "status_level": _status_level(r[2], r[3], r[5], r[4]),
            })
        return jsonify({"total": total, "page": page, "page_size": page_size, "items": items})
    except Exception:
        return jsonify({"total": 0, "page": 1, "page_size": 20, "items": []})


@app.route("/api/summaries")
def api_summaries():
    # Build simple on-demand summaries
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Daily (last 30 days)
        cur.execute(
            """
            SELECT substr(timestamp,1,10) as day,
                   COUNT(*),
                   SUM(has_abnormality),
                   AVG(bme_temp), AVG(bme_humidity), AVG(bme_pressure), AVG(bme_gas),
                   SUM(CASE WHEN symptoms_detected IS NOT NULL AND LENGTH(symptoms_detected)>0 THEN 1 ELSE 0 END)
            FROM logs
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30
            """
        )
        daily = [
            {
                "day": row[0],
                "count": row[1],
                "abnormal": int(row[2] or 0),
                "avg_temp": float(row[3]) if row[3] is not None else None,
                "avg_humidity": float(row[4]) if row[4] is not None else None,
                "avg_pressure": float(row[5]) if row[5] is not None else None,
                "avg_gas": float(row[6]) if row[6] is not None else None,
                "symptom_count": int(row[7] or 0),
            }
            for row in cur.fetchall()
        ]
        # status summary donut (last 30 days)
        cur.execute(
            """
            SELECT timestamp, bme_temp, bme_humidity, bme_gas, bme_pressure
            FROM logs
            WHERE timestamp >= date('now','-30 day')
            """
        )
        status_counts = {"normal": 0, "warning": 0, "critical": 0}
        rows = cur.fetchall()
        def _status_level(t, h, g, p):
            try:
                level = "normal"
                if t is not None:
                    if t > 33 or t < 20:
                        return "critical"
                    if 30 <= t <= 33:
                        level = "warning"
                if h is not None:
                    if h > 85 or h < 50:
                        return "critical"
                    if h > 75:
                        level = "warning"
                if g is not None:
                    gv = float(g); gk = gv / 1000.0 if gv > 1000 else gv
                    if gk < 50:
                        level = "critical"
                    elif gk < 100:
                        level = "warning"
                if p is not None:
                    if p < 995:
                        level = "critical"
                    elif p < 1000 or p > 1020:
                        level = "warning"
                return level
            except Exception:
                return "normal"
        for (_, t, h, g, p) in rows:
            status_counts[_status_level(t, h, g, p)] += 1
        # Weekly (last 12 weeks)
        cur.execute(
            """
            SELECT strftime('%Y-W%W', timestamp) as week,
                   COUNT(*),
                   SUM(has_abnormality)
            FROM logs
            GROUP BY week
            ORDER BY week DESC
            LIMIT 12
            """
        )
        weekly = [
            {"week": row[0], "count": row[1], "abnormal": int(row[2] or 0)} for row in cur.fetchall()
        ]
        # Monthly (last 12 months)
        cur.execute(
            """
            SELECT substr(timestamp,1,7) as month,
                   COUNT(*),
                   SUM(has_abnormality)
            FROM logs
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
            """
        )
        monthly = [
            {"month": row[0], "count": row[1], "abnormal": int(row[2] or 0)} for row in cur.fetchall()
        ]
        return jsonify({
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly,
            "status_summary": status_counts,
        })
    except Exception as e:
        print(f"[Summaries] Error: {e}")
        return jsonify({
            "daily": [],
            "weekly": [],
            "monthly": [],
            "status_summary": {"normal": 0, "warning": 0, "critical": 0}
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.route("/api/logs/delete", methods=["POST"])
def api_logs_delete():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get("ids", [])
        if not isinstance(ids, list) or not ids:
            return jsonify({"ok": False, "error": "no_ids"}), 400
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # fetch images to remove
        qmarks = ",".join(["?"] * len(ids))
        cur.execute(f"SELECT image_path_visible, image_path_thermal FROM logs WHERE id IN ({qmarks})", ids)
        rows = cur.fetchall()
        for vp, tp in rows:
            for p in (vp, tp):
                if p and os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        cur.execute(f"DELETE FROM logs WHERE id IN ({qmarks})", ids)
        conn.commit()
        return jsonify({"ok": True, "deleted": len(ids)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.route("/model_status")
def model_status():
    with detector_lock:
        available = tflite_interpreter is not None and model_input_shape is not None
    sym_available = sym_interpreter is not None and sym_input_shape is not None
    reason = None
    if not available:
        if not os.path.exists(MODEL_PATH):
            reason = "model_missing"
        elif Interpreter is None:
            reason = "tflite_runtime_missing"
        else:
            reason = "load_failed"
    return jsonify({
        "available": available,
        "reason": reason,
        "model": MODEL_FILE,
        "labels": len(labels),
        "symptoms_available": sym_available,
        "symptoms_labels": len(sym_labels),
    })


@app.teardown_appcontext
def _cleanup(exception):  # noqa: ANN001
    release_camera()


def daily_summaries_worker() -> None:
    # Generate CSV + PNG at midnight based on last day's data
    while True:
        try:
            now = datetime.now()
            # compute next midnight
            nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            sleep_s = max(5.0, (nxt - now).total_seconds())
            time.sleep(sleep_s)
            # fetch yesterday day string
            yday = (nxt - timedelta(days=1)).strftime('%Y-%m-%d')
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT timestamp, bme_temp, bme_humidity, bme_pressure, bme_gas,
                       has_abnormality,
                       CASE WHEN symptoms_detected IS NOT NULL AND LENGTH(symptoms_detected)>0 THEN 1 ELSE 0 END
                FROM logs
                WHERE substr(timestamp,1,10)=?
                ORDER BY timestamp ASC
                """,
                (yday,)
            )
            rows = cur.fetchall()
            conn.close()
            if not rows:
                # Still post a daily summary notification without files
                add_alert(f"Daily summary ready: {yday}")
                # Weekly/monthly checks still run based on calendar
                pass
            # write CSV
            csv_path = os.path.join(SUMMARIES_DIR, f"{yday}.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('timestamp,temp,humidity,pressure,gas,abnormal,symptom\n')
                for r in rows:
                    f.write(','.join(str(x if x is not None else '') for x in r) + '\n')
            # plot PNG if matplotlib available
            if plt is not None and rows:
                times = [r[0][11:16] for r in rows]
                temp = [r[1] for r in rows]
                hum  = [r[2] for r in rows]
                gas  = [ (r[4]/1000.0 if (r[4] is not None and r[4]>1000) else r[4]) for r in rows ]
                fig, ax1 = plt.subplots(figsize=(10,4))
                ax1.set_title(f"NDV Daily Summary - {yday}")
                # trend color: green if falling humidity, orange if rising
                if hum and hum[0] is not None and hum[-1] is not None and hum[-1] > hum[0]:
                    c_h = '#f59e0b'
                else:
                    c_h = '#16a34a'
                ax1.plot(times, temp, color='#ef4444', label='Temp (C)')
                ax1.set_ylabel('Temp (C)', color='#ef4444')
                ax2 = ax1.twinx()
                ax2.plot(times, hum, color=c_h, label='Humidity (%)')
                ax2.plot(times, gas, color='#0ea5e9', linestyle='--', label='Gas (kOhm)')
                ax2.set_ylabel('Humidity / Gas (kOhm)')
                fig.autofmt_xdate(rotation=45)
                fig.tight_layout()
                png_path = os.path.join(SUMMARIES_DIR, f"{yday}.png")
                fig.savefig(png_path)
                plt.close(fig)
                add_alert(f"Daily summary ready: {yday}")

            # Also raise weekly and monthly summary notifications at boundaries
            # Weekly: when yday is Sunday (ISO weekday 7) we announce Weekly
            try:
                ydate = datetime.strptime(yday, '%Y-%m-%d').date()
                # ISO weekday: Monday=1..Sunday=7; if Sunday, post weekly summary
                if ydate.isoweekday() == 7:
                    wk = ydate.isocalendar().week
                    add_alert(f"Weekly summary ready: {ydate.isocalendar().year}-W{wk:02d}")
                # Monthly: if yday is last day of month, post monthly summary
                first_next = (ydate.replace(day=1) + timedelta(days=32)).replace(day=1)
                last_this = first_next - timedelta(days=1)
                if ydate == last_this:
                    add_alert(f"Monthly summary ready: {ydate.strftime('%Y-%m')}")
            except Exception:
                pass
        except Exception:
            time.sleep(60.0)

def main() -> None:
    ensure_initialized()
    # Start symptoms worker if model is available
    if sym_interpreter is not None:
        t = threading.Thread(target=symptoms_worker_loop, daemon=True)
        t.start()
    init_db()
    t = threading.Thread(target=cleanup_normal_logs_loop, daemon=True)
    t.start()
    # Nightly summaries worker
    t2 = threading.Thread(target=daily_summaries_worker, daemon=True)
    t2.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
