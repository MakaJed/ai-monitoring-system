NDV Poultry Monitor Dashboard (Flask)
=====================================

Offline-capable dashboard for Raspberry Pi 4.

Features
- Visible NoIR camera live feed with YOLOv8-TFLite detections
- Thermal MLX90640 visualization with hotspot alerts
- BME680 temperature, humidity, pressure, gas readings
- Thermal overlay mirrors visible-camera boxes with calibration offsets
- Split View: visible + thermal side by side
- Daily CSV logs and analytics summary endpoint

Project Layout
- `dashboard/app.py` (Flask backend and endpoints)
- `dashboard/templates/index.html` (UI)
- `dashboard/static/css/style.css`, `dashboard/static/js/app.js`
- `models/chickenmodel.tflite`, `models/chicken.labels.txt`
- `systemd/ndv-dashboard.service` (autostart service)

Prerequisites (Raspberry Pi OS 64-bit)
```bash
sudo raspi-config  # enable Camera and I2C
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv python3-numpy python3-psutil libatlas-base-dev python3-picamera2
```

Python deps (Python 3.11)
```bash
cd /home/pi/NDV_Monitoring
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Model files
- Put your chicken model at `models/chickenmodel.tflite` (labels in `models/chicken.labels.txt`). Optional: `models/ndvsymptoms.tflite` + `models/symptoms.labels.txt`.

Run (dev)
```bash
python3 dashboard/app.py
# open http://<pi-ip>:5000
```

Install as a service
```bash
sudo cp systemd/ndv-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ndv-dashboard
sudo systemctl restart ndv-dashboard
sudo systemctl status ndv-dashboard --no-pager
```

Endpoints
- `/camera_feed` – Visible MJPEG
- `/thermal_feed` – Thermal MJPEG (with visible boxes overlaid)
- `/sensor_data` – BME680 JSON (status, temperature_c, humidity_pct, gas_ohms, pressure_hpa)
- `/detect` – Runs YOLOv8-TFLite, returns detections and updates overlay
- `/analytics` – Aggregated daily metrics from CSV logs
- `/logs` – Logs & Analytics UI (table + summaries)
- `/api/logs` – Paginated logs from SQLite
- `/api/summaries` – Daily/weekly/monthly summaries
- `/status`, `/notifications`

Calibration
- Adjust `CALIBRATION_OFFSET_X` and `CALIBRATION_OFFSET_Y` in `dashboard/app.py` if thermal overlay boxes do not align with the visible image.

Notes
- The previous `NDV_LocalMonitor/` module is deprecated; use `dashboard/app.py` exclusively.
- Streams are unified to 640x480. No Logout/Snapshot in the UI.

Storage & cleanup
- SQLite logs at `data/logs/logs.db`; images saved under `data/logs/YYYY-MM-DD/`.
- Background cleanup deletes normal (non‑abnormal) logs older than 2 hours and removes associated images.

Troubleshooting
- Camera offline: ensure `libcamera-hello -t 2000` works; install `python3-picamera2` and reboot.
- BME680 offline: check I2C wiring; we try addresses `0x76` then `0x77`.
