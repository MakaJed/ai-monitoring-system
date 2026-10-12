# HenSight Project
AI-Powered and Sensor-Based Monitoring System for Early Detection of Newcastle Disease Virus (NDV) in Poultry Chickens

An edge-computing, automated surveillance solution combining computer vision, infrared thermal imaging, and environmental sensing to support early detection of Newcastle Disease Virus (NDV) symptoms in poultry flocks[cite: 6].

---

## 📖 Overview

**HenSight** is an intelligent monitoring platform designed for small- to medium-scale poultry farms to provide continuous, non-invasive surveillance[cite: 6]. By deploying YOLOv8-based computer vision optimized with TensorFlow Lite on a Raspberry Pi 4, the system tracks individual chickens, identifies visible indicators such as twisted necks and abnormal postures, measures body surface temperatures via an MLX90640 thermal sensor, and evaluates ambient coop conditions (temperature, humidity, pressure, and air quality) using a BME680 sensor[cite: 6]. Operating fully offline via a local Wi-Fi hotspot, it provides real-time data visualization, automated alerts, and timestamped historical logs through a Flask-powered web dashboard[cite: 6].

---

## ✨ Key Features

- **Real-Time AI Detection:** YOLOv8-based object detection model optimized with TensorFlow Lite (INT8 quantization) to identify chickens and flag visible symptoms like twisted necks and unnatural postures[cite: 6].
- **Non-Invasive Thermal Monitoring:** Utilizes an MLX90640 infrared thermal sensor to monitor surface temperatures and approximate body heat to detect early fever or heat stress[cite: 6].
- **Environmental Sensing:** Tracks coop temperature, humidity, atmospheric pressure, and indoor air quality (IAQ) via a BME680 sensor[cite: 6].
- **Centralized Web Dashboard:** Flask-powered web interface accessible via mobile or desktop over a local Wi-Fi hotspot, featuring live regular and thermal camera streams, alert notifications, and weekly summary reports[cite: 6].
- **Local Data Logging & Offline Operation:** SQLite database backend stores timestamped logs and snapshots locally, ensuring full functionality without internet dependency[cite: 6].

---

#### 📸 System Screenshots & Hardware Setup

| Hardware Enclosure Setup | Real-Time Testing & Monitoring |
|---|---|
| <img src="Hensightpics (2)_2.png" width="350"> | <img src="Hensightpics (1).jpg" width="350"> |

| Web Dashboard (Live Camera & Thermal Feeds) | Historical Logs & Data Analytics |
|---|---|
| <img src="Hensightpics (3).png" width="350"> | <img src="Hensightpics (1).png" width="350"> |

| Environmental Analytics Chart | Mobile Interface |
|---|---|
| <img src="Hensightpics (2).jpg" width="350"> | <img src="Hensightpics (3).jpg" width="350"> |

---

## 🛠️ Project Layout

- `dashboard/app.py` (Flask backend and endpoints)
- `dashboard/templates/index.html` (UI)
- `dashboard/static/css/style.css`, `dashboard/static/js/app.js`
- `models/chickenmodel.tflite`, `models/chicken.labels.txt`
- `systemd/ndv-dashboard.service` (autostart service)

---

## ⚙️ Prerequisites & Installation

### Prerequisites (Raspberry Pi OS 64-bit)
```bash
sudo raspi-config  # enable Camera and I2C
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv python3-numpy python3-psutil libatlas-base-dev python3-picamera2

Python deps (Python 3.11)   
Bash
cd /home/pi/NDV_Monitoring
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
Model files   
Put your chicken model at models/chickenmodel.tflite (labels in models/chicken.labels.txt). Optional: models/ndvsymptoms.tflite + models/symptoms.labels.txt.   

Run (dev)   
Bash
python3 dashboard/app.py
# open http://<pi-ip>:5000
Install as a service   
Bash
sudo cp systemd/ndv-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ndv-dashboard
sudo systemctl restart ndv-dashboard
sudo systemctl status ndv-dashboard --no-pager
📡 Endpoints   
/camera_feed – Visible MJPEG   

/thermal_feed – Thermal MJPEG (with visible boxes overlaid)   

/sensor_data – BME680 JSON (status, temperature_c, humidity_pct, gas_ohms, pressure_hpa)   

/detect – Runs YOLOv8-TFLite, returns detections and updates overlay   

/analytics – Aggregated daily metrics from CSV logs   

/logs – Logs & Analytics UI (table + summaries)   

/api/logs – Paginated logs from SQLite   

/api/summaries – Daily/weekly/monthly summaries

/status, /notifications

🔧 Calibration & Configuration
Adjust CALIBRATION_OFFSET_X and CALIBRATION_OFFSET_Y in dashboard/app.py if thermal overlay boxes do not align with the visible image.

The previous NDV_LocalMonitor/ module is deprecated; use dashboard/app.py exclusively.

Streams are unified to 640x480. No Logout/Snapshot in the UI.

🗄️ Storage & Cleanup
SQLite logs at data/logs/logs.db; images saved under data/logs/YYYY-MM-DD/.

Background cleanup deletes normal (non‑abnormal) logs older than 2 hours and removes associated images.

🛠️ Troubleshooting
Camera offline: ensure libcamera-hello -t 2000 works; install python3-picamera2 and reboot.

BME680 offline: check I2C wiring; we try addresses 0x76 then 0x77.
