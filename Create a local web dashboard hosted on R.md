Perfect 💡— that’s exactly the kind of detailed UI structure a Cursor-based web dashboard prompt needs.
Now that I have your full layout and requirements (thermal + BME + camera + model + analytics + offline dashboard), here’s your **complete Cursor prompt** for the **NDV Poultry Monitor (Raspberry Pi 4, Legacy 64-bit, Full Desktop)** build.

This will generate the **front-end dashboard** + **Flask backend** setup, all **offline-capable**, served via the Pi’s hotspot (`192.168.4.1` or `raspberrypi.local`).

---

## 🧠 Prompt for Cursor — NDV Poultry Monitor (Offline Web Dashboard)

**Goal:**
Build an offline-capable **Flask-based web dashboard** for the **NDV Poultry Monitor**, designed for **Raspberry Pi 4 (8GB, Legacy 64-bit)**.
The system integrates:

* **NoIR Camera V2 (IMX219)** → Live feed + YOLOv8-TFLite detection
* **MLX90640BAB Thermal Sensor** → 32×24 array with hotspot mapping
* **BME680 Sensor** → Temperature, humidity, gas, and pressure readings
* **Local web dashboard** (dark mode, fully offline, responsive)
* **Daily summaries + real-time alerts**
* **Hostapd hotspot access** (`192.168.4.1` / `raspberrypi.local`)
* **No auto-disconnects** from hotspot

---

### 🧩 Functional Overview

#### 1️⃣ Local Backend

* Flask backend running on `http://0.0.0.0:5000`
* Auto-start on boot (systemd service)
* Endpoints:

  * `/camera_feed` – Streams live PiCam (NoIR)
  * `/thermal_feed` – Streams MLX90640 as color-mapped JPEG
  * `/sensor_data` – JSON output from BME680 (temp, humidity, gas, pressure)
  * `/detect` – YOLOv8-TFLite model inference on latest PiCam frame
  * `/analytics` – Serve daily logs (CSV → JSON summary)
  * `/logs` – Logs & Analytics UI page
  * `/api/logs` – Paginated logs from SQLite
  * `/api/summaries` – Daily/weekly/monthly summaries
  * `/status` – System metrics (CPU, RAM, uptime, etc.)
  * `/notifications` – Returns alerts (e.g., “Hotspot detected”, “Eye Discharge Found”)
* Hardware detection:

  * If any module (PiCam, MLX90640, BME680) is missing → return `"sensor/cam offline"` instead of simulating data.

---

#### 2️⃣ TFLite Integration

* Chicken detector: `/models/chickenmodel.tflite` with `/models/chicken.labels.txt` (single label `Chicken`)
* Optional symptoms: `/models/ndvsymptoms.tflite` with `/models/symptoms.labels.txt`
* Use `tflite_runtime.interpreter` for inference.
* Align bounding boxes between **visible** and **thermal** frames by calibration offset.
* Detect **hotspots**:

  ```
  if thermal_pixel > ambient_avg + HOTSPOT_DELTA:
      trigger alert “Hotspot detected”
  ```
* Display bounding boxes and alert overlays on the visible camera stream.

---

#### 3️⃣ Dashboard UI Design (Matches Your Layout)

##### Navigation Bar

* Fixed dark top bar
* Left: “**NDV Poultry Monitor**”
* Subtext: “Raspberry Pi + OpenCV Chicken Health”
* Right: Buttons for Dashboard / Logs & Analytics, Notification bell with badge, Online/Offline indicator

##### System Status Strip

* Horizontal line below nav bar
* Displays live system stats (CPU %, Temp, RAM %, Uptime) with small colored icons

##### Main Grid

* **Left Panel:**

  * Title: Active camera type (“Visible Feed” / “Thermal View”)
  * Camera switch button
  * Live stream in bordered frame
  * Overlay bounding boxes + alert highlights
  * Split View button to show visible + thermal side by side
  * Alert banner above feed

* **Right Panel:**

  * Stack of sensor cards:

    * Temperature
    * Humidity
    * Gas Resistance
    * Pressure
  * Each card: rounded corner, small badge (“Normal”, “Warning”, “Offline”)
  * Header: “All Sensors Online” summary

##### Bottom Analytics Section

* Title: “Historical Data & Analytics”
* Tabs for:

  * Temperature
  * Humidity
  * Air Quality
  * Alerts
* Graph area: dark theme, thin grid lines, line or bar chart
* Legend below chart
* Data from `/analytics` endpoint (daily summaries)

##### Color/Theme

| Element         | Color/Style                           |
| --------------- | ------------------------------------- |
| Background      | `#121212` (deep gray)                 |
| Text            | `#FFFFFF` or light gray               |
| Accent (Normal) | Green                                 |
| Warning         | Yellow                                |
| Alert           | Red                                   |
| Panels          | Rounded corners + subtle shadow       |
| Font            | Sans-serif, modern, legible           |
| Layout          | Responsive flex/grid, mobile-friendly |

---

#### 4️⃣ Hotspot Configuration (Offline Access)

Auto-start hotspot on boot:

* SSID: `NDV_Monitor_Hotspot`
* Password: `ndvmonitor123`
* IP: `192.168.4.1`
* DNS/DHCP enabled for clients
* Flask served via:

  ```
  http://192.168.4.1:5000
  or
  http://raspberrypi.local:5000
  ```

Ensure:

* Pi stays connected to its own hotspot
* Disable power saving on Wi-Fi to prevent disconnection

---

#### 5️⃣ Local Data Handling

* CSV saved to `/home/pi/NDV_Monitoring/logs/YYYY-MM-DD.csv`
* SQLite DB at `data/logs/logs.db` with images under `data/logs/YYYY-MM-DD/`
* Background cleanup deletes normal logs older than 2 hours
* Daily analytics summarize:

  * Avg temperature/humidity
  * Number of detections per class
  * Max hotspot readings
* Alerts are saved and displayed on dashboard bell icon

---

### 🧠 Required Python Packages

```bash
sudo apt install python3-flask python3-opencv python3-numpy python3-matplotlib python3-pandas python3-pil python3-picamera2
pip install adafruit-circuitpython-mlx90640 adafruit-circuitpython-bme680 tflite-runtime==2.14.0
```

---

### ⚙️ Run Command

```bash
python3 /home/pi/NDV_Monitoring/dashboard/app.py
```

---

### ✅ Deliverables for Cursor

* `/dashboard/app.py` → Flask backend + camera/sensor model integration
* `/dashboard/templates/index.html` → full dark UI layout
* `/dashboard/static/css/style.css` and `/static/js/app.js`
* `/models/best.tflite` and `/models/labels.txt`
* `/systemd/ndv-dashboard.service` → autostart service

---

Would you like me to include a **Flask folder structure layout** (so Cursor can generate it ready to deploy), or just leave this as the full prompt for you to paste directly into Cursor?
