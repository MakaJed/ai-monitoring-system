from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional


class AlertLogger:
    def __init__(self, log_file: str) -> None:
        self._path = Path(log_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(self, source: str, message: str) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": source,
            "message": message,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def tail(self, max_lines: int = 100) -> List[Dict[str, Any]]:
        try:
            with self._path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-max_lines:]
            return [json.loads(l) for l in lines]
        except Exception:
            return []


class SummaryManager:
    def __init__(self, summary_file: str) -> None:
        self._path = Path(summary_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._detections: Dict[str, int] = {}
        self._environment_latest: Dict[str, Any] = {}
        # rolling windows for last 24h
        self._env_window: List[Dict[str, Any]] = []  # {ts, temperature_c, humidity_percent, pressure_hpa}
        self._thermal_window: List[Dict[str, Any]] = []  # {ts, avg_temp, hotspots}
        self._last_write = time.time()
        self._stop_event = threading.Event()

    def add_detections(self, detections: List[Dict[str, Any]]) -> None:
        with self._lock:
            for det in detections:
                label = det.get("label")
                if not label:
                    continue
                self._detections[label] = self._detections.get(label, 0) + 1

    def update_environment(self, env: Dict[str, Any]) -> None:
        if not env:
            return
        with self._lock:
            self._environment_latest = env
            sample = {
                "ts": time.time(),
                "temperature_c": env.get("temperature_c"),
                "humidity_percent": env.get("humidity_percent"),
                "pressure_hpa": env.get("pressure_hpa"),
            }
            self._env_window.append(sample)
            self._prune_24h_locked()

    def update_thermal(self, avg_temp: float, hotspots: int) -> None:
        with self._lock:
            self._thermal_window.append({
                "ts": time.time(),
                "avg_temp": avg_temp,
                "hotspots": int(hotspots),
            })
            self._prune_24h_locked()

    def _prune_24h_locked(self) -> None:
        cutoff = time.time() - 24 * 3600
        self._env_window = [s for s in self._env_window if s.get("ts", 0) >= cutoff]
        self._thermal_window = [s for s in self._thermal_window if s.get("ts", 0) >= cutoff]

    def _compose_summary(self) -> Dict[str, Any]:
        now = datetime.utcnow()
        with self._lock:
            # environment averages over 24h
            temps = [s["temperature_c"] for s in self._env_window if s.get("temperature_c") is not None]
            hums = [s["humidity_percent"] for s in self._env_window if s.get("humidity_percent") is not None]
            press = [s["pressure_hpa"] for s in self._env_window if s.get("pressure_hpa") is not None]

            def _avg(vals):
                return float(sum(vals) / len(vals)) if vals else None

            env_summary = {
                "avg_temp": _avg(temps),
                "avg_humidity": _avg(hums),
                "avg_pressure": _avg(press),
            }

            # thermal aggregation over 24h
            t_avgs = [s["avg_temp"] for s in self._thermal_window if s.get("avg_temp") is not None]
            t_hotspots = [s["hotspots"] for s in self._thermal_window if s.get("hotspots") is not None]
            thermal_summary = {
                "avg_temp": _avg(t_avgs),
                "max_hotspots": max(t_hotspots) if t_hotspots else 0,
            }

            summary = {
                "date": now.date().isoformat(),
                "detections": self._detections.copy(),
                "thermal": thermal_summary,
                "environment": env_summary,
            }
        return summary

    def write_summary(self) -> None:
        data = self._compose_summary()
        tmp = self._path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
        except Exception:
            # Best-effort write
            pass

    def start_background_writer(self, interval_seconds: int = 60) -> None:
        def _worker():
            while not self._stop_event.is_set():
                self.write_summary()
                self._stop_event.wait(interval_seconds)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def stop_background_writer(self) -> None:
        self._stop_event.set()

    def read_summary(self) -> Dict[str, Any]:
        try:
            with self._path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


