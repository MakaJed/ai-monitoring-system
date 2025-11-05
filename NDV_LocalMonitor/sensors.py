from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any

import numpy as np


class SensorManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mlx = None
        self._bme680 = None
        self._thermal_frame_buffer = [0.0] * 768  # 32x24
        self._init_hardware()

    def _init_hardware(self) -> None:
        try:
            import board
            import busio
            import adafruit_mlx90640
            import adafruit_bme680

            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

            self._mlx = adafruit_mlx90640.MLX90640(i2c)
            self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

            self._bme680 = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x76)
            self._bme680.sea_level_pressure = 1013.25
        except Exception:
            # Hardware not available (e.g., developing on non-Pi host)
            self._mlx = None
            self._bme680 = None

    def status(self) -> Dict[str, Any]:
        return {
            "mlx90640": "online" if self._mlx is not None else "offline",
            "bme680": "online" if self._bme680 is not None else "offline",
        }

    def read_thermal_frame(self) -> Optional[np.ndarray]:
        if self._mlx is None:
            return None
        try:
            import adafruit_mlx90640  # noqa: F401  # ensure dependency present

            success = False
            for _ in range(3):
                try:
                    self._mlx.getFrame(self._thermal_frame_buffer)
                    success = True
                    break
                except RuntimeError:
                    time.sleep(0.02)
            if not success:
                return None
            arr = np.array(self._thermal_frame_buffer, dtype=np.float32).reshape(24, 32)
            return arr
        except Exception:
            return None

    def read_environment(self) -> Dict[str, Any]:
        if self._bme680 is None:
            return {
                "temperature_c": None,
                "humidity_percent": None,
                "pressure_hpa": None,
                "gas_ohms": None,
                "altitude_m": None,
            }
        try:
            env = {
                "temperature_c": float(self._bme680.temperature),
                "humidity_percent": float(self._bme680.humidity),
                "pressure_hpa": float(self._bme680.pressure),
                "gas_ohms": float(self._bme680.gas),
                "altitude_m": float(self._bme680.altitude),
            }
            return env
        except Exception:
            return {
                "temperature_c": None,
                "humidity_percent": None,
                "pressure_hpa": None,
                "gas_ohms": None,
                "altitude_m": None,
            }


