#!/usr/bin/env python3
"""Thermal sensor diagnostic and troubleshooting script."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import board
    import busio
    import time
    import adafruit_mlx90640
    print("✓ All libraries imported successfully")
except ImportError as e:
    print(f"✗ Library import failed: {e}")
    sys.exit(1)

print("\n=== MLX90640 Thermal Sensor Diagnostics ===\n")

# 1. Check I2C pins
print("1. I2C Configuration:")
print(f"   SCL pin: {board.SCL}")
print(f"   SDA pin: {board.SDA}")
print("   Expected: SCL=GPIO3, SDA=GPIO2\n")

# 2. Test I2C bus
print("2. Testing I2C bus...")
try:
    i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
    print("   ✓ I2C bus created")
    time.sleep(0.5)
    
    # Scan for devices
    print("\n3. Scanning I2C bus for devices...")
    devices = []
    try:
        while not i2c.try_lock():
            pass
        
        for addr in range(0x08, 0x78):
            try:
                result = bytearray(1)
                i2c.writeto_then_readfrom(addr, result, result)
                devices.append(hex(addr))
                print(f"   ✓ Device found at {hex(addr)}")
            except Exception:
                pass
        
        i2c.unlock()
        
        if '0x33' in devices:
            print("\n   ✓ MLX90640 detected at 0x33!")
        elif '0x76' in devices:
            print("\n   ⚠ BME680 detected at 0x76 (environmental sensor)")
            print("   ✗ MLX90640 NOT detected at 0x33")
        else:
            print("\n   ✗ No expected devices found")
            
        if not devices:
            print("\n   ✗ No I2C devices found at all!")
            print("   Troubleshooting:")
            print("     - Check I2C is enabled: sudo raspi-config → Interface Options → I2C")
            print("     - Check wiring connections")
            print("     - Verify sensors are powered")
    except Exception as e:
        print(f"   ✗ I2C scan failed: {e}")
    
    i2c.deinit()
except Exception as e:
    print(f"   ✗ I2C bus creation failed: {e}")
    sys.exit(1)

# 3. Try to initialize MLX90640
print("\n4. Attempting MLX90640 initialization...")
try:
    i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
    time.sleep(0.5)
    
    sensor = adafruit_mlx90640.MLX90640(i2c)
    print("   ✓ Sensor object created")
    
    sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
    print("   ✓ Refresh rate set to 8 Hz")
    time.sleep(1.0)
    
    # Test read
    print("\n5. Testing frame read...")
    frame = [0.0] * 768
    sensor.getFrame(frame)
    
    max_temp = max(frame)
    min_temp = min(frame)
    avg_temp = sum(frame) / len(frame)
    
    print(f"   ✓ Frame read successful!")
    print(f"   Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C")
    print(f"   Average: {avg_temp:.1f}°C")
    
    if max_temp > 10.0 and min_temp >= -40.0:
        print("\n   ✓✓✓ THERMAL SENSOR IS WORKING! ✓✓✓")
    else:
        print("\n   ⚠ Temperature values seem invalid")
        
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    print("\n   Troubleshooting steps:")
    print("   1. Verify MLX90640 is connected:")
    print("      - VDD → 3.3V")
    print("      - GND → Ground")
    print("      - SCL → GPIO3 (pin 5)")
    print("      - SDA → GPIO2 (pin 3)")
    print("   2. Check power supply (sensor needs stable 3.3V)")
    print("   3. Verify I2C is enabled: sudo raspi-config")
    print("   4. Check I2C permissions: groups (should include 'i2c')")
    print("   5. Try: sudo i2cdetect -y 1 (should show 0x33)")
    import traceback
    traceback.print_exc()

print("\n=== End Diagnostics ===")

