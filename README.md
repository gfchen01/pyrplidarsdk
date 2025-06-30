# PyRPLIDARSDK

Python wrapper for the [Slamtec RPLIDAR SDK](https://github.com/Slamtec/rplidar_sdk) using nanobind.

This package provides a high-performance Python interface to RPLIDAR series laser range scanners, supporting both serial and UDP connections.

## Installation

### From PyPI (coming soon)

```bash
pip install pyrplidarsdk
```

### From Source

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/dexmate-ai/pyrplidarsdk.git
cd pyrplidarsdk

# Install in development mode
pip install -e .
```

## Quick Start

### Serial Connection

```python
import pyrplidarsdk
import time

# Create driver instance for serial connection
driver = pyrplidarsdk.RplidarDriver(port="/dev/ttyUSB0")

# Connect to the device
if not driver.connect():
    print("Failed to connect!")
    exit(1)

# Get device information
info = driver.get_device_info()
if info:
    print(f"Connected to RPLIDAR model {info.model}")
    print(f"Firmware: {info.firmware_version}")
    print(f"Hardware: {info.hardware_version}")
    print(f"Serial: {info.serial_number}")

# Check device health
health = driver.get_health()
if health:
    print(f"Device health: {health.status}")

# Start scanning
if driver.start_scan():
    print("Scanning started...")
    
    try:
        for i in range(10):  # Get 10 scans
            scan_data = driver.get_scan_data()
            if scan_data:
                angles, ranges, qualities = scan_data
                print(f"Scan {i+1}: {len(angles)} points")
            time.sleep(0.1)
    finally:
        driver.stop_scan()
        driver.disconnect()
```

### UDP Connection

```python
import pyrplidarsdk

# Create driver instance for UDP connection
driver = pyrplidarsdk.RplidarDriver(ip_address="192.168.1.100")

# Rest of the code is the same...
```

## API Reference

### RplidarDriver

Main class for controlling RPLIDAR devices.

#### Constructor

```python
RplidarDriver(port=None, ip_address=None, baudrate=1000000, udp_port=8089)
```

- `port` (str, optional): Serial port path (e.g., "/dev/ttyUSB0", "COM3")
- `ip_address` (str, optional): IP address for UDP connection
- `baudrate` (int): Serial baudrate (default: 1000000)
- `udp_port` (int): UDP port (default: 8089)

#### Methods

- `connect() -> bool`: Connect to the device
- `disconnect()`: Disconnect from the device
- `get_device_info() -> DeviceInfo | None`: Get device information
- `get_health() -> DeviceHealth | None`: Get device health status
- `start_scan() -> bool`: Start laser scanning
- `stop_scan()`: Stop laser scanning  
- `get_scan_data() -> tuple[list[float], list[float], list[int]] | None`: Get scan data

### DeviceInfo

Device information structure.

- `model` (int): Device model number
- `firmware_version` (int): Firmware version
- `hardware_version` (int): Hardware version  
- `serial_number` (str): Device serial number (hex string)

### DeviceHealth

Device health information.

- `status` (int): Health status code
- `error_code` (int): Error code if any

## Utility Functions

The package includes utility functions in `pyrplidarsdk.utils`:

```python
from pyrplidarsdk.utils import polar_to_cartesian, filter_by_range

# Convert polar to cartesian coordinates
x_coords, y_coords = polar_to_cartesian(angles, ranges)

# Filter by range
filtered_angles, filtered_ranges, filtered_qualities = filter_by_range(
    angles, ranges, qualities, min_range=0.1, max_range=5.0
)
```

## Support

- **Issues**: [GitHub Issues](https://github.com/dexmate-ai/pyrplidarsdk/issues)
- **Email**: contact@dexmate.ai

## Acknowledgments

- [Slamtec](https://www.slamtec.com/) for the RPLIDAR SDK
- [nanobind](https://github.com/wjakob/nanobind) for the excellent Python-C++ binding framework 