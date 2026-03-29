"""Hardware monitor — GPU temp/util/VRAM/power via amdgpu_top, psutil CPU, NPU status.

Background thread, 500ms updates. All metrics available as a snapshot dict.
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GpuMetrics:
    temp_c: float = 0.0
    util_pct: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    power_w: float = 0.0
    clock_mhz: int = 0
    throttling: bool = False


@dataclass
class CpuMetrics:
    util_pct: float = 0.0
    freq_mhz: float = 0.0
    temp_c: float = 0.0
    load_avg: tuple = (0.0, 0.0, 0.0)


@dataclass
class NpuMetrics:
    present: bool = False
    device_path: str = ""
    driver: str = ""
    status: str = "unknown"


@dataclass
class SystemSnapshot:
    timestamp: float = 0.0
    gpu: GpuMetrics = field(default_factory=GpuMetrics)
    cpu: CpuMetrics = field(default_factory=CpuMetrics)
    npu: NpuMetrics = field(default_factory=NpuMetrics)
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "gpu": {
                "temp_c": self.gpu.temp_c,
                "util_pct": self.gpu.util_pct,
                "vram_used_mb": self.gpu.vram_used_mb,
                "vram_total_mb": self.gpu.vram_total_mb,
                "power_w": self.gpu.power_w,
                "clock_mhz": self.gpu.clock_mhz,
                "throttling": self.gpu.throttling,
            },
            "cpu": {
                "util_pct": self.cpu.util_pct,
                "freq_mhz": self.cpu.freq_mhz,
                "temp_c": self.cpu.temp_c,
                "load_avg": list(self.cpu.load_avg),
            },
            "npu": {
                "present": self.npu.present,
                "device_path": self.npu.device_path,
                "driver": self.npu.driver,
                "status": self.npu.status,
            },
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
        }


class HardwareMonitor:
    """Background hardware monitor with 500ms polling."""

    def __init__(self, interval_ms: int = 500):
        self._interval = interval_ms / 1000.0
        self._snapshot = SystemSnapshot()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._npu_checked = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def snapshot(self) -> SystemSnapshot:
        with self._lock:
            return self._snapshot

    def _poll_loop(self):
        while not self._stop_event.is_set():
            snap = SystemSnapshot(timestamp=time.time())
            self._read_gpu(snap.gpu)
            self._read_cpu(snap.cpu)
            self._read_memory(snap)
            if not self._npu_checked:
                self._read_npu(snap.npu)
                self._npu_checked = True
            else:
                with self._lock:
                    snap.npu = self._snapshot.npu
            with self._lock:
                self._snapshot = snap
            self._stop_event.wait(self._interval)

    def _read_gpu(self, gpu: GpuMetrics):
        # Try amdgpu_top JSON output (single sample)
        try:
            result = subprocess.run(
                ["amdgpu_top", "--apu", "--no-pc", "-J", "-n", "1"],
                capture_output=True, text=True, timeout=2.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if isinstance(data, list) and data:
                    data = data[0]
                devices = data.get("devices", data.get("gpu", []))
                if isinstance(devices, list) and devices:
                    dev = devices[0]
                elif isinstance(devices, dict):
                    dev = devices
                else:
                    dev = data
                # Temperature
                sensors = dev.get("Sensors", dev.get("sensors", {}))
                if isinstance(sensors, dict):
                    gpu.temp_c = sensors.get("edge", sensors.get("junction", sensors.get("Temperature", 0)))
                    if isinstance(gpu.temp_c, dict):
                        gpu.temp_c = gpu.temp_c.get("value", 0)
                # Power
                gpu.power_w = sensors.get("power", sensors.get("Power", 0))
                if isinstance(gpu.power_w, dict):
                    gpu.power_w = gpu.power_w.get("value", 0)
                # Utilization via GRBM
                grbm = dev.get("GRBM", dev.get("grbm", {}))
                if isinstance(grbm, dict):
                    gpu.util_pct = grbm.get("Graphics Pipe", grbm.get("gui", 0))
                # VRAM
                vram = dev.get("VRAM", dev.get("vram", {}))
                if isinstance(vram, dict):
                    total = vram.get("Total VRAM", vram.get("total", {}))
                    used = vram.get("Total VRAM Usage", vram.get("used", {}))
                    if isinstance(total, dict):
                        gpu.vram_total_mb = total.get("value", 0)
                    elif isinstance(total, (int, float)):
                        gpu.vram_total_mb = total
                    if isinstance(used, dict):
                        gpu.vram_used_mb = used.get("value", 0)
                    elif isinstance(used, (int, float)):
                        gpu.vram_used_mb = used
                return
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

        # Fallback: sysfs
        self._read_gpu_sysfs(gpu)

    def _read_gpu_sysfs(self, gpu: GpuMetrics):
        hwmon_base = Path("/sys/class/drm/card1/device/hwmon")
        if not hwmon_base.exists():
            hwmon_base = Path("/sys/class/drm/card0/device/hwmon")
        if hwmon_base.exists():
            hwmon_dirs = list(hwmon_base.iterdir())
            if hwmon_dirs:
                hwmon = hwmon_dirs[0]
                try:
                    gpu.temp_c = int((hwmon / "temp1_input").read_text().strip()) / 1000
                except (FileNotFoundError, ValueError):
                    pass
                try:
                    gpu.power_w = int((hwmon / "power1_average").read_text().strip()) / 1_000_000
                except (FileNotFoundError, ValueError):
                    pass
        gpu_busy = Path("/sys/class/drm/card1/device/gpu_busy_percent")
        if not gpu_busy.exists():
            gpu_busy = Path("/sys/class/drm/card0/device/gpu_busy_percent")
        if gpu_busy.exists():
            try:
                gpu.util_pct = float(gpu_busy.read_text().strip())
            except ValueError:
                pass

    def _read_cpu(self, cpu: CpuMetrics):
        # Load average
        try:
            cpu.load_avg = os.getloadavg()
        except OSError:
            pass
        # CPU frequency from /proc/cpuinfo
        try:
            with open("/proc/cpuinfo") as f:
                freqs = []
                for line in f:
                    if line.startswith("cpu MHz"):
                        freqs.append(float(line.split(":")[1].strip()))
                if freqs:
                    cpu.freq_mhz = sum(freqs) / len(freqs)
        except (FileNotFoundError, ValueError):
            pass
        # CPU utilization from /proc/stat
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()
                if parts[0] == "cpu":
                    vals = [int(x) for x in parts[1:]]
                    idle = vals[3] if len(vals) > 3 else 0
                    total = sum(vals)
                    if total > 0:
                        cpu.util_pct = 100.0 * (1.0 - idle / total)
        except (FileNotFoundError, ValueError, IndexError):
            pass
        # CPU temp
        for hwmon in Path("/sys/class/hwmon").iterdir():
            try:
                name = (hwmon / "name").read_text().strip()
                if name in ("k10temp", "zenpower"):
                    cpu.temp_c = int((hwmon / "temp1_input").read_text().strip()) / 1000
                    break
            except (FileNotFoundError, ValueError):
                continue

    def _read_npu(self, npu: NpuMetrics):
        accel = Path("/dev/accel/accel0")
        npu.present = accel.exists()
        if npu.present:
            npu.device_path = str(accel)
            # Check driver
            uevent = Path("/sys/class/accel/accel0/device/uevent")
            if uevent.exists():
                try:
                    for line in uevent.read_text().splitlines():
                        if line.startswith("DRIVER="):
                            npu.driver = line.split("=", 1)[1]
                except (FileNotFoundError, PermissionError):
                    pass
            npu.status = "available" if npu.driver else "no_driver"

    def _read_memory(self, snap: SystemSnapshot):
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
                snap.mem_total_mb = info.get("MemTotal", 0) / 1024
                snap.mem_used_mb = snap.mem_total_mb - info.get("MemAvailable", 0) / 1024
        except (FileNotFoundError, ValueError):
            pass
