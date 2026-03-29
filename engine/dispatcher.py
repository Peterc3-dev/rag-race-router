"""Workload dispatcher — profiles operations per device, builds routing table.

After N runs, encodes dispatch patterns as lightweight rules. Re-evaluates
when exceptions occur (thermal spike, device error, shape mismatch).
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .personality import Personality
from .monitor import HardwareMonitor, SystemSnapshot
from .pulse import PulseController


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


@dataclass
class DispatchDecision:
    device: Device
    reason: str
    confidence: float = 0.0
    fallback: Optional[Device] = None


@dataclass
class OpProfile:
    """Runtime profile for an operation on a specific device."""
    device: Device
    operation: str
    avg_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    count: int = 0
    failures: int = 0
    last_temp: float = 0.0

    def update(self, duration_ms: float, temp: float = 0.0):
        self.count += 1
        self.last_temp = temp
        if self.count == 1:
            self.avg_ms = duration_ms
        else:
            self.avg_ms = self.avg_ms + (duration_ms - self.avg_ms) / self.count
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


class Dispatcher:
    """Routes operations to the optimal device based on learned profiles."""

    LEARNING_THRESHOLD = 10  # Runs before encoding rules

    def __init__(
        self,
        monitor: HardwareMonitor,
        personality: Personality,
        pulse: PulseController,
        gpu_available: bool = True,
        npu_available: bool = False,
    ):
        self.monitor = monitor
        self.personality = personality
        self.pulse = pulse
        self.gpu_available = gpu_available
        self.npu_available = npu_available
        self._profiles: dict[tuple[str, str], OpProfile] = {}
        self._total_dispatches = 0
        self._rules_encoded = False

    def dispatch(self, operation: str, input_size: int = 0) -> DispatchDecision:
        """Decide which device should handle this operation."""
        snap = self.monitor.snapshot
        self._total_dispatches += 1

        # Phase 1: Use personality rules if available
        if self._rules_encoded or self._total_dispatches > self.LEARNING_THRESHOLD:
            suggested = self.personality.suggest(
                operation, temp=snap.gpu.temp_c, input_size=input_size
            )
            device = Device(suggested)

            # Validate: is the suggested device actually usable right now?
            if device == Device.GPU and not self._gpu_usable(snap):
                device = Device.CPU
                return DispatchDecision(
                    device=device,
                    reason=f"personality suggests gpu but thermal/availability override → cpu",
                    confidence=0.7,
                    fallback=Device.CPU,
                )

            return DispatchDecision(
                device=device,
                reason=f"personality rule: {operation} → {device.value}",
                confidence=0.8,
            )

        # Phase 0: Heuristic routing during learning period
        return self._heuristic_dispatch(operation, input_size, snap)

    def _heuristic_dispatch(
        self, operation: str, input_size: int, snap: SystemSnapshot
    ) -> DispatchDecision:
        """Simple heuristics before enough data for learned rules."""

        # Large matrix ops → GPU if available and cool enough
        if operation in ("matmul", "conv2d", "attention", "gemm"):
            if self.gpu_available and self._gpu_usable(snap):
                return DispatchDecision(
                    device=Device.GPU,
                    reason=f"heuristic: {operation} is compute-heavy → gpu",
                    confidence=0.5,
                    fallback=Device.CPU,
                )

        # Lightweight normalization → CPU (low overhead, not worth GPU dispatch)
        if operation in ("layernorm", "rmsnorm", "softmax", "relu"):
            return DispatchDecision(
                device=Device.CPU,
                reason=f"heuristic: {operation} is lightweight → cpu",
                confidence=0.6,
            )

        # NPU for quantized inference if available
        if operation in ("quantized_matmul", "int8_gemm") and self.npu_available:
            return DispatchDecision(
                device=Device.NPU,
                reason=f"heuristic: {operation} is quantized → npu",
                confidence=0.5,
                fallback=Device.CPU,
            )

        # Default to CPU
        return DispatchDecision(
            device=Device.CPU,
            reason=f"heuristic: default → cpu",
            confidence=0.3,
        )

    def _gpu_usable(self, snap: SystemSnapshot) -> bool:
        """Check if GPU is available and within thermal budget."""
        if not self.gpu_available:
            return False
        if not self.pulse.should_fire_gpu(snap.gpu.temp_c):
            return False
        return True

    def record_result(
        self,
        device: Device,
        operation: str,
        duration_ms: float,
        input_size: int = 0,
        success: bool = True,
    ):
        """Record execution result to improve future dispatching."""
        snap = self.monitor.snapshot
        key = (device.value, operation)

        if key not in self._profiles:
            self._profiles[key] = OpProfile(device=device, operation=operation)
        self._profiles[key].update(duration_ms, snap.gpu.temp_c)

        if not success:
            self._profiles[key].failures += 1

        self.personality.record_run(
            device=device.value,
            operation=operation,
            duration_ms=duration_ms,
            input_size=input_size,
            temp_before=snap.gpu.temp_c,
            power_w=snap.gpu.power_w,
            success=success,
        )

        # Check if we should encode rules
        total_profile_runs = sum(p.count for p in self._profiles.values())
        if not self._rules_encoded and total_profile_runs >= self.LEARNING_THRESHOLD * 3:
            self.personality.update_rules()
            self._rules_encoded = True

    def force_reencode(self):
        """Force re-evaluation of routing rules (e.g., after driver update)."""
        self.personality.update_rules()
        self._rules_encoded = True

    @property
    def stats(self) -> dict:
        return {
            "total_dispatches": self._total_dispatches,
            "rules_encoded": self._rules_encoded,
            "profiles": {
                f"{k[0]}/{k[1]}": {
                    "avg_ms": round(p.avg_ms, 2),
                    "count": p.count,
                    "failures": p.failures,
                }
                for k, p in self._profiles.items()
            },
        }
