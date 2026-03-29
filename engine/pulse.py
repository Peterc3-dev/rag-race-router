"""Pulsed GPU execution controller.

Adapts burst duration to thermal headroom. The GPU fires in short pulses
(default 50ms) with cooldown gaps (default 10ms), backing off as temperature
approaches the ceiling. This prevents thermal throttling and maintains
consistent throughput.
"""

import time
from dataclasses import dataclass


@dataclass
class PulseConfig:
    gpu_burst_ms: float = 50.0
    cooldown_ms: float = 10.0
    temp_ceiling: float = 80.0
    temp_floor: float = 60.0
    min_burst_ms: float = 5.0
    max_burst_ms: float = 200.0


class PulseController:
    """Controls GPU burst/cooldown cycles based on thermal headroom."""

    def __init__(self, config: PulseConfig = None):
        self.config = config or PulseConfig()
        self._last_fire = 0.0
        self._burst_count = 0
        self._total_burst_ms = 0.0
        self._total_cool_ms = 0.0

    def thermal_budget(self, current_temp: float) -> float:
        """Returns 0.0-1.0 indicating thermal headroom.

        1.0 = at floor temp (full headroom)
        0.0 = at or above ceiling (no headroom)
        """
        if current_temp <= self.config.temp_floor:
            return 1.0
        if current_temp >= self.config.temp_ceiling:
            return 0.0
        span = self.config.temp_ceiling - self.config.temp_floor
        return 1.0 - (current_temp - self.config.temp_floor) / span

    def effective_burst_ms(self, current_temp: float) -> float:
        """Burst duration scaled by thermal budget."""
        budget = self.thermal_budget(current_temp)
        burst_range = self.config.max_burst_ms - self.config.min_burst_ms
        return self.config.min_burst_ms + burst_range * budget

    def effective_cooldown_ms(self, current_temp: float) -> float:
        """Cooldown duration — inverse of thermal budget."""
        budget = self.thermal_budget(current_temp)
        # More cooldown when hotter
        return self.config.cooldown_ms * (1.0 + 2.0 * (1.0 - budget))

    def should_fire_gpu(self, current_temp: float) -> bool:
        """Whether the GPU should accept a new burst right now."""
        if current_temp >= self.config.temp_ceiling:
            return False

        now = time.monotonic()
        cooldown_s = self.effective_cooldown_ms(current_temp) / 1000.0
        if now - self._last_fire < cooldown_s:
            return False

        return True

    def record_burst(self, duration_ms: float):
        """Record that a GPU burst completed."""
        self._last_fire = time.monotonic()
        self._burst_count += 1
        self._total_burst_ms += duration_ms

    def record_cooldown(self, duration_ms: float):
        """Record cooldown time."""
        self._total_cool_ms += duration_ms

    @property
    def duty_cycle(self) -> float:
        """Fraction of time spent in bursts vs total (burst + cooldown)."""
        total = self._total_burst_ms + self._total_cool_ms
        if total == 0:
            return 0.0
        return self._total_burst_ms / total

    @property
    def stats(self) -> dict:
        return {
            "burst_count": self._burst_count,
            "total_burst_ms": round(self._total_burst_ms, 2),
            "total_cool_ms": round(self._total_cool_ms, 2),
            "duty_cycle": round(self.duty_cycle, 4),
        }
