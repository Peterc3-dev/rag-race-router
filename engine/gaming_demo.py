"""Gaming simulation — frame pipeline across CPU/GPU/NPU.

Simulates the compute pattern of a game frame:
1. Game logic (CPU) — physics, AI decisions
2. Geometry processing (GPU) — vertex transforms
3. Shading (GPU) — pixel operations
4. AI upscaling (NPU) — FSR-like upscale 720p -> 1080p
5. Post-processing (CPU) — tone mapping, UI overlay
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class FrameResult:
    frame_id: int
    total_ms: float
    logic_ms: float = 0.0
    geometry_ms: float = 0.0
    shading_ms: float = 0.0
    upscale_ms: float = 0.0
    post_ms: float = 0.0
    gpu_temp: float = 0.0
    devices: dict = field(default_factory=dict)


def game_logic(frame_id: int) -> np.ndarray:
    """CPU: simulate physics — matrix transforms, collision detection."""
    np.random.seed(frame_id)
    # 100 entities, each with 4x4 transform matrix
    entities = np.random.randn(100, 4, 4).astype(np.float32)
    # Apply rotation + translation
    rotation = np.eye(4, dtype=np.float32)
    angle = frame_id * 0.01
    rotation[0, 0] = np.cos(angle)
    rotation[0, 1] = -np.sin(angle)
    rotation[1, 0] = np.sin(angle)
    rotation[1, 1] = np.cos(angle)
    transformed = np.array([e @ rotation for e in entities])
    return transformed


def geometry_processing(vertices: np.ndarray) -> np.ndarray:
    """GPU: vertex transforms — large matrix multiply representing batch transform."""
    # Simulate: project 100 entities * 1000 vertices each through MVP matrix
    batch = np.random.randn(1000, 4).astype(np.float32)
    mvp = np.random.randn(4, 4).astype(np.float32)
    projected = batch @ mvp
    return projected


def shading(geometry: np.ndarray) -> np.ndarray:
    """GPU: pixel shading — batch of operations representing lighting."""
    # Simulate 720p framebuffer (1280x720x3 channels, downsampled for speed)
    pixels = np.random.randn(180, 320, 3).astype(np.float32)  # 1/4 res
    # Lighting: dot product with normal, multiply by albedo
    normals = np.random.randn(180, 320, 3).astype(np.float32)
    light_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(np.sum(normals * light_dir, axis=-1, keepdims=True), 0, 1)
    lit = pixels * intensity
    return lit


def ai_upscale(frame_720p: np.ndarray) -> np.ndarray:
    """NPU: AI upscaling — simulate FSR-like upscale from 720p to 1080p."""
    # Simple bilinear + sharpen as proxy for neural upscaling
    from scipy.ndimage import zoom
    upscaled = zoom(frame_720p, (2, 2, 1), order=1)
    # Sharpen pass
    kernel = np.array([[-0.1, -0.1, -0.1],
                       [-0.1, 1.8, -0.1],
                       [-0.1, -0.1, -0.1]], dtype=np.float32)
    for c in range(min(3, upscaled.shape[2])):
        from scipy.signal import convolve2d
        upscaled[:, :, c] = convolve2d(upscaled[:, :, c], kernel, mode='same', boundary='wrap')
    return np.clip(upscaled, 0, 1)


def post_process(frame: np.ndarray) -> np.ndarray:
    """CPU: tone mapping + UI overlay."""
    # Reinhard tone mapping
    mapped = frame / (1.0 + frame)
    # Gamma correction
    gamma = np.power(np.clip(mapped, 0, 1), 1.0 / 2.2)
    return (gamma * 255).astype(np.uint8)


def run_frame_pipeline(frame_id: int, engine=None, use_engine: bool = True) -> FrameResult:
    """Run one frame through the pipeline."""
    result = FrameResult(frame_id=frame_id, total_ms=0.0)
    t_start = time.perf_counter()

    if engine and use_engine:
        from .executor import WorkItem

        # 1. Game logic → CPU
        t = time.perf_counter()
        work = WorkItem(operation="game_logic", fn=game_logic, args=(frame_id,))
        frag = engine.executor.execute(work)
        transforms = frag.result
        result.logic_ms = (time.perf_counter() - t) * 1000
        result.devices["game_logic"] = frag.device.value

        # 2. Geometry → dispatched (should go to GPU)
        t = time.perf_counter()
        work = WorkItem(operation="matmul", fn=geometry_processing, args=(transforms,),
                        input_size=transforms.size)
        frag = engine.executor.execute(work)
        geometry = frag.result
        result.geometry_ms = (time.perf_counter() - t) * 1000
        result.devices["geometry"] = frag.device.value

        # 3. Shading → dispatched (should go to GPU)
        t = time.perf_counter()
        work = WorkItem(operation="attention", fn=shading, args=(geometry,),
                        input_size=geometry.size)
        frag = engine.executor.execute(work)
        frame_720p = frag.result
        result.shading_ms = (time.perf_counter() - t) * 1000
        result.devices["shading"] = frag.device.value

        # 4. AI upscale → dispatched (should go to NPU)
        t = time.perf_counter()
        work = WorkItem(operation="normalize", fn=ai_upscale, args=(frame_720p,),
                        input_size=frame_720p.size)
        frag = engine.executor.execute(work)
        frame_1080p = frag.result
        result.upscale_ms = (time.perf_counter() - t) * 1000
        result.devices["upscale"] = frag.device.value

        # 5. Post-process → CPU
        t = time.perf_counter()
        work = WorkItem(operation="decode", fn=post_process, args=(frame_1080p,))
        frag = engine.executor.execute(work)
        result.post_ms = (time.perf_counter() - t) * 1000
        result.devices["post"] = frag.device.value

        snap = engine.monitor.snapshot
        result.gpu_temp = snap.gpu.temp_c
    else:
        # All-CPU baseline
        t = time.perf_counter()
        transforms = game_logic(frame_id)
        result.logic_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        geometry = geometry_processing(transforms)
        result.geometry_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        frame_720p = shading(geometry)
        result.shading_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        frame_1080p = ai_upscale(frame_720p)
        result.upscale_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        post_process(frame_1080p)
        result.post_ms = (time.perf_counter() - t) * 1000

    result.total_ms = (time.perf_counter() - t_start) * 1000
    return result


def run_gaming_benchmark(engine, n_frames: int = 100, verbose: bool = False) -> dict:
    """Run N frames and collect statistics."""
    import time as _time

    engine.start()
    _time.sleep(1.0)

    # Warmup
    for i in range(3):
        run_frame_pipeline(i, engine, use_engine=True)

    # Engine-routed frames
    engine_results = []
    for i in range(n_frames):
        r = run_frame_pipeline(i + 3, engine, use_engine=True)
        engine_results.append(r)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames}: {r.total_ms:.1f}ms | GPU {r.gpu_temp:.0f}C")

    # All-CPU baseline
    cpu_results = []
    for i in range(min(n_frames, 50)):  # Fewer CPU frames to save time
        r = run_frame_pipeline(i, engine=None, use_engine=False)
        cpu_results.append(r)

    def stats(results):
        times = [r.total_ms for r in results]
        times.sort()
        return {
            "avg_ms": round(sum(times) / len(times), 1),
            "min_ms": round(min(times), 1),
            "max_ms": round(max(times), 1),
            "p1_low_ms": round(times[max(0, len(times) // 100)], 1),
            "p99_ms": round(times[min(len(times) - 1, int(len(times) * 0.99))], 1),
            "avg_fps": round(1000 / max(sum(times) / len(times), 0.1), 0),
            "p1_low_fps": round(1000 / max(times[-max(1, len(times) // 100)], 0.1), 0),
        }

    engine_stats = stats(engine_results)
    cpu_stats = stats(cpu_results)

    temps = [r.gpu_temp for r in engine_results if r.gpu_temp > 0]
    peak_temp = max(temps) if temps else 0

    engine.stop()

    return {
        "frames": n_frames,
        "engine": engine_stats,
        "cpu_baseline": cpu_stats,
        "peak_gpu_temp": peak_temp,
        "device_map": engine_results[-1].devices if engine_results else {},
    }
