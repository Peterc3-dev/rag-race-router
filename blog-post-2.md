---
title: "I Got All Three Processors Talking to Each Other on My AMD Laptop"
published: false
description: "CPU, GPU, and NPU dispatching real work through a pulsed inference engine on Ryzen AI 300 — with benchmarks."
tags: amd, machinelearning, linux, opensource
---

Last week I published [an architecture](https://dev.to/peterc3/the-scheduler-that-learns-your-chip-building-tri-processor-inference-for-amd-ryzen-ai-300-5gci). This week it runs.

The R.A.G-Race-Router engine now dispatches real workloads across all three processors on my GPD Pocket 4 (AMD Ryzen AI 9 HX 370): CPU (Zen 5), iGPU (Radeon 890M via Vulkan), and NPU (XDNA 2 at 50 TOPS). Here's what actually happened when I wired it up.

## The Three-Processor Demo

```
Processors: CPU ok  GPU (Vulkan) ok  NPU (XDNA) ok
GPU Temp: 43C | VRAM: 1732/8192 MB | Pulse: READY

  tokenize  -> CPU  (7ms)     [lightweight, no dispatch overhead]
  embed     -> NPU  (282ms)   [embedding lookup, NPU efficient]
  matmul    -> GPU  (5ms)     [512x512 via Vulkan SPIR-V shader]
  attention -> GPU  (5ms)     [scaled dot-product, pulsed burst]
  normalize -> NPU  (5ms)     [RMS norm, NPU sweet spot]
  project   -> GPU  (6ms)     [linear projection, pulsed burst]
  decode    -> CPU  (6ms)     [greedy argmax, trivial]

Pipeline: 316ms total (16ms GPU, 287ms NPU, 13ms CPU)
```

Each operation is dispatched to the device the engine thinks is best, based on heuristics during the first few runs and learned routing rules after that. The personality database records every execution and gradually builds a profile of this specific chip.

## The NPU Was Broken — We Fixed It

The NPU on Strix Halo refused to initialize. The kernel driver's SMU (System Management Unit) failed with `smu cmd 4 failed, 0xff` on every boot. Three sessions of debugging later:

**Root cause**: The driver calls SMU init *before* loading firmware via PSP (Platform Security Processor). On Strix Halo, the SMU doesn't respond until firmware is loaded. Classic init-order bug.

**The fix**: A three-line patch to the out-of-tree amdxdna driver that skips SMU when it fails, loads firmware via PSP anyway, and continues without power management. The NPU runs at default BIOS clocks.

Result: Llama 3.2 1B at 40-46 tok/s prefill, 14-24 tok/s decode, running on the NPU via FastFlowLM. The patched driver loads automatically on boot via a systemd service.

## Vulkan Is Still Faster Than ROCm on This GPU

Updated benchmarks with the engine's Kompute integration:

| Backend | Workload | Performance |
|---------|----------|-------------|
| CPU (NumPy/BLAS) | 512x512 matmul | 7.6ms |
| GPU (Vulkan/Kompute) | 512x512 matmul | 5.0ms |
| GPU (Vulkan/IREE) | 1024x1024 matmul | 1,085 GFLOPS |
| NPU (FLM) | Llama 3.2 1B prefill | 40-46 tok/s |

The Vulkan path uses pre-compiled SPIR-V shaders (matmul, attention, fused add-scale) dispatched through Kompute 0.9.0. ROCm's `hipMallocManaged` remains broken on gfx1150 — Vulkan accesses the full VRAM+GTT pool while HIP only sees the BIOS carveout.

## The Engine Learns Your Chip

After 5 runs, the personality database encodes routing rules:

```
Hardware Personality (35 runs):
Operation        Best on  Avg (ms)   Confidence
tokenize         cpu      0.04       100%
embed            npu      199.04     100%
matmul           gpu      0.50       (learning)
attention        gpu      0.53       (learning)
decode           cpu      0.06       100%
```

The system learns that embedding is best on NPU, tokenization belongs on CPU, and matrix ops go to GPU. When GPU temperature spikes, the dispatcher reroutes small ops to NPU or CPU. Every reroute is logged and fed back into the personality.

## Thermal Stress Test

I pushed the GPU for 30 seconds of continuous compute to test adaptive rerouting:

- 438 operations dispatched
- 54 reroutes (matmul -> CPU when GPU was busy)
- GPU temperature: 45C -> 50C (well within thermal budget)
- Distribution: 380 GPU, 53 NPU, 5 CPU

The pulsed execution model (burst on GPU, check temperature, cooldown if needed) prevents thermal throttling. The engine's pulse controller adapts the burst/cooldown ratio based on real-time temperature readings from amdgpu_top.

## What's Next

This is still pre-alpha. The dispatch overhead matters — for tiny operations, routing through the engine is slower than just running on CPU. The win is thermal management and sustained throughput for long-running workloads.

Next steps:
- Route MusicGen audio generation through the engine (text encoder on CPU, decoder on pulsed GPU, EnCodec on CPU)
- Reduce dispatch overhead for small ops (batch scheduling)
- IREE integration for compiled NPU kernels
- Upstream the amdxdna SMU bypass patch

The code is at [Peterc3-dev/rag-race-router](https://github.com/Peterc3-dev/rag-race-router). MIT license.

---

*This project is part of CIN (Collaborative Intelligence Network), a distributed inference system spanning a ThinkCentre M70q hub and this GPD Pocket 4 mobile workstation, connected via Tailscale mesh.*
