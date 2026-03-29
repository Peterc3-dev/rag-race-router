# Social Post Drafts — R.A.G-Race-Router

DO NOT POST — drafts for user review only.

---

## r/LocalLLaMA

**Title:** Built a three-processor inference runtime for AMD Ryzen AI 300 — CPU, GPU, and NPU dispatching real work with learned routing

I've been building an open-source inference engine that coordinates all three processors on AMD Ryzen AI 300 APUs (Zen 5 CPU + Radeon 890M iGPU + XDNA 2 NPU).

**What it does:**
- Dispatches ML operations across CPU, GPU (Vulkan), and NPU based on learned per-operation profiles
- Thermal-aware "pulsed" GPU execution — burst compute, check temperature, cooldown if needed
- Personality database that learns which device is best for each operation on your specific chip
- Adaptive rerouting when GPU heats up (shifts ops to NPU/CPU)

**Key findings:**
- Vulkan is ~60% faster than ROCm on the 890M for our workloads (`hipMallocManaged` is broken on gfx1150)
- The NPU kernel driver had an init-order bug — we patched it (SMU init before firmware load)
- Llama 3.2 1B runs on the NPU at 40-46 tok/s prefill via FastFlowLM
- The personality database correctly learns routing (embed→NPU, matmul→GPU, tokenize→CPU)

**Status:** Pre-alpha, three-processor dispatch working, learning personality working. No production inference yet — this is architecture + PoC.

Code: https://github.com/Peterc3-dev/rag-race-router (MIT)

---

## r/AMD

**Title:** First open-source runtime using all three processors on Ryzen AI 300 (CPU + Radeon 890M + XDNA NPU)

Built an inference engine that actually uses all three processors on the Ryzen AI 9 HX 370 (GPD Pocket 4):

- **CPU** (Zen 5): tokenization, lightweight ops, overflow
- **GPU** (Radeon 890M via Vulkan/Kompute): matrix multiply, attention — Vulkan SPIR-V shaders, NOT ROCm
- **NPU** (XDNA 2, 50 TOPS): embeddings, normalization via FastFlowLM

**Why Vulkan, not ROCm?** `hipMallocManaged` is broken on gfx1150 (Strix Point). HIP only sees the BIOS VRAM carveout (~512MB) while Vulkan accesses the full VRAM+GTT pool (8GB+). We measured 1,085 GFLOPS on Vulkan vs much less on HIP for the same matmul.

**NPU was broken — we fixed it.** The in-tree amdxdna driver has an init-order bug on Strix Halo: it calls SMU init before PSP firmware loading. The SMU doesn't respond until firmware is loaded. Our patch skips SMU, loads firmware via PSP, runs without power management. Llama 3.2 1B at 40-46 tok/s on the NPU.

**Thermal pulsing:** The engine monitors GPU temp via amdgpu_top and applies burst/cooldown cycles. Under 30s stress test: GPU stayed at 50C peak, 54 ops rerouted from GPU to NPU/CPU adaptively.

Pre-alpha, MIT license: https://github.com/Peterc3-dev/rag-race-router

Hardware: GPD Pocket 4, Ryzen AI 9 HX 370, 32GB LPDDR5X, CachyOS (Arch-based)
