---
title: "Your AMD APU Has Three Processors. Why Does ML Only Use One?"
published: false
description: "Building a self-optimizing inference runtime that coordinates CPU, iGPU, and NPU on Ryzen AI 300 — and why nobody's done it yet."
tags: amd, machinelearning, linux, opensource
---

I've been staring at my AMD Ryzen AI HX 370 for months thinking the same thing: this chip has three processors that share a memory bus, and every ML runtime ignores two of them.

The CPU runs inference. The GPU sits there unless you explicitly set it up. The NPU — 50 TOPS of dedicated neural compute at 2 watts — does literally nothing unless you're on Windows blurring your webcam background.

What if a runtime used all three? And what if it *learned* the optimal split for your specific chip?

## The hardware nobody's exploiting

The Ryzen AI 300 series is a monolithic die. CPU (Zen 5), iGPU (RDNA 3.5, 16 CUs), and NPU (XDNA 2) share physical DDR5X through one memory controller. No dedicated VRAM. True unified memory architecture.

The theoretical pipeline:

- **NPU** handles efficient inference at 50 TOPS / 2W — your always-on workhorse
- **iGPU** handles flexible parallel compute — batch processing, larger models
- **CPU** orchestrates, preprocesses, and fills gaps
- A **scheduler** (running on the NPU itself) learns which ops run best where on *your* chip

After 47 generations, it tells you: *"guidance_scale=4.2 produces the cleanest output on this hardware."* After a driver update: *"ROCm 7.3 improved GPU throughput 15% — redistributing layers."*

## I did the research. Here's what's actually possible today.

I spent a week mapping every dependency, every driver, every research paper. The short version:

**What works (March 2026):**
- NPU inference via FastFlowLM: Llama 3.2 1B at ~60 tok/s, under 2 watts
- XDNA kernel driver mainlined in Linux 6.14
- iGPU inference via Vulkan llama.cpp (and it's 60% *faster* than ROCm — more on that below)
- All three processors sharing physical memory

**What doesn't:**
- ONNX Runtime's Vitis AI EP is completely broken on Linux
- `hipMallocManaged` returns "not supported" on the 890M
- No DMA-BUF bridge between the GPU and NPU drivers
- Nobody has run all three processors simultaneously for inference

## The Vulkan surprise

Here's something the ROCm community hasn't fully absorbed: **Vulkan outperforms ROCm by ~60% for prompt processing on the Radeon 890M.**

The reason is memory access. ROCm's `hipMalloc` can only address the BIOS-configured VRAM carveout. On a 96GB system, that might be 48GB. Vulkan sees the entire pool — VRAM plus GTT, 80+ GB — via `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.

On a memory-bandwidth-bound workload over a 120 GB/s LPDDR5X bus, that gap is decisive. For this project, Vulkan is the iGPU backend.

## The NPU-as-scheduler concept

This is the part that has no prior art in the literature. I checked.

The idea: dedicate a few NPU columns to running a tiny scheduling neural network (~500 bytes, based on CoDL's latency predictor architecture) that monitors CPU and GPU utilization, thermal state, and inference latency — then dynamically redistributes operators across all three processors.

The NPU is perfect for this because:
- It runs at <2W, always-on without thermal impact
- XDNA 2 supports dynamic spatial partitioning at column boundaries
- The remaining NPU columns still handle inference workloads
- It's literally a neural processor running a neural scheduling policy

The closest analogy is SmartNIC-as-orchestrator in distributed systems (Wave, Conspirator, RingLeader) — an auxiliary processor dedicated to scheduling decisions. Nobody's applied this pattern to NPUs.

## First of its kind

I reviewed 25+ papers, AMD's own 2025 scheduling research, and every major heterogeneous inference project I could find. Nobody has built this.

CPU+GPU co-execution is well-studied (CoDL, SparOA). NPU+GPU scheduling exists in limited forms. But a self-optimizing runtime that dynamically partitions operators across all three processors on a consumer APU? No prior implementation. AMD's Karami et al. (2025) characterize the *problem* — a combinatorial scheduling search space of O(2^125) — but don't build a runtime that solves it.

To be clear: I've architected this and validated feasibility, not shipped a running system. But the architecture itself, the feasibility study, and the research synthesis represent the first open-source attempt at this category of runtime.

## Six novel contributions

From the full literature review, these have no existing implementation:

1. **NPU-as-scheduling-agent** for CPU+GPU workload orchestration
2. **Persistent hardware personality** — an evolving model of your chip's specific behavior over weeks/months
3. **Three-processor dynamic operator placement** on a single SoC (CPU+GPU is studied; all three is not)
4. **Cross-model transfer learning** for on-device scheduling (learning from Model A improves scheduling of Model B)
5. **Vulkan+XRT memory bridge** — combining Vulkan's superior unified memory access with XRT buffer objects via CPU-mediated sharing
6. **NPU-bookended assembly line** — NPU dispatches at the start, assembles at the end; CPU and GPU are decoupled async producers. 1000:1 speed ratio makes scheduling overhead effectively zero

## What's next

I'm calling the project **R.A.G-Race-Router** [Adaptive Tri-Processor Inference Runtime]. The runtime treats the three processors as an assembly line: CPU and GPU are asynchronous production belts, and the NPU bookends the pipeline — dispatching work at the start, assembling output at the end. At 50 TOPS, the NPU evaluates scheduling decisions in microseconds while CPU/GPU compute takes milliseconds. It appears to be in two places at once. After a few runs, it encodes the dispatch pattern as lightweight rules that auto-execute with near-zero overhead, only re-engaging when something changes.

The full feasibility study, architecture, literature review, and Phase 1 build instructions are here:

**→ [github.com/Peterc3-dev/rag-race-router](https://github.com/Peterc3-dev/rag-race-router)**

Phase 1 is proving three-processor data flow on a Ryzen AI 300 under CachyOS. The immediate step is getting FastFlowLM running on the NPU and benchmarking the three-way pipeline.

This is pre-alpha. No code yet — just architecture, validated feasibility, and a clear build path. If you're running Ryzen AI 300 on Linux and this resonates, I'd love to hear from you.

## The bigger picture

AMD shipped hardware that could redefine edge inference. The silicon is there. The drivers are (mostly) there. What's missing is a runtime that treats the whole SoC as a unified inference machine instead of three separate devices that happen to share a bus.

Every chip is slightly different. Thermal characteristics, silicon lottery, memory controller behavior, driver versions. A runtime that learns *your* chip's personality isn't an optimization — it's a new category.

The models are coming. The question is whether any runtime will know how to actually use the hardware it's running on.

---

*I'm Peter Clemente ([@Peterc3-dev](https://github.com/Peterc3-dev)). I build systems on Linux. This project is part of a broader architecture called CIN — a distributed inference network that treats every device as a node.*
