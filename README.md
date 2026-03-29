# R.A.G-Race-Router [Adaptive Tri-Processor Inference Runtime]

**A self-optimizing inference runtime that coordinates CPU, iGPU, and NPU on AMD Ryzen AI 300 series APUs.**

Every machine develops its own optimized inference personality. The more you use it, the better it gets at using *your* specific hardware.

> **To our knowledge, this is the first open-source architecture for self-optimizing tri-processor inference on a consumer APU.** A full literature review (25+ papers, AMD's own 2025 scheduling research, every major heterogeneous inference project) found no prior implementation of dynamic three-way CPU+iGPU+NPU operator placement on a single SoC, no prior use of an NPU as a scheduling agent, and no system that builds a persistent model of individual hardware behavior over time. AMD's Karami et al. (2025) characterize the *problem* — O(2^125) scheduling search space — but do not build a runtime that solves it. This project is the first attempt.

---

## The idea

Modern AMD APUs have three distinct processors sharing a single memory bus — and nobody is using all three together for ML inference.

- **CPU** (Zen 5): Flexible, handles everything, mediates memory
- **iGPU** (RDNA 3.5, Radeon 890M): 16 CUs of parallel compute, excellent at batch matrix ops
- **NPU** (XDNA 2): 50 TOPS at 2W, purpose-built for neural network inference

Today, ML runtimes pick *one* and ignore the others. This project builds a runtime that uses all three simultaneously, learns the optimal workload split for your specific chip, and adapts as drivers update, hardware degrades, or models change.

### What it would look like in practice

```
→ "Your GPU runs 15% faster after the ROCm 7.3 update — redistributing layers"
→ "Sequences over 20s crash on GPU — auto-switching to NPU+CPU for longer tracks"
→ "Based on 47 generations, guidance_scale=4.2 produces the cleanest output on this chip"
→ "New MIOpen kernels available — retuning convolution strategy"
```

### Why this doesn't exist yet

- AMD shipped XDNA 2 recently — Linux support just became functional in March 2026
- Nobody thinks of NPU as a *scheduler* — they think of it as a webcam blur accelerator
- Three-processor coordination requires unified memory — only exists on APUs
- The ML community is NVIDIA-brained — they don't consider this topology
- AMD's own software stacks for CPU (ROCm), GPU (ROCm/Vulkan), and NPU (XDNA/XRT) are completely siloed

---

## Current state of the platform (March 2026)

### What works

| Component | Stack | Status | Notes |
|-----------|-------|--------|-------|
| **NPU inference** | FastFlowLM v0.9.35+ via Lemonade 10.0 | ✅ Working | Llama 3.2 1B @ ~60 tok/s, <2W power |
| **NPU kernel driver** | `amdxdna.ko` mainlined in Linux 6.14 | ✅ Stable | Arch PKGBUILDs available for XRT userspace |
| **iGPU via Vulkan** | llama.cpp Vulkan backend | ✅ Working | Full memory access (VRAM + GTT), **60% faster than ROCm** |
| **iGPU via ROCm** | ROCm 7.2.1 (native gfx1150) | ⚠️ Partial | `hipMallocManaged` broken; limited to BIOS VRAM carveout |
| **CPU inference** | llama.cpp CPU, ONNX Runtime CPU EP | ✅ Working | Baseline fallback |
| **NPU bare-metal** | IRON / MLIR-AIE | 🧪 Research | Full programmability, custom kernels possible |

### What doesn't work

| Component | Status | Blocker |
|-----------|--------|---------|
| **ONNX Runtime Vitis AI EP (Linux)** | ❌ Broken | `voe.passes` module missing; 100% CPU fallback |
| **`hipMallocManaged` on gfx1150** | ❌ Not supported | VMM disabled; ROCm sees only BIOS VRAM |
| **DMA-BUF between iGPU and NPU** | ❌ Not implemented | `amdxdna` driver lacks PRIME import/export |
| **ONNX RT multi-EP (ROCm + VitisAI + CPU)** | ❌ Blocked | Vitis AI EP is non-functional on Linux |

### The critical finding: Vulkan > ROCm on the 890M

ROCm's `hipMalloc` can only address the BIOS-configured VRAM carveout (e.g., 48GB on a 96GB system). Vulkan correctly accesses both VRAM and GTT memory (80+ GB total) via `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`. On a memory-bandwidth-bound workload like LLM inference over a 120 GB/s LPDDR5X bus, this gap is decisive.

**For this project, Vulkan is the iGPU backend.** ROCm remains available for the PyTorch/HIP ecosystem where needed.

---

## Architecture

### Memory model

All three processors share physical DDR5X through a unified memory controller. The hardware is UMA — there is no dedicated VRAM. The software hasn't caught up:

```
┌─────────────────────────────────────────────────────┐
│                  Physical DDR5X                      │
│                (shared memory bus)                    │
├──────────────┬──────────────────┬───────────────────┤
│   CPU        │   iGPU           │   NPU             │
│   Direct     │   Vulkan:        │   IOMMU SVA:      │
│   access     │   HOST_VISIBLE   │   process VA space │
│              │   (full pool)    │   + DMA to L2 tiles│
│              │                  │                    │
│              │   ROCm:          │   XRT buffer       │
│              │   hipHostMalloc  │   objects           │
│              │   (mapped+coh.)  │                    │
└──────────────┴──────────────────┴───────────────────┘

Data flow (current feasible path):
  CPU ←→ iGPU: zero-copy via Vulkan HOST_VISIBLE or hipHostMalloc(Mapped)
  CPU ←→ NPU:  near-zero-copy via IOMMU SVA (NPU DMAs from host DDR to L2)
  iGPU ←→ NPU: CPU-mediated (both read/write host DDR, sync via fences)
```

The GPU↔NPU bridge requires CPU-mediated host buffers today. A DMA-BUF PRIME implementation in the amdxdna driver would enable direct transfer — architecturally straightforward but not yet done.

### The assembly line model

The core runtime architecture treats the three processors as an assembly line with the NPU bookending the pipeline:

```
                         NPU
                    ┌────────────┐
                    │  CONTROL   │    Dispatches work to CPU & GPU belts.
                    │  PANEL     │    Learned rules, interrupt-driven.
                    │            │    Only intervenes when pattern breaks.
                    └──────┬─────┘
                      ┌────┴────┐
                      ▼         ▼
               CPU BELT       GPU BELT
               ════════       ════════
              │ task A │     │ task X │    Asynchronous, decoupled.
              │ task B │     │ task Y │    No coordination between
              │ task C │     │ task Z │    CPU and GPU needed.
              └───┬────┘     └────┬───┘
                  └───────┬───────┘
                          ▼
                    ┌────────────┐
                    │  ASSEMBLY  │    Collects fragments from both belts.
                    │  STATION   │    Combines with NPU's own tools
                    │  + TOOLS   │    (INT8 compute, normalization).
                    └──────┬─────┘
                           ▼
                       OUTPUT
```

**The NPU has two roles at two positions — but it's one processor.** At 50 TOPS, the NPU evaluates a scheduling decision in microseconds while CPU/GPU compute takes milliseconds. The ratio is 1000:1. It dispatches, zips to assembly, assembles, zips back — the belts haven't moved. One worker appearing as two.

**Why this works:**

1. **CPU and GPU are asynchronous producers.** They push fragments onto their belts independently. No synchronization between them. No fences. No waiting on each other.

2. **The NPU sets the pace.** It pulls from whichever belt has ready fragments. GPU stalls from thermal throttle → NPU keeps assembling from CPU fragments. Natural backpressure, no explicit scheduling logic needed.

3. **Interrupt-driven, not polling.** After the first ~10 runs, the NPU encodes the dispatch pattern as lightweight rules. These auto-execute with near-zero overhead. The NPU only recomputes the full scheduling policy when something changes — a thermal spike, a driver update, a new model shape.

4. **Closed feedback loop.** The same brain that dispatched the work inspects the results. If GPU fragments arrive late → control panel adjusts next dispatch. If a pattern shifts → NPU re-evaluates and encodes new rules. Dispatcher and quality inspector are the same entity with perfect memory.

5. **Scheduling overhead is effectively zero.** The NPU's scheduling cost (microseconds) is negligible relative to the work being scheduled (milliseconds). It does scheduling as a side effect of existing.

**Why no existing runtime works this way:**

| Traditional scheduler | R.A.G-Race-Router assembly line |
|---|---|
| Static dispatcher assigns devices | NPU dispatches AND assembles |
| CPU/GPU must synchronize | CPU/GPU are fully decoupled |
| Scheduler is overhead | Scheduler is also compute (assembly) |
| Fails if scheduler is wrong | Degrades gracefully (fewer fragments, slower, never crashes) |
| No learning | Learns → encodes rules → goes idle → intervenes on exception |

### The three-phase build plan

**Phase 1 (weeks): Prove three-processor inference**
- Get XDNA NPU running on Arch/CachyOS via xdna-driver + FastFlowLM
- Validate iGPU inference via Vulkan llama.cpp
- Build a minimal pipeline: CPU → NPU → CPU → GPU → CPU with shared host buffers
- Benchmark each processor independently and in combination

**Phase 2 (weeks): Three-way workload splitter**
- Profile which ONNX operators are most efficient on each processor
- Build operator-level graph partitioner (extend ONNX RT multi-EP or build custom)
- Implement CPU-mediated memory bridge with fence-based synchronization
- Measure end-to-end latency vs. single-processor baselines

**Phase 3 (months): Self-optimizing agent**
- Design metrics collection layer (latency, power, thermal, memory pressure)
- Train a lightweight scheduling model (~500 bytes, per CoDL research)
- Deploy scheduler on NPU as persistent background agent (dedicated column partition)
- Build the feedback loop: run → measure → update policy → run

---

## Novel contributions

Based on a full literature review, six aspects of this project have no prior art:

1. **NPU-as-scheduling-agent** — Running a placement policy neural network on dedicated NPU columns to orchestrate CPU+GPU workloads. The SmartNIC-as-orchestrator pattern (Wave, Conspirator, RingLeader) validates the concept; no one has applied it to NPUs.

2. **Persistent hardware personality** — An evolving model of *your specific chip's* behavior (thermal curves, frequency-dependent throughput, degradation over time). Existing auto-tuning creates static lookup tables. This builds a continuously learning hardware model.

3. **Three-processor dynamic operator placement on a single SoC** — CPU+GPU co-execution is well-studied (CoDL, SparOA). NPU+GPU scheduling exists. No system does RL-based dynamic partitioning across all three.

4. **Cross-model transfer learning for on-device scheduling** — Learning from scheduling Model A to improve scheduling of Model B on the same hardware. Placeto and REGAL demonstrate transfer in cloud settings; on-device heterogeneous transfer is unexplored.

5. **Vulkan+XRT memory bridge** — Exploiting Vulkan's superior unified memory access on gfx1150 as the GPU-side API with CPU-mediated sharing to XRT for NPU access. No existing project combines these two stacks.

6. **NPU-bookended assembly line** — The NPU dispatches work at the pipeline start AND assembles output at the end, with CPU and GPU as asynchronous decoupled producers. The 1000:1 speed ratio (NPU microseconds vs CPU/GPU milliseconds) makes the NPU appear to occupy both positions simultaneously. After initial learning, dispatch rules are encoded as lightweight interrupt-driven policies with near-zero steady-state overhead. No existing runtime uses a single processor for both dispatch and assembly in a closed feedback loop.

---

## Research references

This project builds on work from multiple domains. Key papers and projects:

### Heterogeneous scheduling
- **Karami et al. (2025)** — *Exploring the Dynamic Scheduling Space of Real-Time Generative AI Applications on Emerging Heterogeneous Systems* ([arXiv:2507.14715](https://arxiv.org/abs/2507.14715)) — AMD-affiliated; characterizes CPU+GPU+NPU scheduling on Ryzen AI. Found O(2^125) search space, 3× NPU advantage for LLM prefill.
- **CoDL (2022)** — *Efficient CPU-GPU Co-execution for Deep Learning Inference on Mobile Devices* ([ACM](https://dl.acm.org/doi/10.1145/3498361.3538932)) — Operator-level co-execution with ~500-byte latency predictor.
- **SparOA (2025)** — *Sparse and Operator-aware Hybrid Scheduling for Edge DNN Inference* ([arXiv:2511.19457](https://arxiv.org/abs/2511.19457)) — SAC reinforcement learning for dynamic CPU-GPU allocation.
- **ADMS (2025)** — *Optimizing Multi-DNN Inference on Mobile Devices through Heterogeneous Processor Co-Execution* ([arXiv:2503.21109](https://arxiv.org/abs/2503.21109)) — Processor-state-aware scheduling (load, temperature, frequency).

### Learned device placement
- **Mirhoseini et al. (2017)** — *Device Placement Optimization with Reinforcement Learning* ([arXiv:1706.04972](https://arxiv.org/abs/1706.04972)) — Google Brain; foundational RL-based operator placement.
- **Placeto (2019)** — *Learning Generalizable Device Placement Algorithms* ([NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/71560ce98c8250ce57a6a970c9991a5f-Abstract.html)) — GNN embeddings for transfer across computation graphs.
- **REGAL (2019)** — *Transfer Learning For Fast Optimization of Computation Graphs* — Deep RL + genetic algorithms with transfer learning.

### Auto-tuning and self-optimization
- **Apache TVM Meta Schedule** — Template-free auto-tuning via probabilistic scheduling DSL ([RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md)).
- **Halide learned auto-scheduler (Adams et al., 2019)** — Cost model trained on random programs, 2× over human experts ([ACM](https://dl.acm.org/doi/10.1145/3306346.3322967)).
- **APHES (KIT)** — Always-on autotuning during production, continuous adaptation.
- **OLTunes (2025)** — Online learning for continuous runtime optimization, 58% GPU utilization improvement.

### NPU architecture and bare-metal programming
- **Rösti et al. (2025)** — *Unlocking the AMD Neural Processing Unit for ML Training on the Client Using Bare-Metal-Programming Tools* ([arXiv:2504.03083](https://arxiv.org/abs/2504.03083)) — GPT-2 fine-tuning via IRON on XDNA.
- **IRON / MLIR-AIE** — AMD's bare-metal NPU programming toolkit ([GitHub](https://github.com/Xilinx/mlir-aie)).

### Auxiliary-processor-as-orchestrator (SmartNIC analogy)
- **Wave (2024)** — RPC scheduling offloaded to SmartNIC ARM cores as persistent agents.
- **Conspirator (USENIX ATC 2024)** — SmartNICs as control plane for distributed ML.
- **RingLeader (NSDI 2023)** — Intra-server orchestration offloaded to NICs.

### Unified memory exploitation
- **torch-apu-helper** — Custom HIP allocator for PyTorch on AMD APUs ([GitHub](https://github.com/pomoke/torch-apu-helper)).
- **AMD AOMP 19.0-2** — Zero-copy CPU-GPU shared memory via `HSA_XNACK=1 OMPX_APU_MAPS=1`.
- **OpenVINO HETERO plugin** — Automatic operation-level device assignment with fallback chains.

---

## Getting started (Phase 1)

### Prerequisites
- AMD Ryzen AI 300 series APU (HX 370, HX 365, etc.)
- Linux kernel ≥ 6.14 (avoid 6.18–6.18.7 — IOMMU regression)
- Arch Linux, CachyOS, or Ubuntu 24.04+
- BIOS: IPU enabled, fixed VRAM allocation, IOMMU on (do NOT use `amd_iommu=off`)

### Step 1: Verify NPU hardware
```bash
ls /dev/accel/accel0
zcat /proc/config.gz | grep AMDXDNA
cat /proc/cmdline | grep iommu
```

### Step 2: Build XDNA userspace (Arch/CachyOS)
```bash
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver/xrt/build/arch && makepkg -p PKGBUILD-xrt-base
sudo pacman -U xrt-base-*.pkg.tar.zst
cd ../../../build/arch && makepkg -p PKGBUILD-xrt-plugin
sudo pacman -U xrt-plugin-amdxdna-*.pkg.tar.zst
```

### Step 3: Run first NPU inference
```bash
pip install lemonade-server
lemonade --model Qwen/Qwen3-0.6B --recipe flm --device npu
# Expected: ~89 tok/s at <2W
```

### Step 4: Validate iGPU via Vulkan
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
./build/bin/llama-cli -m model.gguf -ngl 99
```

### Step 5: Prove three-processor data flow
```
CPU (preprocess) → NPU (stage 1 via Lemonade) → CPU (bridge) → iGPU (stage 2 via Vulkan) → CPU (output)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed memory bridge design.

---

## Key resources

| Resource | Link |
|----------|------|
| XDNA driver (kernel + XRT) | [github.com/amd/xdna-driver](https://github.com/amd/xdna-driver) |
| FastFlowLM (NPU LLM inference) | [fastflowlm.com](https://fastflowlm.com/) |
| Lemonade SDK (multi-device orchestration) | [lemonade-server.ai](https://lemonade-server.ai/) |
| IRON (bare-metal NPU programming) | [github.com/amd/IRON](https://github.com/amd/IRON) |
| MLIR-AIE (NPU compiler toolchain) | [github.com/Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) |
| ROCm for Ryzen docs | [rocm.docs.amd.com/projects/radeon-ryzen/](https://rocm.docs.amd.com/projects/radeon-ryzen/) |
| torch-apu-helper (PyTorch APU memory fix) | [github.com/pomoke/torch-apu-helper](https://github.com/pomoke/torch-apu-helper) |
| AMD NPU kernel docs | [docs.kernel.org/accel/amdxdna/amdnpu.html](https://docs.kernel.org/accel/amdxdna/amdnpu.html) |
| Arch Linux NPU packaging thread | [bbs.archlinux.org/viewtopic.php?id=304979](https://bbs.archlinux.org/viewtopic.php?id=304979) |

---

## Status

🟡 **Pre-alpha / Research** — Architecture defined, feasibility validated, no code yet.

Phase 1A (NPU proof of life) is the immediate next step. Contributions, discussion, and collaboration welcome — especially from anyone running Ryzen AI 300 on Linux.

---

## Author

**Peter Clemente** ([@Peterc3-dev](https://github.com/Peterc3-dev))

---

## License

MIT
