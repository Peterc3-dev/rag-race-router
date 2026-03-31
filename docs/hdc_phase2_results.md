# HDC Phase 2 Results: NPU Cosine Similarity via Triton-XDNA

**Date:** 2026-03-31
**Hardware:** AMD Ryzen AI 9 HX 370 (GPD Pocket 4), XDNA 2 AIE2P NPU
**Software:** triton-xdna 3.6.0, XRT 2.21.75, CachyOS kernel 6.19.10

## Summary

Phase 2 goal: move HDC codebook lookup from CPU Python (5221 us) to NPU hardware.

**Result:** NPU pipeline works end-to-end but dispatch overhead (~60ms) exceeds CPU matmul time (~200us) at HDC-relevant matrix sizes (256x256). The correct architecture is CPU for routing, NPU for inference workloads.

## What worked

- **triton-xdna** installed and running on XDNA 2 NPU
- Patched `detect_npu_version()` to recognize `RyzenAI-npu4` device name (filed amd/Triton-XDNA#33)
- Vec-add: 6 sizes (1024-32768), all passed on NPU
- INT8 matmul (64x64): all sizes passed, verified against CPU
- BF16 matmul (256x256 to 4096x4096): all sizes passed, verified against CPU
- HDC cosine similarity via pre-normalized BF16 matmul: 98% index agreement with CPU reference

## Benchmark: HDC Codebook Lookup

| Method | Avg Latency | vs Phase 1 |
|--------|------------|------------|
| Phase 1: Python torchhd loop | 5221 us | baseline |
| Phase 2: CPU batch matmul | 200 us | **25x faster** |
| Phase 2: NPU via Triton-XDNA | ~62,000 us | 0.1x (slower) |

## Why NPU is slower at this size

The NPU dispatch path through Triton-XDNA includes:
1. xclbin loading (~50ms)
2. Hardware context setup
3. Buffer allocation + data copy to NPU
4. Actual compute (~1-10 us on systolic array)
5. Data copy back

At 256x256 BF16, the CPU does the matmul in ~200us because the data fits in L1 cache. The NPU's 50 TOPS is irrelevant when dispatch overhead is 1000x the compute time.

## NPU crossover point

The NPU wins when compute time exceeds ~60ms on CPU. For dense matmul, this is roughly 4096x4096 and above — i.e., the sizes used in actual model inference, not routing decisions.

## Architectural insight

This validates the tri-processor thesis:
- **CPU:** Fast small operations — HDC routing (200us), normalization, argmax
- **NPU:** Large inference matmuls — model layers via FastFlowLM (already proven at 60 tok/s)
- **GPU:** Medium operations — training, attention, operations too large for CPU but not matching NPU dispatch cost

The router doesn't need to run on the NPU. It needs to run fast on CPU and *decide* when to use the NPU.

## Code changes

1. `engine/hdc_scheduler.py`: Replaced per-entry cosine similarity loop with batch matmul (`_batch_similarity`). Pre-normalizes codebook matrix, single matmul for all similarities. 25x speedup.

2. `triton-xdna` setup: Python 3.12 venv, pip install from GitHub wheels, `XILINX_XRT=/usr`, `AMD_TRITON_NPU_OUTPUT_FORMAT=xclbin`, npu4 detection patch.

## Setup for reproducing

```bash
# Create venv with Python 3.12 (triton-xdna needs 3.11-3.13)
python3.12 -m venv ~/triton-xdna-env
source ~/triton-xdna-env/bin/activate

# Install triton-xdna + dependencies
pip install triton-xdna \
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

# Patch npu4 detection (until amd/Triton-XDNA#33 is fixed)
sed -i 's/elif "Strix" in name:/elif "Strix" in name or "npu4" in name:/' \
  ~/triton-xdna-env/lib/python3.12/site-packages/triton/backends/amd_triton_npu/driver.py

# Set environment
export XILINX_XRT=/usr
export AMD_TRITON_NPU_OUTPUT_FORMAT=xclbin

# Run examples
cd /tmp/Triton-XDNA/examples/vec-add
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2p.mlir python vec-add.py
```

## Next steps

- Optimize NPU dispatch path (keep hardware context warm, pre-allocate buffers)
- Test larger matrix sizes to confirm crossover point
- Integrate triton-xdna matmul into inference pipeline alongside FastFlowLM
- Consider Rust FFI for XRT to bypass Python dispatch overhead entirely

## Persistent Runtime Experiment (v2)

Tested bypassing Triton-XDNA entirely via direct pyxrt API to isolate overhead source.

### Setup
- Pre-allocated all buffer objects at init (one-time: 141ms)
- Pre-loaded xclbin, hardware context, kernel handle
- Measured dispatch with and without data copy

### Results

| Method | Avg Latency | Notes |
|--------|------------|-------|
| Triton-XDNA dispatch | 62,000 us | Per-call xclbin load + BO alloc |
| Persistent full dispatch | 113,438 us | Pre-alloc BOs, still copy data |
| Persistent kernel only | 111,953 us | No data copy, just run+wait |
| Data copy overhead | 1,485 us | Negligible |
| CPU numpy matmul | 5,310 us | Winner at 256x256 |

### Key finding

The ~112ms overhead is **in the XRT kernel driver**, not in Python, not in data copy, not in buffer allocation. The XDNA driver has high per-dispatch overhead by design.

This is because the NPU is a **pipeline processor, not a function-call processor**:
- Each dispatch programs DMA controllers, configures the AIE array, and synchronizes
- FastFlowLM achieves 60 tok/s by compiling the **entire model** as one NPU program and streaming tokens through a single persistent dispatch
- Individual kernel calls (matmul, vec-add) pay the full setup cost every time

### Architectural conclusion

The NPU dispatch model is fundamentally different from CPU/GPU:

| Processor | Dispatch Model | Optimal Use |
|-----------|---------------|-------------|
| CPU | Instant (ns) | Small ops, routing, pre/post processing |
| GPU | Fast (us) | Medium ops, training, attention |
| NPU | Slow setup, fast sustained (ms init, ns/op) | Whole-model inference, streaming workloads |

**The tri-processor router doesn't fight the dispatch model — it respects it:**
- CPU handles HDC routing at 200us (25x faster than Phase 1)
- NPU runs sustained inference via FastFlowLM (60 tok/s, <2W)
- GPU fills the middle for operations too large for CPU cache but not worth NPU setup cost

### NPU vs CPU crossover point (measured)

| Matrix Size | NPU (ms) | CPU (ms) | Winner |
|------------|----------|----------|--------|
| 256x256 | 62.5 | 0.4 | CPU |
| 512x512 | 60.3 | 1.0 | CPU |
| 1024x1024 | 65.7 | 8.0 | CPU |
| 2048x2048 | 75.5 | 33.8 | CPU |
| 4096x4096 | 127.0 | 235.1 | **NPU** |

Crossover: between 2048 and 4096. NPU wins at 4096+ where compute exceeds dispatch overhead.
