# R.A.G-Race-Router — Roadmap

## Phase 1: Bootstrap Router (current)

*Status: Implementation started*

Static and learned routing rules. ONNX inference on NPU via FastFlowLM. Basic CPU/iGPU/NPU task dispatch with thermal-aware pulsing.

- [x] Three-processor detection (CPU, GPU via Vulkan, NPU via amdxdna)
- [x] Pulsed inference engine (7 modules, CLI interface)
- [x] Vulkan GPU compute via Kompute (1,085 GFLOPS, 14.6x vs CPU)
- [x] Hardware personality database (SQLite, learns routing per-op)
- [x] Thermal pulse controller (adaptive burst/cooldown)
- [x] NPU alive (patched amdxdna driver, Llama 3.2 1B at 40-46 tok/s)
- [x] PyTorch built from source for gfx1150 (first public build for RDNA 3.5)
- [ ] ONNX graph dispatcher (parse real models, route each op to optimal device)
- [ ] NPU execution belt in engine (IREE runtime installed, integration pending)
- [ ] MusicGen routed through tri-processor pipeline end-to-end
- [ ] Custom audio codec (APU-Codec) — encoder→NPU, decoder→GPU, quantizer→CPU

## Phase 2: Adaptive Feedback Loop

*Status: Research*

Runtime telemetry drives routing decisions. The "RAG" in the name — retrieval-augmented routing where the router queries its own performance history to make dispatch decisions.

- [ ] Per-run telemetry: latency, throughput, power draw, thermal curves per device per op
- [ ] Router queries personality DB before each dispatch (retrieval-augmented)
- [ ] Learned scheduler model (~500 params) trained via policy gradient from run data
- [ ] Deploy scheduler on NPU as persistent background agent
- [ ] Anomaly detection: identify performance regressions from driver/kernel updates
- [ ] Multi-model support: switch between models, transfer learned routing knowledge
- [ ] NPU per-process memory tracking (when Linux 7.1 AMDXDNA patch lands in CachyOS)

## Phase 3: Hyperdimensional Computing (HDC) Routing Layer

*Status: Research spike*

Encode processor states, model characteristics, and workload signatures as hypervectors. The NPU performs similarity search to snap to the optimal routing configuration via field resolution instead of if/else logic.

### The insight

Current routing is arithmetic: `if gpu_temp > 75: reroute_to_npu()`. This works but scales poorly as the state space grows (3 devices × N ops × thermal states × memory pressure × model characteristics).

Hyperdimensional Computing encodes states as 10,000-dimensional vectors and finds optimal configurations through geometric similarity — cosine distance across a learned codebook. The answer emerges from resonance, not branching logic.

### Why this fits the hardware

The XDNA 2 NPU's systolic array is a grid of identical cells passing data to neighbors in rhythmic pulses — two perpendicular data flows crossing, multiply-accumulate at each intersection. HDC's core operations map directly:

| HDC Operation | What it does | NPU mapping |
|---|---|---|
| **Bind** (XOR) | Associate two concepts | Bitwise op, trivial on any processor |
| **Bundle** (majority vote) | Combine multiple vectors | Element-wise accumulate → threshold |
| **Similarity** (cosine distance) | Find nearest match | Matrix multiply — exactly what the systolic array does |

The key connection: HDC's associative memory search ("which stored pattern is most similar to this input?") IS a matrix multiply — cosine similarity across a codebook. The NPU already does this at 50 TOPS. Nobody has wired it up for HDC inference on XDNA 2.

### Implementation path

1. Install [torchhd](https://github.com/hyperdimensional-computing/torchhd) (`pip install torch-hd`)
2. Encode routing states as hypervectors: `state_hv = bind(gpu_temp_hv, workload_hv, model_hv)`
3. Build a codebook of known-good routing configurations as hypervectors
4. At dispatch time: compute similarity between current state and codebook → snap to nearest optimal config
5. Profile which ops map efficiently to the XDNA systolic array
6. Compare latency/accuracy vs Phase 2's scalar policy gradient approach

### Key properties

- **10x more error-tolerant** than neural nets (noise in individual dimensions cancels out)
- **No training loop** — codebook entries are constructed, not learned via gradient descent
- **Incremental learning** — new patterns added by bundling, no catastrophic forgetting
- **Massively parallel** — all similarity comparisons happen simultaneously
- **Natural fit for edge hardware** — low precision (even binary) works because information is distributed across thousands of dimensions

### References

- [torchhd](https://github.com/hyperdimensional-computing/torchhd) — PyTorch library for Hyperdimensional Computing
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
- Imani, M. et al. (2019). "A Framework for Collaborative Learning in Secure High-Dimensional Space" (HDC on edge devices)
- Energy-based models / Hopfield networks are the theoretical ancestor

## Phase 4: HDC as Inference Primitive

*Status: Speculative research*

Run actual model inference through hyperdimensional operations on the NPU. Not just routing decisions — the inference itself uses geometric resolution instead of brute-force matrix multiply.

This is the "Magnetix layer" — data encoded as high-dimensional field states, answers self-assemble from resonance dynamics rather than sequential arithmetic.

### What this would mean

Instead of:
```
input → 10 billion multiply-accumulate ops → output
```

This:
```
input → encode as hypervector → similarity search across learned field → output snaps into place
```

The computation is the same hardware operation (matrix multiply for similarity), but the representation is fundamentally different. Information is distributed holographically — every dimension contains partial information about every concept. Answers emerge from the geometry of the space, not from chaining arithmetic.

### Prerequisites

- Phase 3 working (HDC routing proven on XDNA 2)
- Hypervector codebook for the target domain (language, audio, vision)
- Benchmarks showing HDC inference quality vs transformer baseline
- Understanding of which model layers can be replaced with HDC ops vs which can't

### Why this might work

The XDNA 2 systolic array does thousands of multiply-accumulate operations per cycle in a grid pattern. HDC similarity search is embarrassingly parallel across codebook entries. The hardware topology (grid of cells, neighbor-passing, rhythmic pulses) maps naturally to HDC's distributed representation.

The question isn't whether the hardware CAN do it — it already does matrix multiply at 50 TOPS. The question is whether HDC representations produce useful inference results for the target domain.

### Why this might not work (yet)

- HDC has shown strong results for classification but less proven for generation tasks
- Audio/music generation may require the sequential precision that transformers provide
- The quality bar for "sounds good" is higher than for "classifies correctly"
- Phase 3 results will inform whether Phase 4 is viable

---

## Timeline

| Phase | Status | Estimated |
|---|---|---|
| Phase 1 | Implementation started | Current |
| Phase 2 | Research | Weeks |
| Phase 3 | Research spike | Months |
| Phase 4 | Speculative | Unknown |

Phases 1-2 are engineering. Phases 3-4 are research. The repo ships whatever works; roadmap items are signals of direction, not commitments.
