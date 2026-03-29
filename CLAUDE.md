# CLAUDE.md — R.A.G-Race-Router [Adaptive Tri-Processor Inference Runtime]

## Project identity

**R.A.G-Race-Router** [Adaptive Tri-Processor Inference Runtime] is the first open-source architecture for self-optimizing tri-processor inference on AMD Ryzen AI 300 series APUs. It dynamically routes ML workloads across CPU (Zen 5), iGPU (RDNA 3.5 / Radeon 890M), and NPU (XDNA 2) on a single SoC with shared memory — learning the optimal split for each specific chip over time.

- **Repo:** `Peterc3-dev/rag-race-router`
- **Author:** Peter Clemente (@Peterc3-dev)
- **Status:** Pre-alpha / architecture + feasibility study. No runtime code yet.
- **License:** MIT

## What's in the repo

```
rag-race-router/
├── CLAUDE.md             ← You are here
├── CLAUDE_CODE_TASK.md   ← One-time publish task (proofread + push to GitHub)
├── README.md             ← Full project README (architecture, research, build plan)
├── blog-post.md          ← dev.to article (publish after repo is live)
└── LICENSE               ← MIT
```

## Current task: Proofread and publish

Read `CLAUDE_CODE_TASK.md` for the immediate task. Summary:

1. **Proofread** README.md and blog-post.md — fix typos, verify URLs, check technical consistency, flag any overclaiming
2. **Rename the project** from "tri-processor-inference" to "R.A.G-Race-Router" across all files:
   - Repo description and README title/header should reference "R.A.G-Race-Router"
   - Keep the repo slug as `rag-race-router` (not `tri-processor-inference`)
   - Blog post title can stay as-is (it's a hook, not a brand announcement) but mention the project name somewhere natural
3. **Create the repo** on GitHub under Peterc3-dev and push
4. **Do NOT** publish the blog post to dev.to — just confirm it's ready

## Project context

### The architecture in one paragraph

R.A.G-Race-Router coordinates all three processors on an AMD Ryzen AI 300 APU for ML inference. The NPU handles efficient inference (50 TOPS at 2W) and runs a tiny scheduling neural network as a persistent background agent. The iGPU handles parallel compute via Vulkan (not ROCm — Vulkan is 60% faster on the 890M due to unified memory access). The CPU orchestrates, preprocesses, and fills gaps. A feedback loop collects runtime metrics (latency, power, thermal, memory pressure) and continuously refines the scheduling policy. Over time, each machine develops a unique "hardware personality" — an evolving model of that specific chip's behavior.

### Key technical facts (do not contradict these)

- **Vulkan > ROCm on 890M** for prompt processing (~60% faster). Reason: `hipMallocManaged` is broken on gfx1150; ROCm sees only BIOS VRAM carveout while Vulkan accesses full VRAM+GTT pool.
- **ONNX Runtime Vitis AI EP is non-functional on Linux.** The `voe.passes` module is missing. This is a confirmed blocker, not speculation.
- **No DMA-BUF bridge exists between amdgpu and amdxdna drivers.** GPU↔NPU data flows through CPU-mediated host buffers.
- **FastFlowLM v0.9.35+** (via Lemonade 10.0) is the working NPU inference path on Linux. Llama 3.2 1B at ~60 tok/s, <2W.
- **XDNA kernel driver mainlined in Linux 6.14.** XRT userspace must be built from source on Arch/CachyOS.
- **The "first of its kind" claim is scoped precisely:** first open-source *architecture* for self-optimizing tri-processor inference on a consumer APU. Not "first runtime" — there is no runtime code yet. This distinction is load-bearing for credibility.
- **Six confirmed novel contributions** (no prior art found in 25+ paper review): NPU-as-scheduling-agent, persistent hardware personality, three-processor dynamic operator placement, cross-model transfer learning for on-device scheduling, Vulkan+XRT memory bridge, NPU-bookended assembly line.
- **AMD's own research (Karami et al., 2025)** characterizes the scheduling problem (O(2^125) search space) but does not build a runtime.

### Hardware target

AMD Ryzen AI 300 series ("Strix Point"):
- CPU: Zen 5 / Zen 5c
- iGPU: RDNA 3.5 (gfx1150), 16 CUs, Radeon 890M
- NPU: XDNA 2, 50 TOPS INT8/BFP16, dynamic spatial partitioning at column boundaries
- Memory: Shared LPDDR5X, true UMA, no dedicated VRAM

Primary dev machine: GPD Pocket 4 (Ryzen AI HX 370, 32GB RAM, CachyOS)

### Connections to broader work

This project is part of **CIN** (a distributed inference network). The CIN context is mentioned once at the end of the blog post — don't over-explain it, don't remove it. The ThinkCentre M70q Gen 5 is the always-on CIN hub; the GPD is the mobile workstation. Both connect via Tailscale mesh.

## Style and voice

- **Technical but accessible.** Not academic paper voice. Not casual/chatty. Think: senior engineer writing a project brief for other senior engineers.
- **No hedging language** like "it should be noted that" or "it is worth mentioning." Say the thing.
- **No marketing superlatives.** The research speaks for itself. "First of its kind" is a factual claim backed by a literature review, not hype.
- **Tables should render on GitHub.** Verify markdown table formatting.
- **Code blocks should be copy-pasteable.** No smart quotes, no invisible characters.
- **Phosphor green (#00ff00) is the aesthetic** if any visual branding is ever needed, but don't add any now.

## What NOT to do

- Do not add code files, CI/CD, GitHub Actions, or issue templates
- Do not create ARCHITECTURE.md yet (referenced in README but intentionally deferred)
- Do not change the MIT license
- Do not weaken the "first of its kind" claim — it's carefully scoped and defensible
- Do not strengthen it beyond "first open-source architecture" — there's no running code
- Do not add contributor guidelines or CODE_OF_CONDUCT yet
- Do not restructure the README's section order — it's intentionally narrative (idea → what works → architecture → novel contributions → research → getting started)
