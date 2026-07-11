# CacheFlow: Bounded-Memory Mixture-of-Experts Inference on Edge Devices

**Turja Dutta, Md. Iftaker Ahamed Sayem, Arif Mohammed Asfe, Jamil As-Ad**

Submitted to *Future Generation Computer Systems* (Elsevier)

---

## What Problem Does CacheFlow Solve?

Large Mixture-of-Experts (MoE) models — where only a sparse subset of expert parameters activates per token — are computationally efficient during the forward pass but still demand **monolithic memory allocation** in conventional serving frameworks. On edge hardware like an NVIDIA T4 (15 GB VRAM), loading all expert weights triggers out-of-memory failures even though only a fraction are actually used per token.

CacheFlow is a **runtime framework** that dynamically manages which experts reside in GPU memory at any moment, enabling models up to **20 billion parameters** to run on a single memory-constrained GPU.

## Core Idea

Rather than loading the entire model into VRAM, CacheFlow streams only the **token-activated expert subset** from disk through pinned host memory into a bounded GPU cache. The total parameter count is decoupled from active VRAM consumption — the memory footprint scales with the number of active experts, not the monolithic model size.

## Key Mechanisms

| Component | What It Does |
|-----------|-------------|
| **MmapExpertStore** | Uses OS memory-mapped I/O (`mmap`) to access specific tensor byte ranges directly from disk — no full checkpoint deserialization, no redundant host-page-cache copies |
| **ExpertCache (LRU)** | Maintains a fixed-capacity GPU buffer of recently used experts. LRU eviction exploits the temporal locality of token routing within semantic domains, purging stale parameters when topics shift |
| **Pinned Memory Staging** | Page-locked host memory buffers act as staging slots for CPU→GPU transfers, enabling efficient PCIe DMA without user-space copies |
| **Asynchronous Prefetcher** | Overlaps host-to-device expert transfers with active GPU computation on a dedicated CUDA stream, hiding PCIe latency behind ongoing matrix operations |

## Key Results

- **20B-parameter model** (GPT-OSS-Safeguard) executed successfully within **6.2 GB peak VRAM** on an NVIDIA T4
- 20× increase in model size → only **3.4× increase** in VRAM (sub-linear scaling)
- **2.26× throughput improvement** from asynchronous prefetching vs. synchronous I/O
- Warm-cache hit rates consistently **>85%** within stable semantic domains
- LRU structurally outperforms LFU — frequency-based eviction suffers from ghost context pollution where stale high-frequency experts occupy cache slots long after the topic changes
- Mean generation latency: **87 ± 6 ms/token**, tight statistical concentration despite disk-backed storage
- Ablation shows memory-mapped I/O is the single most impactful component — disabling it causes a **3.75× slowdown**

## Limitations

- Long-context generation (>4096 tokens) triggers **KV-cache contention**: the dynamically growing key-value cache eventually cannibalizes ExpertCache capacity, dropping expert hit rates from 88% to 72% at 8192 tokens
- Abrupt semantic domain shifts incur transient **8–12% latency spikes** as the cache flushes and repopulates
- Evaluated on a single hardware profile (T4, PCIe Gen3 x16); broader hardware characterization remains future work
- Does not address generation quality — strictly a systems-level memory management study

## Design Principle: Correctness Preservation

CacheFlow is purely a memory management layer — it modifies **only where and when** expert weights reside in memory. It introduces **zero changes** to model parameters, activation functions, routing decisions, or the computational graph. Given deterministic I/O, outputs are mathematically identical to fully-resident execution.

## Why This Matters

Conventional wisdom suggests deploying larger models requires proportionally larger hardware. CacheFlow demonstrates that **intelligent runtime scheduling and dynamic offloading can substitute for scaled accelerators** — the bottleneck shifts from VRAM capacity to predictable PCIe bandwidth, and the framework converts hard out-of-memory failures into tunable memory-latency tradeoffs.
