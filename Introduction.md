# 1. INTRODUCTION

## 1.1 Background and Motivation

The emergence of Mixture-of-Experts (MoE) architectures has marked a transformative shift in large-scale deep learning, enabling the training and deployment of increasingly capable language models within fixed computational budgets. Prominent examples include Mistral 7B (sparse expert switching with 32 experts), Qwen2-MoE-A2.7B and Qwen2-MoE-57B (with 64 experts per layer), and Mixtral 8x7B (eight expert pathways per token). These models achieve state-of-the-art performance on numerous benchmarks by routing each token to a small subset of specialized experts, thereby increasing model capacity while maintaining efficient computation.

The theoretical appeal of MoE models is well-established: they provide favorable compute-to-parameter ratios compared to dense alternatives. A 60-expert MoE layer processes tokens through only 2–4 selected experts per token, incurring O(num_selected_experts) compute while maintaining O(num_total_experts) parameters. However, this parameter abundance creates an acute practical challenge: the total weight matrices of all experts must be accessible during inference, consuming enormous quantities of GPU VRAM. A typical MoE layer with 60 experts, each weighing ~100 MB, requires 6 GB of expert parameters alone. When combined with transformer static parameters (embeddings, attention layers, output projections), KV caches for autoregressive decoding, and intermediate activation buffers, the cumulative GPU memory demand can easily exceed 20–30 GB—far beyond the capacity of consumer-grade GPUs (typically 6–12 GB) and even many mid-range server accelerators.

This VRAM bottleneck has emerged as a critical barrier to the widespread deployment of MoE models. While the original MoE training and inference research often assumed access to highly specialized hardware with abundant memory, practical deployment scenarios frequently lack such resources. Academic researchers, small organizations, and edge computing scenarios are forced to either abandon MoE models entirely or resort to crude approximations (expert pruning, quantization) that sacrifice model quality. This motivates the development of efficient memory-aware inference systems that decouple GPU memory requirements from total expert parameter count, enabling deployment of full-capacity MoE models on resource-constrained hardware.

---

## 1.2 Problem Statement

Existing approaches to address the MoE memory bottleneck fall into two broad categories, each with significant limitations.

**Distributed Expert Parallelism:** Systems such as DeepSpeed-MoE and Megatron employ expert parallelism, where experts are partitioned across multiple GPUs with synchronized communication. While effective in datacenter environments with high-bandwidth interconnects, this approach introduces substantial complexity and is inaccessible to practitioners with single-GPU or limited hardware budgets. Furthermore, synchronous communication at expert boundaries creates hard latency floors that preclude deployment in latency-sensitive applications.

**Host Memory Offloading (Naive Staging):** An alternative strategy loads expert weights to host (CPU) pinned RAM prior to inference, then transfers them to GPU on-demand. This approach (exemplified by MoE-Infinity and other disk-offloading frameworks) reduces the GPU VRAM requirement but introduces a new bottleneck: the sequential nature of the prefill phase. When the system must load an expert from pinned host memory to GPU memory before computing on it, the forward pass stalls, blocking token processing until the PCIe transfer completes. Empirically, this results in latencies of **920 ms/token** or higher—a **1.5–3× penalty** compared to baseline systems. Moreover, maintaining all experts in pinned host RAM incurs **5–10 GB** of host memory allocation, which constrains usable GPU capacity and limits the number of concurrent inference requests.

**Joint Limitations:** Both approaches fail to exploit a critical opportunity: **asynchronous overlapping of disk I/O, PCIe transfers, and GPU computation**. In a synchronous design, the critical path is: (disk read) → (wait for completion) → (PCIe transfer) → (wait for completion) → (GPU compute). Each stage serializes, resulting in severe latency penalties. Furthermore, neither system maintains a principled caching strategy to exploit temporal locality in expert access patterns; experts are repeatedly transferred even if accessed consecutively.

The root cause of these limitations is architectural: existing systems do not implement a true three-tier memory hierarchy with non-blocking asynchronous operations. They either consolidate experts in GPU memory (scalability failure), consolidate experts in pinned host memory (latency failure), or implement synchronous staging (latency failure). No existing system seamlessly chains disk I/O → pinned staging → asynchronous GPU transfer with an intelligent caching layer.

---

## 1.3 Proposed Solution: CacheFlow

CacheFlow addresses these limitations through a comprehensive three-tier memory hierarchy and asynchronous execution model. The system is architected as follows:

**Tier 1: Disk Storage (Persistent Archive):** Expert weights are stored on disk in safetensors format, a compact binary encoding that enables efficient random-access slicing. Unlike naive approaches that read entire weight files, CacheFlow reads only the requested expert's slice—typically 100–256 MB per expert, drastically reducing I/O volume.

**Tier 2: Pinned CPU Memory (Staging Buffers):** A small fixed-size pool of pinned (page-locked) CPU buffers stage weights during loading from disk. The key innovation is that the number of staging buffers (e.g., 4–8) is **decoupled** from the total number of experts (e.g., 60). This allows the system to pipeline multiple disk reads and PCIe transfers concurrently, exploiting the high I/O throughput available from modern storage devices and PCIe interconnects.

**Tier 3: GPU VRAM (Working Set Cache):** A fixed-capacity LRU cache in GPU memory holds the subset of experts currently in use. Critically, this cache employs an **intra-batch freezing mechanism**: at the beginning of each forward pass, all required experts (determined by token routing) are locked in the cache, guaranteeing they will not be evicted while the batch is computing. This eliminates correctness hazards and enables safe asynchronous loading of cold experts.

**Asynchronous Execution Model:** The system schedules disk I/O, PCIe transfers, and GPU computation concurrently via `torch.cuda.Stream`. While Expert A's weights are being read from disk by the CPU, Expert B's previously-staged weights are simultaneously being transferred to GPU VRAM via PCIe, and the GPU is computing on Expert C's weights already resident in cache. This triple concurrency dramatically reduces effective latency.

**Transparent to Application:** CacheFlow does not modify the underlying MoE architecture, model structure, or routing mechanisms. It operates at the expert module level as a drop-in replacement for standard parallel expert implementations, making it broadly compatible with existing frameworks and models.

---

## 1.4 Research Objectives

This thesis aimed to achieve the following objectives:

- **Objective 1:** Design and implement a memory-efficient three-tier architecture for MoE inference that decouples GPU VRAM requirements from total expert parameter count while maintaining computational correctness.

- **Objective 2:** Demonstrate that asynchronous prefetching via non-blocking PCIe streams can effectively overlap disk I/O, data staging, and GPU computation, mitigating latency penalties typically incurred by disk offloading.

- **Objective 3:** Empirically validate the system across diverse capacity constraints (GPU memory budgets from 900 MB to 2 GB) and model architectures (Granite-3B, Qwen2-MoE variants), quantifying the achievable memory-latency trade-off frontier.

- **Objective 4:** Provide architectural insights and design guidelines for practitioners seeking to deploy large MoE models on resource-constrained hardware, including memory-optimal capacity configurations and staging buffer tuning strategies.

---

## 1.5 Key Contributions

The CacheFlow system and accompanying evaluation contribute the following advances to the field:

**1. A Novel Three-Tier Memory Hierarchy with Asynchronous Staging:** 
CacheFlow introduces a principled architectural design that cleanly separates concerns across three memory tiers—disk, pinned host RAM, and GPU VRAM—with asynchronous non-blocking data transfers. This design is fundamentally different from prior work and achieves several key properties: (a) disk I/O is proportional only to a single expert's size, not the entire weight file; (b) pinned memory allocation is bounded by the number of staging buffers (typically 4–8), not the total expert count; (c) GPU cache capacity is independently tunable via the capacity constraint C without affecting host-level allocations.

**2. Intra-Batch Freezing for Correctness and Safety:** 
The `freeze_many()` mechanism provides a simple yet powerful correctness guarantee: experts required during a batch cannot be evicted while the batch is computing. This eliminates deadlocks, data races, and partial-result loss that would otherwise occur in naive asynchronous systems. The freezing protocol adds negligible overhead (~0.1% latency) while providing transparent safety semantics.

**3. Quantified Memory-Latency Trade-off Frontier:** 
Empirical evaluation demonstrates that CacheFlow achieves **5–9× reductions in peak GPU VRAM** compared to baseline systems across constrained capacity settings. Specifically:
- At **C = 1** (ultra-constrained, ~1 expert slot per MoE layer): **988 MB** peak VRAM, **598 ms/token** latency
- At **C = 8** (moderate constraint, ~8 expert slots per MoE layer): **1845 MB** peak VRAM, **462 ms/token** latency
- Throughput maintained at **100–130 tokens/minute** across all configurations, compared to near-zero throughput for baseline systems in constrained settings

This represents a practical Pareto frontier where practitioners can choose operating points balancing memory savings against acceptable latency overhead.

**4. Architectural Generalization Across Diverse MoE Models:** 
CacheFlow's design is model-agnostic and scales correctly across diverse MoE architectures. Evaluation on Granite-3B (8 experts, 48 MB each), primary test configuration (60 experts, 107 MB each), and analysis of Qwen2-MoE (64 experts, 256 MB each) demonstrates consistent performance characteristics. The system's effectiveness is shown to improve with larger expert sizes, making it particularly valuable for next-generation large-scale MoE models.

**5. Practical Deployment Enablement:** 
By reducing GPU memory requirements by an order of magnitude while maintaining reasonable latencies, CacheFlow enables deployment of production MoE models on consumer and mid-range GPUs. This democratizes access to state-of-the-art model architectures and reduces per-inference costs in datacenter environments through improved resource utilization.

---

## 1.6 Thesis Organization

The remainder of this thesis is structured as follows. **Chapter 2** provides a comprehensive literature review, discussing prior work in MoE model architectures, memory-efficient inference techniques, and disk-offloading systems, contextualizing CacheFlow's contributions within the broader research landscape. **Chapter 3** presents the complete methodological design of the CacheFlow system, detailing the three-tier memory hierarchy, the asynchronous execution model, the freezing mechanism, and the mathematical formulation of memory budgets and capacity constraints. **Chapter 4** reports comprehensive experimental results, comparing CacheFlow against baseline systems (DeepSpeed MoE and MoE-Infinity), analyzing the memory-latency trade-off across varying capacity constraints, evaluating cache hit rates and prefetching dynamics, and assessing generalization across diverse model architectures. **Chapter 5** concludes the thesis with a synthesis of key findings, discussion of system limitations and their implications, and identification of promising directions for future research. Throughout, emphasis is placed on rigorous quantitative evaluation, clarity of exposition, and practical applicability to deployment scenarios.

