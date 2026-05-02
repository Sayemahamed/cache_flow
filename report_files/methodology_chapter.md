# 3. METHODOLOGY

## 3.1 Overview of the Proposed Architecture

The CacheFlow system is designed to enable efficient Mixture-of-Experts (MoE) inference on GPU hardware with severe memory constraints. The core innovation extends beyond simple GPU caching: it implements a comprehensive three-tier memory hierarchy that seamlessly chains disk-resident expert weights through pinned CPU RAM staging to GPU VRAM slots via asynchronous, non-blocking PCIe transfers. This architecture fundamentally decouples GPU memory requirements from the total expert parameter count, enabling deployment of massive MoE models on resource-constrained hardware.

The proposed approach is a system-level optimization that does not modify the underlying MoE architecture or transformer model structure. Instead, it operates at the layer level through the replacement of standard parallel expert modules with an optimized implementation that incorporates direct disk I/O, intelligent memory staging, asynchronous prefetching, and temporal cache freezing mechanisms. Rather than maintaining all expert weights in host RAM (as in traditional approaches), CacheFlow leverages the observation that expert weight files are compactly stored on disk in the safetensors format. The system reads only the required expert weights directly from disk on-demand, stages them through CPU-pinned memory buffers, and transfers them asynchronously to GPU slots. A sophisticated caching mechanism with intra-batch freezing ensures that frequently-accessed experts remain resident in GPU memory while allowing cold experts to be loaded and accommodated within a fixed capacity constraint.

This three-tier architecture—Disk, Pinned RAM, and GPU VRAM—creates a memory-performance tradeoff landscape where capacity can be tuned per-application. The system maintains computational correctness while achieving dramatic reductions in GPU memory footprint and enabling models previously inaccessible to resource-constrained environments.

---

---

## 3.2 System Architecture and Memory Pipeline

### 3.2.1 Three-Tier Memory Hierarchy

The CacheFlow system explicitly organizes expert weight storage and access across three distinct memory tiers:

**Tier 1: Disk Storage (Persistent):**
- Authoritative storage medium for all expert weight matrices
- Expert weights are persisted in safetensors format, a compressed binary serialization enabling efficient random access
- Access is performed via `MmapExpertStore`, which implements zero-copy slicing of expert weight tensors
- Large capacity (model-dependent, typically 10–100 GB for large MoE models)
- Slow access bandwidth (approximately 100–500 MB/s depending on storage medium)
- No GPU involvement; CPU reads weights directly from disk

**Tier 2: Pinned Host Memory (Staging Buffer):**
- Fixed-size set of CPU-resident memory buffers allocated with CUDA page-locking semantics
- Serves as a temporary staging area for weights being loaded from disk
- Each staging buffer holds a single expert's weight matrix
- Pinned allocation allows asynchronous, zero-copy PCIe transfers to GPU without CPU intervention
- Typically allocated with `torch.cuda.pin_memory()` or platform-specific pinning APIs
- Number of staging buffers (`staging_slots`) is a configurable hyperparameter (e.g., 8 or 16)
- Acts as a decoupling point between slow disk I/O and fast GPU transfer

**Tier 3: GPU VRAM (Working Set):**
- Reserved for active computation and inference
- Contains: model static parameters (embeddings, attention layers, output head), intermediate activations, and a fixed set of expert weight slots in GPU memory
- Limited capacity typically between 6 GB and 24 GB on consumer-grade hardware
- Ultra-high-bandwidth access (up to 900 GB/s on modern NVIDIA GPUs)
- Expert slots are implemented as a preallocated `torch.Tensor` of shape `(num_gpu_slots, output_size, input_size)`

### 3.2.2 Direct Disk I/O via MmapExpertStore

The `MmapExpertStore` class abstracts expert weight access directly from disk storage. Unlike traditional approaches that load entire weight matrices into RAM, MmapExpertStore implements memory-efficient slicing:

**Initialization Phase:**
```
store = MmapExpertStore(weight_file)
```

The constructor calls `_read_metadata()`, which inspects the safetensors file header to determine:
- The tensor key associated with expert weights (typically "model.experts.weight" or equivalent)
- The tensor shape `(num_experts, output_size, input_size)`
- The tensor data type (float32, float16, or int8)

This metadata reading is O(1) in complexity and requires only header inspection; no weight data is loaded into memory.

**Per-Expert Loading:**
```
expert_weight = store.load_expert(expert_id, out=staging_buffer)
```

The `load_expert()` method executes the following steps:

1. **Bounds Checking**: Verify that `0 <= expert_id < num_experts`.

2. **Safetensors Slicing**: Open the safetensors file and invoke `get_slice()` to extract only the weight matrix corresponding to `expert_id`. Safetensors' slice operation reads precisely `output_size × input_size × dtype_bytes` bytes from disk, skipping the preceding experts' data.

3. **Buffer Population**: Copy the sliced weights into the provided `out` buffer (a pinned memory allocation). This copy operation uses efficient CPU memory operations but does not transfer data to GPU at this stage.

4. **Return**: Return the populated buffer tensor.

The key advantage is that disk I/O is proportional to a single expert's size, not the entire weight file. For a model with 60 experts of 100 MB each (6 GB total), loading one expert reads only 100 MB from disk.

---

### 3.2.3 Pinned Memory Staging and Async PCIe Transfer

The `_next_staging()` method in `GraniteMoeParallelExperts` manages the lifecycle of pinned memory buffers:

```python
def _next_staging(self) -> Optional[torch.Tensor]:
    # Returns an available pinned buffer, or waits for one to be free
    # After the buffer is used for disk I/O, it is immediately scheduled for
    # asynchronous transfer to GPU. The system waits for prior transfers if needed.
```

The staging pipeline operates as follows:

1. **Buffer Availability Check**: Maintain a list of pinned buffers. If all buffers are currently in-flight (pending PCIe transfers), the system blocks until one completes.

2. **Synchronization Point**: Before reusing a pinned buffer, ensure that any prior PCIe transfer using that buffer has completed. This is tracked via `torch.cuda.Event` objects.

3. **Return Buffer**: Return an available pinned buffer for the next disk read operation.

After a weight is loaded from disk into a pinned buffer, an asynchronous PCIe transfer is scheduled using `torch.cuda.Stream`:

```python
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    gpu_slot[:] = pinned_buffer  # Copy via PCIe
    event = torch.cuda.Event()
    event.record(stream)  # Record completion in stream
pending[expert_id] = (event, gpu_slot)  # Track pending transfer
```

This design allows multiple experts' weights to be loaded from disk and transferred to GPU concurrently. While Expert A's weights are being read from disk by the CPU, Expert B's previously-staged weights are being transferred to GPU via PCIe, and Expert C is being computed on the GPU. This overlapping of I/O, transfer, and computation dramatically reduces overall latency.

### 3.2.4 ExpertCache: GPU-Resident Cache with Freezing

The `ExpertCache` class manages a fixed-size LRU cache of experts in GPU memory. Beyond standard LRU operations, it introduces a critical freezing mechanism to prevent intra-batch conflicts:

**Cache State:**
- `cache: OrderedDict[int, torch.Tensor]`: Maps expert IDs to their GPU-resident weight tensors
- `_frozen: set[int]`: Set of expert IDs that are protected from eviction (locked during current batch processing)
- `capacity: int`: Maximum number of experts that can be cached simultaneously

**Freezing Mechanism:**
```python
def freeze_many(self, expert_ids: List[int]) -> None:
    self._frozen.update(expert_ids)

def unfreeze_many(self, expert_ids: List[int]) -> None:
    for expert_id in expert_ids:
        self._frozen.discard(expert_id)
```

At the beginning of a forward pass, the set of required experts (determined by token routing) is frozen in the cache. This ensures that if the system must fetch an expert that is not cached (a cold expert), it cannot accidentally evict a required expert that is already cached. Freezing is released at the end of the batch, allowing normal LRU eviction to proceed for the next batch.

[INSERT FIGURE 1: System Memory Hierarchy Diagram Here]

**Figure Caption (Figure 1):** The three-tier CacheFlow memory hierarchy illustrates the complete data pathway from persistent disk storage through pinned CPU memory staging to GPU VRAM. Expert weights are stored compactly in safetensors format on disk. The MmapExpertStore reads specific expert slices on-demand. Pinned memory buffers stage the weights for asynchronous transfer. The GPU cache maintains the working set of experts. The diagram shows multiple experts at different stages: some resident in GPU cache, some pending transfer to GPU, and some available for disk loading.

---

## 3.3 Mathematical Formulation of CacheFlow Routing

### 3.3.1 Memory Budget Decomposition with Three-Tier Architecture

The total GPU memory consumption for an optimized MoE model is decomposed across the three-tier hierarchy:

$$M_{VRAM} = M_{static} + M_{cache} + M_{KV} + M_{act}$$

Where:

- $M_{static}$: Memory consumed by static model parameters (embeddings, attention layers, output head, layer normalization) that remain permanently in GPU memory
- $M_{cache}$: Memory allocated to expert weight slots in GPU VRAM (the working set)
- $M_{KV}$: Memory for key-value caches in attention mechanisms (if applicable)
- $M_{act}$: Memory for intermediate activations during forward computation

The critical innovation is that $M_{cache}$ is now **independently constrained** from the total expert parameter count. The cache memory is governed by:

$$M_{cache} = L_{MoE} \times C \times P_{expert} \times b_o$$

Where:

- $L_{MoE}$: Number of MoE layers in the model
- $C$: Capacity constraint—the maximum number of expert slots in GPU VRAM per layer (e.g., $C = 4$ means at most 4 experts reside in GPU VRAM per layer simultaneously)
- $P_{expert}$: The size of a single expert weight matrix (in number of parameters)
- $b_o$: The byte size per parameter (typically 2 bytes for float16, 4 bytes for float32)

### 3.3.2 Host Pinned Memory Footprint for Staging

The pinned memory staging buffer allocation is distinct from both GPU memory and disk storage:

$$M_{Pinned} = L_{MoE} \times S_{staging} \times P_{expert} \times b_o$$

Where:

- $S_{staging}$: The number of staging buffers allocated (e.g., $S_{staging} = 8$)

Critically, $S_{staging}$ is typically much smaller than the total number of experts ($S_{slots}$). While $S_{slots}$ may be 60 or more, $S_{staging}$ is often 4–16. This design allows the system to pipeline disk I/O and PCIe transfers without requiring proportional pinned memory allocation.

### 3.3.3 Disk Storage and the Complete Memory Budget

The disk storage requirement remains fixed and proportional only to the total expert count:

$$M_{Disk} = L_{MoE} \times S_{slots} \times P_{expert} \times b_o$$

Where:

- $S_{slots}$: The total number of experts across the entire MoE layer (e.g., 60 for models like Qwen2-MoE)

This completes the three-tier budget:

$$M_{Total} = M_{VRAM} + M_{Pinned} + M_{Disk}$$

The critical decoupling is evident:

- $M_{VRAM}$ depends only on $C$ (GPU capacity constraint)
- $M_{Pinned}$ depends only on $S_{staging}$ (number of staging buffers)
- $M_{Disk}$ depends on $S_{slots}$ (total expert count) but is not constrained by hardware resources

### 3.3.4 Memory Reduction Formula

By controlling capacity constraints $C$ and staging buffers $S_{staging}$, CacheFlow enables deployment of large expert models on memory-constrained GPUs:

$$\text{VRAM Reduction} = 1 - \frac{C}{S_{slots}}$$

For example, with $C = 4$ and $S_{slots} = 60$:

$$\text{VRAM Reduction} = 1 - \frac{4}{60} = 93.3\%$$

This represents a 93.3% reduction in expert-related GPU memory consumption compared to loading all experts simultaneously.

### 3.3.5 Dynamic Cache Buffer Definition

The CacheFlow dynamic buffer allocation is mathematically defined as:

$$M_{cache} = L_{MoE} \times C \times P_{expert} \times b_o$$

The host RAM pinned staging footprint is:

$$M_{Pinned} = L_{MoE} \times S_{staging} \times P_{expert} \times b_o$$

Together, these formulas show how the system decouples the total parameter count from the VRAM requirement by controlling the capacity $C$.

### 3.3.6 Token Routing and Cache Hit Efficiency

The MoE routing mechanism determines which experts process each token. A gating network produces a distribution over all experts for each input token. The routing typically selects $K$ experts per token (e.g., $K = 2$) based on gating logits.

The set of required experts for a batch is defined as:

$$\text{Required\_Experts} = \{i : \text{expert\_size}[i] > 0\}$$

The efficiency of CacheFlow depends directly on the overlap between required experts and the cache capacity:

$$\text{Cache Hit Rate} = \frac{|\text{Required\_Experts} \cap \text{Cached\_Experts}|}{|\text{Required\_Experts}|}$$

Temporal locality in expert access patterns—where recent tokens tend to route to overlapping sets of experts—is exploited to maximize hit rates within a fixed capacity $C$.

---

---

## 3.4 Asynchronous Execution Flow

### 3.4.1 Forward Pass Initialization and Batch Setup

When a forward pass is invoked on a `GraniteMoeParallelExperts` module, the following initialization sequence executes:

1. **Routing Distribution Computation**: The gating network computes expert routing logits for all input tokens, producing a distribution over experts.

2. **Required Experts Identification**: The set of experts required for the current batch is determined:

   ```
   required_experts = [i for i, size in enumerate(expert_size) if size > 0]
   ```

   This operation partitions the input tensor by expert assignment, creating sub-tensors for each expert.

3. **Cache Freezing**: All required experts are locked in the cache to prevent accidental eviction:

   ```python
   cache.freeze_many(required_experts)
   ```

   This ensures that while fetching cold experts, the system cannot evict required experts that are already cached.

4. **Residency Classification**: The system partitions required experts into two classes:

   **Cached Experts** (Cache Hits):
   ```
   cached_experts = [e for e in required_experts if e in cache.cache]
   ```

   **Cold Experts** (Cache Misses):
   ```
   cold_experts = [e for e in required_experts if e not in cache.cache]
   ```

### 3.4.2 Asynchronous Prefetching and Pending Transfers

For cold experts, the system initiates asynchronous loading:

```python
pending = {}  # expert_id -> (torch.cuda.Event, gpu_slot)
```

For each cold expert, the following pipeline is initiated:

1. **Disk Read (CPU)**: Retrieve an available staging buffer and load the expert's weights from disk:

   ```python
   staging_buffer = _next_staging()  # Block if all buffers are in-flight
   store.load_expert(expert_id, out=staging_buffer)
   ```

   The `load_expert()` call reads the expert slice from disk using safetensors, blocking the CPU thread until the read completes.

2. **Asynchronous Transfer Scheduling (GPU)**: Schedule a non-blocking PCIe transfer from pinned memory to GPU:

   ```python
   stream = torch.cuda.Stream()
   with torch.cuda.stream(stream):
       gpu_slot[:] = staging_buffer  # Non-blocking copy to GPU slot
       event = torch.cuda.Event()
       event.record(stream)  # Record completion event
   pending[expert_id] = (event, gpu_slot)
   ```

   The key insight is that `gpu_slot[:] = staging_buffer` is asynchronous; the GPU's DMA engine handles the transfer without blocking CPU or GPU compute. The system does not wait for completion.

3. **Overlap with Computation**: While cold experts' weights are being transferred asynchronously:

   - Cached experts begin their forward pass computations on the GPU
   - The CPU may initiate disk reads for other cold experts' weights
   - Multiple experts' transfers occur concurrently via PCIe

### 3.4.3 Synchronization Before Expert Computation

Before computing a cold expert's contribution, the system must ensure its weights have arrived in GPU memory:

```python
if expert_id in pending:
    event, gpu_slot = pending[expert_id]
    event.synchronize()  # Wait for PCIe transfer to complete
    out_chunks[expert_id] = F.linear(input_list[expert_id], gpu_slot)
    del pending[expert_id]
else:
    # Expert was already cached; compute immediately
    expert_weight = cache.cache[expert_id]
    out_chunks[expert_id] = F.linear(input_list[expert_id], expert_weight)
```

The `event.synchronize()` call blocks CPU execution until the GPU's DMA transfer completes. This ensures that the expert's weights are resident in GPU memory before computation begins.

### 3.4.4 Execution Ordering for Latency Minimization

The execution order is strategically chosen to minimize wait time:

```
execution_order = cached_experts + cold_experts
```

This ordering ensures that:
1. Cached experts complete immediately (no transfer delay)
2. Cold experts are processed sequentially after their asynchronous transfers complete
3. By the time the system reaches a cold expert, its weights have had maximum time to transfer from disk → staging → GPU

[INSERT FIGURE 2: Asynchronous Execution Flow and Prefetching Diagram Here]

**Figure Caption (Figure 2):** The asynchronous execution flow demonstrates the overlap of disk I/O, PCIe transfer, and GPU computation. At Time T0, Expert A is being read from disk, Expert B is being transferred via PCIe, and Expert C is being computed on the GPU. The pending dictionary tracks in-flight transfers. Cache freezing ensures required experts are protected. The diagram shows the temporal evolution of multiple experts through the pipeline: Disk I/O → Pinned Staging → PCIe Transfer → GPU Computation.

### 3.4.5 Output Concatenation and Cache Unfreezing

Once all required experts have been processed:

```python
output = torch.cat(out_chunks, dim=0)
cache.unfreeze_many(required_experts)
```

The output tensor is reconstructed by concatenating expert outputs in the original batch order. Cache freezing is released, allowing normal LRU eviction to proceed for the next batch.

---

## 3.5 Advanced LRU Cache Management and Freezing

### 3.5.1 Cache State Representation with Freezing

The `ExpertCache` class extends standard LRU caching with a freezing mechanism:

```python
cache: OrderedDict[int, torch.Tensor]  # expert_id -> gpu_weight
_frozen: set[int]  # expert_ids that cannot be evicted this batch
```

The OrderedDict maintains insertion and access order. When an expert is accessed, its key is moved to the end, marking it as "most recently used." The first item in the order represents the "least recently used" candidate for eviction.

### 3.5.2 Cache Hit Path

When `get(expert_id)` is called and the expert is cached:

```python
def get(self, expert_id: int) -> Optional[torch.Tensor]:
    if expert_id not in self.cache:
        return None
    self.cache.move_to_end(expert_id)
    return self.cache[expert_id]
```

This operation:
1. Checks for presence in O(1) time
2. Moves the expert to the end of the OrderedDict (mark as recently used)
3. Returns the cached weight tensor without any data movement

Cache hits incur no PCIe transfers or disk I/O.

### 3.5.3 Cache Miss with Available Free Slots

When `put(expert_id, gpu_weight)` is called and free cache slots are available:

```python
if len(self.cache) < self.capacity:
    self.cache[expert_id] = gpu_weight
```

The weight tensor is simply added to the cache. The new expert becomes the "most recently used" entry.

### 3.5.4 Cache Miss with Full Capacity (Eviction)

When the cache is at capacity and a new expert must be added, the least recently used unfrozen expert is evicted:

```python
if len(self.cache) >= self.capacity:
    # Attempt to evict unfrozen experts in LRU order
    for victim_id in list(self.cache.keys()):
        if victim_id not in self._frozen:
            del self.cache[victim_id]
            break
    # If all cached experts are frozen, block (rare case)
    if len(self.cache) >= self.capacity:
        raise RuntimeError("Cache full and all experts are frozen")

self.cache[expert_id] = gpu_weight
```

**Critical Detail - Freezing Protection:**
The loop iterates through the cache in LRU order (from least recently used to most recently used). It skips any expert that is currently frozen (in the `_frozen` set). Only unfrozen experts are candidates for eviction.

This design ensures that **within a batch, no required expert is accidentally evicted**. The freezing mechanism guarantees that if Expert 5 is required by the current batch (and therefore cached or about to be cached), it cannot be evicted to make room for Expert 4, even if Expert 5 is less recently used.

Once an expert is evicted from the GPU cache:
- Its weight tensor is discarded from GPU memory
- Its entry is removed from the cache dictionary
- If the expert is needed in a future batch, it will be reloaded from disk (cache miss)

The evicted expert's weight data remains safely on disk; no writeback is necessary.

---

### 3.5.5 Complete Cache Lifecycle During Forward Pass

The complete cache lifecycle during a single forward pass is:

1. **Freeze Phase**: All required experts are frozen before any transfers begin.

2. **Fetch Phase**: Cold experts are fetched asynchronously. The cache may evict unfrozen experts (those not required by the current batch) to make room.

3. **Compute Phase**: Experts are computed in order. Cached experts compute immediately; cold experts wait for their asynchronous transfers to complete.

4. **Unfreeze Phase**: At the end of the batch, all required experts are unfrozen, allowing them to be evicted if the next batch does not require them.

This cycle repeats for each forward pass (each token or batch of tokens in generation).

[INSERT FIGURE 3: Advanced LRU Cache and Freezing Mechanism Diagram Here]

**Figure Caption (Figure 3):** The advanced LRU cache with freezing mechanism shows how intra-batch protection works. Required experts are frozen at the beginning of the batch (shown in green with lock icons). When a cache miss occurs, the system searches for unfrozen experts to evict (shown in red dashed lines). Frozen experts cannot be evicted, ensuring their protection. The diagram shows a cache state where Expert 2 and Expert 9 are frozen (required by current batch), while Expert 5 is unfrozen and thus is the candidate for eviction when Expert 4 is loaded.

---

## 3.6 Implementation Environment

### 3.6.1 Software Stack

The CacheFlow implementation is developed in **Python 3.8+** using the following core libraries:

- **PyTorch (1.13+)**: Provides tensor operations, GPU memory management, CUDA stream and event handling, and the foundation for all neural network computations. Specifically, `torch.cuda.Stream()` and `torch.cuda.Event()` are used for asynchronous transfer coordination.

- **Safetensors (0.3.1+)**: Provides the `safe_open()` context manager and `get_slice()` method for efficient, zero-copy extraction of expert weight slices from disk-resident weight files.

- **Transformers Library (Hugging Face, 4.0+)**: Enables loading of pretrained MoE model configurations (e.g., IBM Granite, Qwen2-MoE) and tokenization utilities.

- **CUDA Toolkit (11.8+)**: Provides GPU compute, memory management, and PCIe DMA capabilities.

- **Accelerate Library (0.20+)**: Simplifies multi-device inference and model loading strategies.

### 3.6.2 Hardware Configuration

The implementation is validated on the following hardware configurations:

**Development and Experimentation Environment:**
- GPU: NVIDIA T4 (16 GB VRAM) or NVIDIA H100 (80 GB VRAM) on cloud platforms (Google Colab, Lambda Labs)
- CPU: Standard x86-64 processor with 16–64 GB system RAM, PCIe connection to GPU
- Storage: NVMe SSD (for fast disk access, enabling efficient `MmapExpertStore` operations) or network-attached storage
- Interconnect: PCIe 4.0 or PCIe 3.0, providing theoretical bandwidth of 16–32 GB/s for GPU transfers

**Target Deployment Hardware:**
- Consumer-grade GPUs: NVIDIA RTX 3060 (12 GB), RTX 4070 (12 GB), RTX 4090 (24 GB)
- Embedded Systems: NVIDIA Jetson Orin (12 GB VRAM, shared with CPU)
- Storage: Local SSD or external USB 3.1 storage (minimum 500 MB/s read bandwidth recommended)
- System RAM: 32–64 GB minimum for pinned memory allocation and staging buffers

### 3.6.3 Model Under Test

The evaluation focuses on models implementing the `GraniteMoeParallelExperts` architecture. For this work, primary evaluation uses:

**IBM Granite 3.1-3B-A800M-Instruct:**
- Architecture: Transformer-based with sparse MoE routing in specific layers
- Total Parameters: Approximately 3 billion
- MoE Structure: Multiple MoE layers, each with 60 experts
- Expert Dimensions: Typically 2048 hidden dimension with 8192 intermediate dimension
- Expert Dtype: float16 (half precision) or int8 (quantized)
- Inference Mode: Causal language modeling (next-token prediction)
- Weight File Format: safetensors (highly compressed, optimized for slice access)

The choice of this model class allows controlled evaluation on realistic, production-intent MoE architectures while remaining accessible for undergraduate research efforts.

### 3.6.4 Experimental Protocol

The validation of CacheFlow follows a structured protocol:

1. **Baseline Measurement**: Load the model with naive expert loading (all experts in GPU memory or standard disk-loading without staging). Measure:
   - Peak GPU memory consumption (via `torch.cuda.max_memory_allocated()`)
   - Inference latency (time per token generated, averaged over 50 tokens)
   - Throughput (tokens per second)
   - Out-of-memory errors (if applicable)

2. **Configuration Space Exploration**: Apply the CacheFlow system with varying capacity constraints:

   - GPU Slot Capacity: $C \in \{2, 4, 8, 16\}$
   - Staging Buffers: $S_{staging} \in \{4, 8, 16\}$

3. **Post-Optimization Measurement**: For each configuration, measure:
   - Peak GPU memory consumption (via `torch.cuda.max_memory_allocated()`)
   - Inference latency (time per token)
   - Throughput (tokens per second)
   - Cache hit rate (cached experts / total expert requests)
   - Number of evictions (total cache misses requiring disk reload)
   - Disk I/O time (time spent in `store.load_expert()`)
   - PCIe transfer time (time spent in `event.synchronize()` waits)

4. **Inference Tasks**:
   - **Short-form**: Generate 20 new tokens for the prompt "Briefly explain how Mixture-of-Experts models work."
   - **Long-form**: Generate 100 new tokens for the prompt "Write a comprehensive essay on the architectural differences between standard Transformers and Mixture-of-Experts models."

   These tasks allow observation of cache behavior over both short and extended sequences.

5. **Metrics Logging**: All metrics are collected using:
   - PyTorch's CUDA profiling APIs: `torch.cuda.max_memory_allocated()`, `torch.cuda.synchronize()`
   - Custom timing instrumentation around `store.load_expert()`, `_next_staging()`, and `event.synchronize()` calls
   - Cache statistics from `ExpertCache` (hit/miss counters, frozen set size)

### 3.6.5 Correctness Validation

Numerical correctness is validated by comparing the output logits of the optimized model against a reference baseline:

1. **Deterministic Execution**: For a fixed input prompt and random seed, both baseline and CacheFlow-optimized models are executed in deterministic mode (disabling CUDA's non-deterministic operations where possible).

2. **Token-Level Comparison**: The output token sequences are compared for exact match, ensuring that:
   - Routing decisions are identical (same experts selected for each token)
   - Expert computations produce identical results
   - Output logits differ only by floating-point rounding errors

3. **Numerical Tolerance**: Logit-level numerical precision is verified to be within floating-point tolerance:
   - Float32: Relative error < 1e-5
   - Float16: Relative error < 1e-3

4. **Inference Stability**: The system is tested on 10 different input prompts to ensure correctness is consistent across varying routing patterns.

This validation confirms that the CacheFlow scheduling mechanism—including disk loading, staging, asynchronous transfer, freezing, and eviction—does not introduce computational errors or alter the model's inference behavior.

---

---

## Summary

The CacheFlow methodology presents a comprehensive system-level approach to enabling efficient MoE inference on memory-constrained GPUs through a novel three-tier memory hierarchy. By implementing direct disk-to-GPU pipelined data movement through pinned memory staging and asynchronous PCIe transfers, the system decouples GPU memory requirements from the total expert parameter count. The introduction of cache freezing ensures correctness in dynamic, batch-dependent scenarios where required experts must be protected from accidental eviction. Mathematical formulation clarifies how capacity constraints control GPU memory consumption independently of total expert count. Asynchronous execution flow demonstrates the overlapping of disk I/O, PCIe transfer, and GPU computation to minimize latency. Advanced LRU cache management with intra-batch freezing ensures efficient cache utilization while maintaining safety guarantees. The implementation is grounded in production-intent MoE architectures and validated on realistic hardware configurations, demonstrating practical applicability for deploying large MoE models on resource-constrained systems that would otherwise be unable to accommodate such models.

