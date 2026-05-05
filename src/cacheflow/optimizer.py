from __future__ import annotations
import collections
import gc
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.free_slots = list(range(capacity))

    def get_slot(self, expert_id: int) -> tuple[int, bool, Optional[int]]:
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
            return self.cache[expert_id], True, None

        if self.free_slots:
            slot_idx = self.free_slots.pop(0)
            self.cache[expert_id] = slot_idx
            return slot_idx, False, None

        victim_expert_id, slot_idx = self.cache.popitem(last=False)
        self.cache[expert_id] = slot_idx
        return slot_idx, False, victim_expert_id

class BufferManager:
    def __init__(
        self,
        num_slots: int,
        num_experts: int,
        output_size: int,
        input_size: int,
        dtype: torch.dtype,
        device: torch.device,
        weights_src: torch.Tensor,
        layer_id: str = "Unknown"
    ):
        self.num_slots = num_slots
        self.device = device
        self.dtype = dtype
        self.num_experts = num_experts
        self.layer_id = layer_id
        
        self.lru = LRUCache(num_slots)
        
        self.buffers = torch.empty((num_slots, output_size, input_size), dtype=dtype, device=device)
        self.pageable_weights = weights_src.detach().cpu().clone()

    def load_expert(self, expert_id: int, presentation_mode: bool = False) -> tuple[int, bool]:
        slot_idx, is_hit, victim_id = self.lru.get_slot(expert_id)

        if not is_hit:
            if presentation_mode:
                if victim_id is not None:
                    print(f"   Cache Full! Evicting Expert {victim_id} from VRAM to Host.")
                print(f"   Fetching Expert {expert_id} from Host RAM...")
            
            self.buffers[slot_idx].copy_(self.pageable_weights[expert_id], non_blocking=False)
        else:
            if presentation_mode:
                print(f"   Cache HIT: Expert {expert_id} is already resident in VRAM.")

        return slot_idx, is_hit

class GraniteMoeParallelExpertsOptimized(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int, num_gpu_slots: int = 3, layer_id: str = ""):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.layer_id = layer_id
        
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.buffer_manager: Optional[BufferManager] = None
        self.num_slots = min(max(1, num_gpu_slots), max(1, num_experts))
        self.presentation_mode = False

    def _init_buffer_manager(self, device: torch.device):
        if self.buffer_manager is None:
            self.buffer_manager = BufferManager(
                num_slots=self.num_slots,
                num_experts=self.num_experts,
                output_size=self.output_size,
                input_size=self.input_size,
                dtype=self.weight.dtype,
                device=device,
                weights_src=self.weight.data,
                layer_id=self.layer_id
            )
            # The Massive VRAM Drop happens here!
            self.weight.data = torch.empty(0, device=device, dtype=self.weight.dtype)

    def forward(self, inputs: torch.Tensor, expert_size: List[int]) -> torch.Tensor:
        self._init_buffer_manager(inputs.device)
        input_list = inputs.split(expert_size, dim=0)
        out_chunks: List[Optional[torch.Tensor]] = [None] * self.num_experts
        required_experts = [i for i, size in enumerate(expert_size) if size > 0]

        for i in range(self.num_experts):
            if input_list[i].numel() == 0:
                out_chunks[i] = input_list[i].new_empty((0, self.output_size))

        resident_experts = [eid for eid in required_experts if eid in self.buffer_manager.lru.cache]
        cold_experts = [eid for eid in required_experts if eid not in self.buffer_manager.lru.cache]
        execution_order = resident_experts + cold_experts

        should_log = self.presentation_mode and self.layer_id == "model.layers.0.block_sparse_moe.experts"
        if should_log and required_experts:
            print(f"\n[Token Routing] Layer 0 requested Experts: {required_experts}")

        for expert_id in execution_order:
            slot_idx, _ = self.buffer_manager.load_expert(expert_id, presentation_mode=should_log)
            expert_weight = self.buffer_manager.buffers[slot_idx]
            out_chunks[expert_id] = F.linear(input_list[expert_id], expert_weight)

        return torch.cat(out_chunks, dim=0)

def _replace_parallel_experts_in_module(module: nn.Module, num_gpu_slots: int, prefix: str = "") -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        is_parallel_experts = child.__class__.__name__ == "GraniteMoeParallelExperts"
        
        if is_parallel_experts and hasattr(child, "weight"):
            optimized = GraniteMoeParallelExpertsOptimized(
                num_experts=child.num_experts,
                input_size=child.input_size,
                output_size=child.output_size,
                num_gpu_slots=num_gpu_slots,
                layer_id=full_name
            )
            optimized.load_state_dict(child.state_dict(), strict=True)
            optimized.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, child_name, optimized)
            
            replaced += 1
        else:
            replaced += _replace_parallel_experts_in_module(child, num_gpu_slots, full_name)
    return replaced

def cacheflow_optimizer(
    model: nn.Module,
    num_gpu_slots: int = 3,
    presentation_mode: bool = False
) -> nn.Module:
    """Applies the CacheFlow Software-Level LRU Scheduler to a Granite MoE model."""
    replaced = _replace_parallel_experts_in_module(model, num_gpu_slots=num_gpu_slots)

    # ==========================================
    # FIX: EAGER INITIALIZATION
    # We explicitly force the modules to dump the weights to RAM *right now*.
    # ==========================================
    for name, module in model.named_modules():
        if isinstance(module, GraniteMoeParallelExpertsOptimized):
            module.presentation_mode = presentation_mode
            module_device = module.weight.device
            module._init_buffer_manager(module_device)

    # Hard flush PyTorch's memory allocator to ensure Cell 3 prints the exact drop
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n CacheFlow Optimizer Applied Successfully!")
    print(f" Replaced {replaced} Standard MoE blocks with CacheFlow Dynamic Schedulers.")
    print(f" Capacity limited to {num_gpu_slots} Active Experts per layer.")
    
    return model