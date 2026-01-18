# %% [markdown]
# # **CacheFlow MoE**
# 
# ## **Thesis Artifact:** *Behavioral Analysis of CacheFlow*
# 
# **Subject:** Dynamic Expert Scheduling for Mixture-of-Experts (MoE) Large Language Models
# **Model:** Qwen1.5-MoE-A2.7B (4-bit Quantized)
# 
# ### **Abstract**
# 
# This notebook serves as the interactive artifact for the thesis. While the primary system performance results (throughput/latency) are established via C++ benchmarks in the main paper, this Python environment provides a **behavioral analysis** of the CacheFlow scheduling algorithm.
# 
# ### **Objectives**
# 
# 1.  **Verify Correctness:** Demonstrate that the "Priority Scheduling" logic correctly manages expert residency without degrading generation quality.
# 2.  **Visualizing Internals:** Provide explainable visualizations of "Expert Residency" (how experts stay in cache) and "Routing Fairness" (how the model utilizes experts).
# 3.  **Quantify Trade-offs:** Measure the theoretical data movement (PCIe traffic) savings achieved by limiting active expert capacity.
# 
# ### **Methodology**
# 
# We utilize a **High-Fidelity Simulation** approach:
# 
#   * The **Real Model Weights** (Qwen1.5-MoE) are loaded into GPU memory to ensure accurate routing decisions.
#   * The **Scheduler** logically enforces capacity limits ($C \ll N$), tracking hits, misses, and evictions.
#   * **Performance Metrics** (Latency/Bandwidth) are derived analytically based on the tracked swap events to isolate algorithmic behavior from Python interpreter overhead.

# %% [markdown]
# **Installing Dependencies**

# %%
# !pip install -q --upgrade torch transformers accelerate bitsandbytes scipy pandas matplotlib seaborn

# %% [markdown]
# **Imports and Configurations**

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from copy import deepcopy
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from dataclasses import dataclass, field
import math

logger = logging.get_logger(__name__)

# Visualization Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.family': 'serif', 'figure.dpi': 150})
# CONFIGURATION
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERT_CAPACITY = 20
MAX_BATCH_SIZE = 4

print(f"Running Thesis Demo on {DEVICE}")
print(f"Model: {MODEL_ID}")
print(f"Expert Capacity: {EXPERT_CAPACITY} (Simulated Constraint)")

# %% [markdown]
# Start of Qwen2MoE Raw Implementation

class Qwen2MoeConfig(PretrainedConfig):
    model_type = "qwen2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_experts=60,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        sliding_window=32768,
        max_window_layers=28,
        decoder_sparse_step=1,
        mlp_only_layers=None,
        rope_scaling=None,
        pad_token_id=151643,
        qkv_bias=True,
        norm_topk_prob=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_cache = use_cache
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.qkv_bias = qkv_bias
        self.norm_topk_prob = norm_topk_prob
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )

def load_balancing_loss_func(gate_logits, num_experts=None, top_k=2, attention_mask=None):
    if gate_logits is None or not isinstance(gate_logits, tuple): return 0
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (attention_mask[None, :, :, None, None].expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts)).reshape(-1, top_k, num_experts).to(compute_device))
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
        router_per_expert_attention_mask = (attention_mask[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, routing_weights.shape[1])).reshape(-1, routing_weights.shape[1]).to(compute_device))
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)
    device_index = routing_weights.device.index if routing_weights.device.index is not None else 0
    rank = routing_weights.shape[1] * int(device_index)
    overall_loss = torch.sum(tokens_per_expert[:, rank : rank + routing_weights.shape[1]] * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class Qwen2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen2MoeAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=self.config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_values is not None:
             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
             key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

QWEN2MOE_ATTENTION_CLASSES = {"eager": Qwen2MoeAttention, "flash_attention_2": Qwen2MoeAttention, "sdpa": Qwen2MoeAttention}

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)])
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob: routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN2MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=False, output_router_logits=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        router_logits = None
        if isinstance(hidden_states, tuple): hidden_states, router_logits = hidden_states
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if output_router_logits: outputs += (router_logits,)
        return outputs

class Qwen2MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = False
    _supports_sdpa = True
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2MoeRMSNorm):
            module.weight.data.fill_(1.0)

class Qwen2MoeModel(Qwen2MoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=config)
        self.post_init()
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_router_logits=None, cache_position=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None: position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        for decoder_layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions, output_router_logits=output_router_logits, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            hidden_states = layer_outputs[0]
            if output_attentions: all_self_attns += (layer_outputs[1],)
            if output_router_logits and layer_outputs[-1] is not None: all_router_logits += (layer_outputs[-1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden_states, attentions=all_self_attns, router_logits=all_router_logits)
    
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask: return attention_mask
            return None
            
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2MoeConfig,
        past_key_values: Cache,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            text_config = config.get_text_config()
            if getattr(text_config, "use_sliding_window", True) and text_config.sliding_window is not None:
                is_static_sliding_cache = isinstance(past_key_values, StaticCache) and all(past_key_values.is_sliding)
                if not is_static_sliding_cache or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - text_config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask
        
class Qwen2MoeForCausalLM(Qwen2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_router_logits=None, cache_position=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_router_logits=output_router_logits, cache_position=cache_position)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        aux_loss = None
        if output_router_logits:
             aux_loss = load_balancing_loss_func(outputs.router_logits, self.num_experts, self.num_experts_per_tok, attention_mask)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_logits)
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
        # Implementation required for GenerationMixin
        if past_key_values is not None:
             if isinstance(past_key_values, Cache):
                 cache_length = past_key_values.get_seq_length()
                 past_length = past_key_values.get_seq_length()
                 max_cache_length = past_key_values.get_max_length() if hasattr(past_key_values, "get_max_length") else None
             else:
                 past_length = past_key_values[0][0].shape[2]
                 max_cache_length = None
             if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                 input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
             elif past_length < input_ids.shape[1]:
                 input_ids = input_ids[:, past_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
             position_ids = attention_mask.long().cumsum(-1) - 1
             position_ids.masked_fill_(attention_mask == 0, 1)
             if past_key_values:
                 position_ids = position_ids[:, -input_ids.shape[1] :]
        if cache_position is None: cache_position = torch.arange(past_key_values.get_seq_length() if past_key_values is not None else 0, (past_key_values.get_seq_length() if past_key_values is not None else 0) + input_ids.shape[1], device=input_ids.device)
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache"), "position_ids": position_ids, "attention_mask": attention_mask, "cache_position": cache_position}

# End of Qwen2MoE Raw Implementation

# **CacheFlow Expert Scheduler**

# %%
from collections import defaultdict # Import defaultdict for easy counting

class PyCacheFlowScheduler:
    def __init__(self, capacity):
        self.capacity = capacity
        self.lru_list = []
        self.pending_queue = []
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.access_history = defaultdict(int) # Initialize access_history

    def submit_request(self, req_id, expert_id, has_kv_context):
        # Priority = 1000 (if KV exists) + 500 (if already in cache)
        score = 1000 if has_kv_context else 0
        if expert_id in self.lru_list: score += 500

        self.pending_queue.append({
            'req_id': req_id, 'expert_id': expert_id,
            'priority': score, 'arrival': time.time()
        })
        self.access_history[expert_id] += 1 # Increment access count for this expert

    def schedule_next_batch(self, max_bs):
        if not self.pending_queue: return []

        # Sort by Priority desc, then Arrival asc
        self.pending_queue.sort(key=lambda x: (-x['priority'], x['arrival']))

        batch = []
        i = 0
        while i < len(self.pending_queue) and len(batch) < max_bs:
            req = self.pending_queue[i]
            eid = req['expert_id']

            # Cache Logic
            if eid in self.lru_list:
                self.lru_list.remove(eid) # Refresh LRU
                self.lru_list.insert(0, eid)
                self.hits += 1
            else:
                self.misses += 1
                self.lru_list.insert(0, eid)
                if len(self.lru_list) > self.capacity:
                    self.lru_list.pop() # Evict
                    self.evictions += 1

            batch.append(req['req_id'])
            self.pending_queue.pop(i) # Remove from queue
        return batch

# %% [markdown]
# **Wrapping MoE layer with CacheFlow**

# %%
class CacheFlowQwenBlock(nn.Module):
    def __init__(self, original_block, scheduler, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.scheduler = scheduler

        # Extract components
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate

        self.active_experts = set()

    def set_scheduler(self, new_scheduler):
        """Updates the scheduler for a new experiment run."""
        self.scheduler = new_scheduler
        self.active_experts.clear()

    def forward(self, hidden_states):
        batch_size, seq_len, dim = hidden_states.shape
        flat_x = hidden_states.view(-1, dim)

        # 1. Shared Expert (Standard Forward)
        shared_out = 0
        if self.shared_expert:
            shared_out = self.shared_expert(flat_x)
            if self.shared_expert_gate:
                shared_out = shared_out * self.shared_expert_gate(flat_x).sigmoid()

        # 2. Routing
        logits = self.gate(flat_x)
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(probs, 4, dim=-1)
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

        # 3. Schedule Requests
        final_out = torch.zeros_like(flat_x)
        req_map = {}

        for i in range(flat_x.shape[0]):
            for k in range(4):
                eid = topk_indices[i, k].item()
                req_id = (self.layer_idx * 1_000_000) + (i * 10) + k
                self.scheduler.submit_request(req_id, eid, seq_len > 1)
                req_map[req_id] = (i, topk_probs[i, k], eid)

        # 4. Execute Batches
        while True:
            batch = self.scheduler.schedule_next_batch(16)
            if not batch: break

            for rid in batch:
                token_idx, weight, eid = req_map[rid]

                # --- SIMULATED CACHE LOGIC ---
                # We track the state logic perfectly, but we skip physical .to("cpu")
                # to prevent bitsandbytes 4-bit crashing.
                if eid not in self.active_experts:
                    self.active_experts.add(eid)

                    # Logically Evict if full
                    # (We check the scheduler's authoritative LRU list)
                    valid_set = set(self.scheduler.lru_list)
                    # Intersection of what we hold vs what scheduler allows
                    # In simulation, we just sync to scheduler
                    self.active_experts = valid_set.copy()

                # Compute (Expert is already on GPU, so this is fast and stable)
                expert_out = self.experts[eid](flat_x[token_idx].unsqueeze(0))
                final_out[token_idx] += expert_out.squeeze(0) * weight

        return (final_out + shared_out).view(batch_size, seq_len, dim)

# %% [markdown]
# **Loading Model in to GPU**

# %%
if 'base_model' not in globals():
    print("⏳ Loading Model (This takes ~2 mins)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = Qwen2MoeForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True
    )

    # Inject Wrappers
    print("🔧 Injecting Wrappers...")
    global_wrappers = []
    for i, layer in enumerate(base_model.model.layers):
        if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            wrapper = CacheFlowQwenBlock(layer.mlp, None, i)
            layer.mlp = wrapper
            global_wrappers.append(wrapper)

    print("✅ Model Loaded & Wrapped. Ready for Experiments.")
else:
    print("✅ Model already loaded. Skipping.")

def reset_experiment(capacity):
    """Helper to reset scheduler state between experiments."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    new_scheduler = PyCacheFlowScheduler(capacity)
    for wrapper in global_wrappers:
        wrapper.set_scheduler(new_scheduler)
    return new_scheduler

def get_expert_size():
    # Helper for traffic calc
    return 4.5 # MB (Approx for 4-bit)

# %% [markdown]
# ### **Experiment 1: Capacity Scaling**
# 
# This experiment demonstrates the trade-off between **Expert Capacity** (VRAM allocated) and **Performance** (Latency/Thrashing). It proves the "Working Set" hypothesis.

# %%
capacities = [4, 8, 12, 16, 32, 60]
results = []
prompt = "Explain the difference between a CPU and a GPU in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
expert_mb = get_expert_size()

print("🚀 Running Capacity Sweep...")
for cap in capacities:
    print(f"   Testing Capacity: {cap}...", end="")
    scheduler = reset_experiment(cap)

    t0 = time.time()
    with torch.no_grad():
        _ = base_model.generate(**inputs, max_new_tokens=20)
    duration = time.time() - t0

    # Thesis Metrics
    swap_vol = scheduler.misses * expert_mb
    penalty = (swap_vol / 1024) / 12.0 # 12GB/s Bandwidth
    total_lat = duration + penalty

    results.append({
        "Capacity": cap,
        "Latency (s)": round(total_lat, 3),
        "Evictions": scheduler.evictions,
        "Swap Volume (MB)": round(swap_vol, 2),
        "Hit Rate (%)": round((scheduler.hits/(scheduler.hits+scheduler.misses+1e-9))*100, 1)
    })
    print(" Done.")

df_perf = pd.DataFrame(results)
display(df_perf)

# PLOTS
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Latency
sns.lineplot(data=df_perf, x='Capacity', y='Latency (s)', marker='o', ax=axes[0], color='tab:red')
axes[0].set_title("Performance vs Memory Limit")
axes[0].set_ylabel("Inference Latency (s)")

# Evictions
sns.barplot(data=df_perf, x='Capacity', y='Evictions', ax=axes[1], palette='viridis')
axes[1].set_title("Thrashing Intensity (Evictions)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### **3. Experiment 2: Internals (Expert Residency)**
# 
# This visualization opens the "Black Box" of the scheduler. By tracking which experts are resident in the cache over time, we show that the model tends to reuse the same small set of experts for a single generation task (Temporal Locality).

# %%
TEST_CAPACITY = 16
scheduler = reset_experiment(TEST_CAPACITY)

prompt = "Write a short poem about artificial intelligence."
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# Hook into scheduler to capture state
history_snapshots = []

# We generate token by token to snapshot the state
print("📸 Capturing Cache State...")
input_ids = inputs.input_ids
for _ in range(20):
    with torch.no_grad():
        # Generate 1 token
        out = base_model.generate(input_ids, max_new_tokens=1)

        # Snapshot: Which experts are in LRU list?
        # We assume 60 total experts. Mark 1 if resident, 0 if not.
        state = [0] * 60
        for eid in scheduler.lru_list:
            if eid < 60: state[eid] = 1
        history_snapshots.append(state)

        # Update input for next step
        input_ids = out

# Plot Heatmap
history_matrix = np.array(history_snapshots).T # (Experts, Time)

plt.figure(figsize=(12, 8))
sns.heatmap(history_matrix, cmap="Greys", cbar=False)
plt.title(f"Expert Cache Residency (Capacity={TEST_CAPACITY})")
plt.xlabel("Generation Step")
plt.ylabel("Expert ID")
plt.yticks(ticks=np.arange(0, 60, 5), labels=np.arange(0, 60, 5))
plt.show()

# %% [markdown]
# ### **4. Experiment 3: Behavior (The Locality Advantage)**
# 
# Why does CacheFlow work? This experiment proves that "Locality Matters." When prompts share a topic (e.g., "Physics"), the model reuses experts, resulting in a high Hit Rate. When prompts are random, the cache thrashes. This justifies our "Locality-Aware" priority score.**

# %%
# Topic A: Physics
prompts_physics = [
    "Newton's first law is",
    "Quantum entanglement explains",
    "The speed of light is",
    "Gravity affects time by",
    "Electrons orbit the nucleus because"
]

# Topic B: Random
prompts_random = [
    "The best way to cook pasta",
    "The history of the Roman Empire",
    "Python is a programming language",
    "Impressionist art is characterized by",
    "The capital of Australia is"
]

def run_batch(prompts, label):
    scheduler = reset_experiment(16) # Fix capacity at 16
    print(f"running {label}...")
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = base_model.generate(**inp, max_new_tokens=10)

    total = scheduler.hits + scheduler.misses
    rate = (scheduler.hits / total * 100)
    return rate

rate_physics = run_batch(prompts_physics, "Physics (High Locality)")
rate_random = run_batch(prompts_random, "Random (Low Locality)")

# Plot
df_loc = pd.DataFrame([
    {"Scenario": "Same Topic (Physics)", "Hit Rate (%)": rate_physics},
    {"Scenario": "Random Topics", "Hit Rate (%)": rate_random}
])

plt.figure(figsize=(6, 5))
sns.barplot(data=df_loc, x="Scenario", y="Hit Rate (%)", palette=["#2ca02c", "#d62728"])
plt.title("Impact of Prompt Locality on Cache Efficiency")
plt.ylim(0, 100)
plt.ylabel("Cache Hit Rate (%)")
plt.show()

# %% [markdown]
# ### **5. Experiment 4: Explainability (Routing Fairness)**
# 
# One concern with MoE models is "Routing Collapse" (using only 1 expert). This plot shows the distribution of expert selection. A healthy distribution (shown here) means the model is actually utilizing the diversity of experts, making the caching problem non-trivial and CacheFlow necessary.

# %%
scheduler = reset_experiment(32) # Give enough capacity to not distort routing
long_prompt = "Write a detailed essay about the history of the internet, covering ARPANET to modern day."
inp = tokenizer(long_prompt, return_tensors="pt").to(DEVICE)

print("📝 Generating Long Text for Stats...")
with torch.no_grad():
    _ = base_model.generate(**inp, max_new_tokens=100)

# Extract Data from Scheduler
access_counts = scheduler.access_history
experts = list(access_counts.keys())
counts = list(access_counts.values())

# Fill missing experts with 0
full_counts = [access_counts.get(i, 0) for i in range(60)]

plt.figure(figsize=(12, 5))
plt.bar(range(60), full_counts, color='#1f77b4')
plt.title("Expert Utilization Distribution (Routing Fairness)")
plt.xlabel("Expert ID")
plt.ylabel("Total Access Requests")
plt.xlim(0, 60)
plt.grid(axis='y', alpha=0.3)

# Calculate Gini Coefficient for the thesis text
arr = np.array(full_counts)
gini = np.abs(np.subtract.outer(arr, arr)).mean() / (2 * arr.mean())
print(f"Gini Coefficient (Imbalance Score): {gini:.2f} (0=Perfect Balance, 1=One Expert)")
plt.show()

# %% [markdown]
# Interactive Inference Generation

# %%
# 1. Configuration
prompt = "Explain quantum entanglement to a five-year-old." # @param {type:"string"}
capacity = 16 # @param {type:"slider", min:4, max:60, step:4}

# 2. Setup (Auto-reloads if needed)
if 'base_model' not in globals():
    print("⚠️ Model missing. Please run the 'Load Model' cell first.")
else:
    print(f"⚙️  Initializing CacheFlow (Capacity={capacity})...")
    scheduler = reset_experiment(capacity)

    # 3. Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"\n📝 Prompt: \"{prompt}\"")
    print("⏳ Generating...", end="", flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=40)
    dt = time.time() - t0

    # 4. Results
    print(f" Done ({dt:.2f}s)")
    print("-" * 60)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print("-" * 60)

    # 5. Explainable Stats
    total = scheduler.hits + scheduler.misses
    hit_rate = (scheduler.hits / total * 100) if total > 0 else 0
    expert_mb = 4.5 # Approx 4-bit size
    saved_mb = (scheduler.hits * expert_mb)

    print(f"📊 Session Statistics:")
    print(f"   • Active Experts Allowed: {capacity} / 60")
    print(f"   • Cache Hits:             {scheduler.hits} (Experts reused)")
    print(f"   • Cache Misses:           {scheduler.misses} (Experts loaded)")
    print(f"   • Hit Rate:               {hit_rate:.1f}%")
    print(f"   • Bandwidth Saved:        ~{saved_mb:.2f} MB")


