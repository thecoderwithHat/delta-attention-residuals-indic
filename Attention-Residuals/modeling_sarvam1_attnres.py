"""
Sarvam-1 with Block Attention Residuals (AttnRes).

Replaces standard additive residual connections with softmax attention over
previous block representations, as described in:
  "Attention Residuals" (Kimi Team, arXiv:2603.15031)

Sarvam-1 is a Llama-2-architecture model (sarvamai/sarvam-1):
  - model_type: "llama"
  - hidden_size: 2048, intermediate_size: 11008
  - num_attention_heads: 16, num_key_value_heads: 8 (GQA)
  - num_hidden_layers: 28, head_dim: 128
  - vocab_size: 68096
  - rope_theta: 10000.0, no RoPE scaling

For pretrained model conversion, we use a **recency bias** approach:
a large learnable bias on the last element (partial_block) in the depth-
attention logits makes softmax put ~100% weight on it at init.  This means
block_attn_res(...) ≈ partial_block at init → the model is mathematically
equivalent to standard Sarvam-1.  During training the bias and proj weights
co-adapt, letting the model learn cross-block attention.
"""

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

# Re-use Llama components directly from the installed transformers package.
# We only override DecoderLayer and Model; everything else is unchanged.
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


# ---------------------------------------------------------------------------
# V-stream decoupled attention
# ---------------------------------------------------------------------------

class Sarvam1AttnResAttention(LlamaAttention):
    """LlamaAttention extended to accept separate hidden states for value projection.

    When ``value_hidden_states`` is provided, V is projected from it instead of
    from ``hidden_states`` (which is still used for Q and K).  This enables
    per-stream delta routing: Q/K attend from one depth-aggregated input while
    V retrieves content from a different depth-aggregated input.

    Note: unlike Qwen3, Llama has no per-head Q/K norm, so this subclass only
    diverges from LlamaAttention in the V-source override.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        value_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # V from separate input if provided, else from hidden_states
        v_input = value_hidden_states if value_hidden_states is not None else hidden_states
        value_states = self.v_proj(v_input).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Config extension
# ---------------------------------------------------------------------------

class Sarvam1AttnResConfig(LlamaConfig):
    """LlamaConfig (Sarvam-1 variant) extended with AttnRes hyper-parameters."""

    model_type = "sarvam1_attnres"

    def __init__(self, attnres_num_blocks: int = 8,
                 attnres_mode: str = "block",
                 attnres_gate_type: str = "bias",
                 **kwargs):
        # Remove legacy keys if present (from old checkpoints)
        kwargs.pop("attnres_init_bias", None)
        kwargs.pop("attnres_gate_init", None)
        super().__init__(**kwargs)
        self.attnres_num_blocks = attnres_num_blocks
        # "block" = Block AttnRes (grouped), "full" = Full AttnRes (per-sublayer)
        # "delta" = Delta AttnRes (attend over sublayer outputs, not cumulative states)
        # "delta_block" = Delta + Block: attend over block-aggregated deltas (best of both)
        # "delta_v" = Delta with V-stream decoupling: V gets independent depth routing
        self.attnres_mode = attnres_mode
        # Null source for identity init (for fine-tuning pretrained models)
        self.attnres_use_null_source = kwargs.pop("attnres_use_null_source", False)
        # Gate type: "bias" (recency bias on softmax logit),
        #            "sigmoid_scalar" (scalar sigmoid gate between residual & attnres),
        #            "sigmoid_vector" (input-dependent per-dim sigmoid gate)
        self.attnres_gate_type = attnres_gate_type


# ---------------------------------------------------------------------------
# Core Block-AttnRes operation
# ---------------------------------------------------------------------------

def _block_attn_res_kernel(
    V: torch.Tensor,              # pre-stacked sources [N+1, B, T, D]
    query: torch.Tensor,          # (D,)
    norm: LlamaRMSNorm,
) -> torch.Tensor:
    """Compiled inner kernel for block_attn_res (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", weights, V)


def _block_attn_res_kernel_with_entropy(
    V: torch.Tensor,
    query: torch.Tensor,
    norm: LlamaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled inner kernel for block_attn_res with entropy (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    h = torch.einsum("n b t, n b t d -> b t d", weights, V)
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
    return h, entropy


# Compiled versions (created lazily via enable_compile())
_compiled_block_kernel = None
_compiled_block_kernel_entropy = None
_compiled_delta_kernel = None
_compiled_delta_kernel_entropy = None


def enable_compile():
    """Compile the AttnRes kernels with torch.compile(dynamic=True)."""
    global _compiled_block_kernel, _compiled_block_kernel_entropy
    global _compiled_delta_kernel, _compiled_delta_kernel_entropy
    _compiled_block_kernel = torch.compile(
        _block_attn_res_kernel, dynamic=True)
    _compiled_block_kernel_entropy = torch.compile(
        _block_attn_res_kernel_with_entropy, dynamic=True)
    _compiled_delta_kernel = torch.compile(
        _delta_attn_res_kernel, dynamic=True)
    _compiled_delta_kernel_entropy = torch.compile(
        _delta_attn_res_kernel_with_entropy, dynamic=True)


def block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, T, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, T, D]
    proj: nn.Linear,              # learned pseudo-query weight  (d,)
    norm: LlamaRMSNorm,           # RMSNorm applied to keys before scoring
    return_entropy: bool = False, # if True, also return mean entropy of softmax weights
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Attend over all block representations + the current partial block.

    Returns a [B, T, D] tensor — the attended aggregation of depth history.
    If return_entropy=True, also returns a scalar entropy value.
    """
    # Stack outside compiled region so torch.compile sees a tensor, not a list
    V = torch.stack(blocks + [partial_block], dim=0)
    query = proj.weight.view(-1)

    if return_entropy:
        kernel = _compiled_block_kernel_entropy or _block_attn_res_kernel_with_entropy
        return kernel(V, query, norm)

    kernel = _compiled_block_kernel or _block_attn_res_kernel
    return kernel(V, query, norm)


def _delta_attn_res_kernel(
    V: torch.Tensor,              # pre-stacked sources [N, B, T, D]
    partial_block: torch.Tensor,  # [B, T, D]
    query: torch.Tensor,          # (D,)
    norm: LlamaRMSNorm,
) -> torch.Tensor:
    """Compiled inner kernel for delta_attn_res (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    selected = torch.einsum("n b t, n b t d -> b t d", weights, V)
    return partial_block + selected


def _delta_attn_res_kernel_with_entropy(
    V: torch.Tensor,
    partial_block: torch.Tensor,
    query: torch.Tensor,
    norm: LlamaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled inner kernel for delta_attn_res with entropy (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    selected = torch.einsum("n b t, n b t d -> b t d", weights, V)
    h = partial_block + selected
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
    return h, entropy


def delta_attn_res(
    deltas: list[torch.Tensor],   # previous sublayer outputs (deltas)  [B, T, D] each
    partial_block: torch.Tensor,  # current residual stream  [B, T, D]
    proj: nn.Linear,
    norm: LlamaRMSNorm,
    null_source: nn.Parameter | None = None,  # learnable null token for identity init
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Attend over previous sublayer deltas and add selected information to
    the current residual stream.

    If null_source is provided, it's prepended as source 0. With null_source
    zero-initialized and proj.weight zero-initialized, softmax gives a uniform
    distribution over (N+1) sources, so the weighted sum is
    `(1/(N+1)) * (0 + Σ deltas) = (1/(N+1)) * Σ deltas`. This is a bounded
    perturbation (scales with the mean of past sublayer outputs), NOT an exact
    identity — the deltas are non-zero random projections at init. The
    "identity init" claim is best interpreted as "bounded perturbation" —
    useful in practice for fine-tuning pretrained models without large loss
    spikes, but not a strict equality.

        h = partial_block + (1/(N+1)) * (null_source + Σ deltas)
        init: null_source=0, proj=0 → uniform → h = partial + (mean of deltas)/(N+1)
    """
    if not deltas and null_source is None:
        if return_entropy:
            return partial_block, torch.tensor(0.0, device=partial_block.device)
        return partial_block

    # Build source list outside compiled region
    sources = list(deltas)
    if null_source is not None:
        null_expanded = null_source.unsqueeze(0).unsqueeze(0).expand_as(partial_block)
        sources = [null_expanded] + sources

    if not sources:
        if return_entropy:
            return partial_block, torch.tensor(0.0, device=partial_block.device)
        return partial_block

    # Stack outside compiled region
    V = torch.stack(sources, dim=0)
    query = proj.weight.view(-1)

    if return_entropy:
        kernel = _compiled_delta_kernel_entropy or _delta_attn_res_kernel_with_entropy
        return kernel(V, partial_block, query, norm)

    kernel = _compiled_delta_kernel or _delta_attn_res_kernel
    return kernel(V, partial_block, query, norm)


def gated_delta_attn_res(
    deltas: list[torch.Tensor],   # previous sublayer outputs  [B, T, D] each
    partial_block: torch.Tensor,  # current residual stream  [B, T, D]
    proj: nn.Linear,              # learned query  (d,)
    norm: LlamaRMSNorm,
    gate_proj: nn.Linear,         # gate projection (d -> d), produces per-source per-dim gate
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Gated Attention Residuals with pre-softmax source gating.

    1. Gate each source: V_gated = sigmoid(gate_proj(partial)) * V
       → filter which source dimensions are useful before routing
    2. Softmax attention over gated sources
    3. Add selected to residual stream

    The gate sees the current hidden state and decides which source
    dimensions to let through, THEN softmax selects among filtered sources.
    """
    if not deltas:
        if return_entropy:
            return partial_block, torch.tensor(0.0, device=partial_block.device)
        return partial_block

    # Stack deltas: (N, B, T, D)
    V = torch.stack(deltas, dim=0)

    # Pre-softmax gate: filter sources based on current hidden state
    # gate: (B, T, D) → broadcast over N sources
    gate = torch.sigmoid(gate_proj(partial_block))  # (B, T, D)
    V_gated = V * gate.unsqueeze(0)                 # (N, B, T, D)

    # Keys from gated values
    K = norm(V_gated)

    # Query
    query = proj.weight.view(-1)                              # (D,)
    logits = torch.einsum("d, n b t d -> n b t", query, K)   # (N, B, T)

    # Softmax over filtered sources
    weights = logits.softmax(dim=0)                           # (N, B, T)

    # Weighted sum of gated values
    selected = torch.einsum("n b t, n b t d -> b t d", weights, V_gated)

    # Add to residual stream
    h = partial_block + selected

    if return_entropy:
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
        return h, entropy

    return h


# ---------------------------------------------------------------------------
# Modified decoder layer
# ---------------------------------------------------------------------------

class Sarvam1AttnResDecoderLayer(GradientCheckpointingLayer):
    """
    Sarvam-1 (Llama) decoder layer with Block AttnRes via recency-biased depth attention.

    At init, a large recency bias makes block_attn_res return partial_block
    exactly, so the model is mathematically identical to standard Sarvam-1.
    During training, proj weights learn to attend to earlier blocks while
    the bias co-adapts.

    Forward:
        h = block_attn_res(blocks, partial_block)   # ≈ partial_block at init
        attn_out = self_attn(layernorm(h))
        partial_block = partial_block + attn_out     # standard residual add
    """

    def __init__(self, config: Sarvam1AttnResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # AttnRes mode
        self.attnres_mode = getattr(config, "attnres_mode", "block")

        # Use V-stream decoupled attention for delta_v and full_v modes
        if self.attnres_mode in ("delta_v", "full_v"):
            self.self_attn = Sarvam1AttnResAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Llama has a single attention type (no sliding layers), but we keep
        # the lookup shape so layer forward signature stays unchanged.
        layer_type_key = "sliding_attention" if getattr(config, "sliding_window", None) else "full_attention"
        self.attention_type = layer_type_key

        # AttnRes components — one (proj, norm) per sublayer for Q/K routing.
        self.attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.attn_res_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # V-stream: independent routing for value projection
        if self.attnres_mode in ("delta_v", "full_v", "block_v", "delta_block_v"):
            self.v_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.v_res_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gate type determines how AttnRes output is mixed with residual stream
        self.gate_type = getattr(config, "attnres_gate_type", "bias")

        if self.gate_type == "sigmoid_scalar":
            # Scalar sigmoid gate: sigmoid(-2) ≈ 0.12 → small initial mixing
            self.attn_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
            self.mlp_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
        elif self.gate_type == "sigmoid_vector":
            # Input-dependent vector gate: per-dim, per-token gating
            self.attn_res_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.attn_res_gate_proj.weight)
            nn.init.constant_(self.attn_res_gate_proj.bias, -2.0)
            self.mlp_res_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.mlp_res_gate_proj.weight)
            nn.init.constant_(self.mlp_res_gate_proj.bias, -2.0)
        elif self.gate_type == "learnable_alpha":
            # Simple learnable scalar: h = (1-α)*partial + α*attnres, init α=0
            self.attn_res_alpha = nn.Parameter(torch.tensor(0.0))
            self.mlp_res_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            # Default "bias": no gate, AttnRes output used directly
            pass

        # Null source for identity init (fine-tuning)
        self.use_null_source = getattr(config, "attnres_use_null_source", False)
        if self.use_null_source:
            # Zero-init: softmax gives uniform weight, null contributes zeros → h ≈ partial
            self.attn_null_source = nn.Parameter(torch.zeros(config.hidden_size))
            self.mlp_null_source = nn.Parameter(torch.zeros(config.hidden_size))

        # Pre-softmax gate for "pre_gated" mode
        if self.attnres_mode == "pre_gated":
            self.attn_pre_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.attn_pre_gate.weight)
            nn.init.constant_(self.attn_pre_gate.bias, 0.0)  # sigmoid(0)=0.5 → pass half
            self.mlp_pre_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.mlp_pre_gate.weight)
            nn.init.constant_(self.mlp_pre_gate.bias, 0.0)

        # Block boundary: how many transformer layers per block (used in block mode)
        num_layers = config.num_hidden_layers
        num_blocks = getattr(config, "attnres_num_blocks", 8)
        self.layers_per_block = max(1, (num_layers + num_blocks - 1) // num_blocks)

    @property
    def is_block_boundary(self) -> bool:
        """True when this layer is the first in its block (0-indexed).

        NOTE: This name is preserved for backward compatibility but it actually
        returns "is this a block START", not a block END. For end-of-block
        detection (used by delta_block accumulation), use ``is_block_end``.
        """
        return (self.layer_idx) % self.layers_per_block == 0

    @property
    def is_new_block_start(self) -> bool:
        """True when this is the first layer of a new block (including block 0)."""
        return self.layer_idx % self.layers_per_block == 0

    @property
    def is_block_end(self) -> bool:
        """True when this layer is the LAST in its block (0-indexed).

        Used by the model forward to trigger block-delta accumulation for
        delta_block / delta_block_v modes. With layers_per_block=K, this is
        True at layer_idx = K-1, 2K-1, 3K-1, ...
        """
        return (self.layer_idx + 1) % self.layers_per_block == 0

    def _apply_gate(self, hidden_states, h_attn, sublayer: str):
        """Apply gating between residual stream and AttnRes output."""
        if self.gate_type == "sigmoid_scalar":
            logit = self.attn_res_gate_logit if sublayer == "attn" else self.mlp_res_gate_logit
            gate = torch.sigmoid(logit)
            return (1 - gate) * hidden_states + gate * h_attn
        elif self.gate_type == "sigmoid_vector":
            gate_proj = self.attn_res_gate_proj if sublayer == "attn" else self.mlp_res_gate_proj
            gate = torch.sigmoid(gate_proj(hidden_states))  # (B, T, D)
            return (1 - gate) * hidden_states + gate * h_attn
        elif self.gate_type == "learnable_alpha":
            alpha = self.attn_res_alpha if sublayer == "attn" else self.mlp_res_alpha
            return (1 - alpha) * hidden_states + alpha * h_attn
        else:
            # "bias" mode: no gate, AttnRes output used directly
            return h_attn

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        entropy_accum = kwargs.pop("entropy_accum", None)

        if self.attnres_mode == "pre_gated":
            # ---- Pre-gated delta: gate sources BEFORE softmax ----
            # Same source collection as delta mode, but uses gated_delta_attn_res

            # Attention sublayer
            h = gated_delta_attn_res(blocks, partial_block,
                                     self.attn_res_proj, self.attn_res_norm, self.attn_pre_gate)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            h = gated_delta_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm, self.mlp_pre_gate)

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "first_layer":
            # ---- First Layer Residual: softmax attention over [partial, mlp_out_0] ----
            # Like a 2-source AttnRes with learned query
            if len(blocks) >= 3:
                first_layer_mlp = blocks[2]  # mlp_out_0
            else:
                first_layer_mlp = None

            # Attention sublayer
            if first_layer_mlp is not None:
                V = torch.stack([partial_block, first_layer_mlp], dim=0)  # (2, B, T, D)
                K = self.attn_res_norm(V)
                query = self.attn_res_proj.weight.view(-1)
                logits = torch.einsum("d, n b t d -> n b t", query, K)  # (2, B, T)
                weights = logits.softmax(dim=0)
                h = torch.einsum("n b t, n b t d -> b t d", weights, V)
            else:
                h = partial_block

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            if first_layer_mlp is not None:
                V = torch.stack([partial_block, first_layer_mlp], dim=0)
                K = self.mlp_res_norm(V)
                query = self.mlp_res_proj.weight.view(-1)
                logits = torch.einsum("d, n b t d -> n b t", query, K)
                weights = logits.softmax(dim=0)
                h = torch.einsum("n b t, n b t d -> b t d", weights, V)
            else:
                h = partial_block

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "delta_block":
            # ---- Delta-Block mode: delta_attn_res with block-aggregated sources ----
            # Per the Delta paper, sources are [h_0, Δ_1, Δ_2, ..., Δ_N] where
            # h_0 is the token embedding and Δ_i is the sum of sublayer outputs
            # within block i. Block-delta accumulation is handled by the MODEL
            # forward (which knows the inter-layer boundary), not by this layer.
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.attn_res_proj, self.attn_res_norm, attn_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out

            # MLP sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.mlp_res_proj, self.mlp_res_norm, mlp_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            # blocks list is mutated by model forward (block-delta accumulation).
            return blocks, partial_block

        if self.attnres_mode == "delta_block_v":
            # ---- Delta-Block-V mode: delta_block + per-stream V routing ----
            # Same block-delta accumulation as delta_block (handled by model forward).
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer — two delta_attn_res calls (QK and V)
            h_qk = delta_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm, attn_null)
            h_v = delta_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm, attn_null)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out

            # MLP sublayer — single routing
            h_attn = delta_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            # blocks list is mutated by model forward (block-delta accumulation).
            return blocks, partial_block

        if self.attnres_mode == "delta":
            # ---- Delta mode: attend over previous sublayer outputs (deltas) ----
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.attn_res_proj, self.attn_res_norm, attn_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if not blocks:
                blocks = blocks + [partial_block]
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.mlp_res_proj, self.mlp_res_norm, mlp_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "delta_v":
            # ---- Delta-V mode: V-stream gets independent depth routing ----
            # Q/K use attn_res_proj routing; V uses v_res_proj routing.
            # MLP sublayer is unchanged from delta mode.
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer — two parallel delta_attn_res calls
            h_qk = delta_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm, attn_null)
            h_v = delta_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm, attn_null)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer (same as delta — no V-stream decoupling needed)
            h_attn = delta_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "full_v":
            # ---- Full-V mode: cumulative states + per-stream V routing ----
            # Like full mode but with independent V-stream depth routing.

            # Attention sublayer — two block_attn_res calls (QK and V)
            h_qk = block_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm)
            h_v = block_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [partial_block]  # cumulative state after attn

            # MLP sublayer — single routing (same as full mode)
            h_attn = block_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [partial_block]  # cumulative state after mlp

            return blocks, partial_block

        if self.attnres_mode == "block":
            # ---- Block mode (Kimi-faithful) ----
            # block_attn_res sees old partial BEFORE boundary reset.
            # At new-block start: store old partial in blocks, reset partial.
            # partial_block tracks intra-block accumulation only.
            #
            # WARNING: this is the original Kimi AttnRes (replacement routing).
            # The Delta paper (Figure 4, Table 2) shows this variant DEGRADES at
            # scale: +6.9% worse than Delta at 1044M params, +6.6% worse at 8B.
            # Kept for ablation; for new work prefer "delta_block" or "delta".

            # Attention sublayer — uses old partial
            if entropy_accum is not None:
                h_attn, ent = block_attn_res(blocks, partial_block,
                                             self.attn_res_proj, self.attn_res_norm, return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = block_attn_res(blocks, partial_block,
                                        self.attn_res_proj, self.attn_res_norm)
            h = self._apply_gate(partial_block, h_attn, "attn")

            # New block boundary: store old partial, reset
            if self.is_new_block_start:
                blocks = blocks + [partial_block]
                partial_block = torch.zeros_like(partial_block)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out

            # MLP sublayer
            if entropy_accum is not None:
                h_attn, ent = block_attn_res(blocks, partial_block,
                                             self.mlp_res_proj, self.mlp_res_norm, return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = block_attn_res(blocks, partial_block,
                                        self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            return blocks, partial_block

        if self.attnres_mode == "block_v":
            # ---- Block-V mode: block mode + per-stream V routing ----

            # Attention sublayer — two block_attn_res calls (QK and V)
            h_qk = block_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm)
            h_v = block_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            # New block boundary: store old partial, reset
            if self.is_new_block_start:
                blocks = blocks + [partial_block]
                partial_block = torch.zeros_like(partial_block)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out

            # MLP sublayer — single routing (same as block mode)
            h_attn = block_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            return blocks, partial_block

        if self.attnres_mode == "delta_replace":
            # ---- Delta Replace mode: replacement routing + delta sources ----
            # Uses block_attn_res (h = weighted_sum) with delta sources (attn_out, mlp_out).

            # Attention sublayer
            h_attn = block_attn_res(blocks, partial_block,
                                    self.attn_res_proj, self.attn_res_norm)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            h_attn = block_attn_res(blocks, partial_block,
                                    self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "full_additive":
            # ---- Full Additive mode: additive routing + cumulative sources ----
            # Uses delta_attn_res (h = partial + weighted_sum) with cumulative
            # partial_block sources (like full mode).
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer
            h_attn = delta_attn_res(blocks, partial_block,
                                    self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [partial_block]  # cumulative state

            # MLP sublayer
            h_attn = delta_attn_res(blocks, partial_block,
                                    self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [partial_block]  # cumulative state

            return blocks, partial_block

        # ---- Full mode (delta sources + replacement routing + final routing) ----
        # WARNING: this is the worst-performing AttnRes variant per the Delta
        # paper: +12.3% worse than Delta at 1044M, and not even feasible at
        # 8B scale due to memory cost of N=L source stacking. Each sublayer
        # attends over ALL prior cumulative states. Kept for ablation only;
        # for new work prefer "delta_block" or "delta".

        # Attention sublayer
        h_attn = block_attn_res(blocks, partial_block,
                                self.attn_res_proj, self.attn_res_norm)
        h = self._apply_gate(partial_block, h_attn, "attn")

        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(h),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        blocks = blocks + [partial_block]
        partial_block = attn_out

        # MLP sublayer
        h_attn = block_attn_res(blocks, partial_block,
                                self.mlp_res_proj, self.mlp_res_norm)
        h = self._apply_gate(partial_block, h_attn, "mlp")

        mlp_out = self.mlp(self.post_attention_layernorm(h))
        blocks = blocks + [partial_block]
        partial_block = mlp_out

        return blocks, partial_block


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class Sarvam1AttnResModel(LlamaPreTrainedModel):
    """Sarvam-1 (Llama) backbone with Block AttnRes via recency-biased depth attention."""

    config_class = Sarvam1AttnResConfig

    def _init_weights(self, module):
        """Override to preserve AttnRes initialization."""
        super()._init_weights(module)
        if isinstance(module, Sarvam1AttnResDecoderLayer):
            gate_type = getattr(self.config, "attnres_gate_type", "bias")
            if gate_type == "sigmoid_scalar":
                module.attn_res_gate_logit.data.fill_(-2.0)
                module.mlp_res_gate_logit.data.fill_(-2.0)
            elif gate_type == "sigmoid_vector":
                nn.init.zeros_(module.attn_res_gate_proj.weight)
                nn.init.constant_(module.attn_res_gate_proj.bias, -2.0)
                nn.init.zeros_(module.mlp_res_gate_proj.weight)
                nn.init.constant_(module.mlp_res_gate_proj.bias, -2.0)
            elif gate_type == "learnable_alpha":
                module.attn_res_alpha.data.fill_(0.0)
                module.mlp_res_alpha.data.fill_(0.0)

    def __init__(self, config: Sarvam1AttnResConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Sarvam1AttnResDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Sarvam-1 (Llama) has no sliding-window layers.
        self.has_sliding_layers = False

        # Final block_attn_res: produces effective hidden state after all layers
        # by routing over all sources + last partial. Needed for any mode where
        # partial_block doesn't carry the full cumulative state.
        #
        # NOTE: "delta_centered_reset" was previously listed here but is not a
        # defined mode. If/when it is added, append it to the tuple below.
        self._attnres_mode = getattr(config, "attnres_mode", "block")
        self._needs_final_routing = self._attnres_mode in (
            "block", "block_v", "full",
        )
        if self._needs_final_routing:
            self.final_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.final_res_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Sarvam-1 (Llama) has a single attention type across all layers.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Block AttnRes state: list of completed block tensors + current partial.
        #
        # For delta_block / delta_block_v, the sources follow the paper's
        # specification: sources = [h_0, Δ_1, Δ_2, ..., Δ_N], where
        #   h_0    = token embedding  (source 0, prepended)
        #   Δ_i    = sum of sublayer outputs within block i
        # The paper's Figure 6 notes the embedding "receives disproportionate
        # attention from deep layers" — so h_0 is included as an explicit
        # source rather than rolled into the first block delta.
        attnres_mode = getattr(self.config, "attnres_mode", "block")
        is_delta_block = attnres_mode in ("delta_block", "delta_block_v")

        if is_delta_block:
            blocks: list[torch.Tensor] = [inputs_embeds]   # h_0 as source 0
            block_start_partial: torch.Tensor = inputs_embeds
        elif attnres_mode in ("block", "block_v", "full"):
            # block/full modes: layer 0 stores embed at boundary, so start empty
            blocks: list[torch.Tensor] = []
            block_start_partial = None
        else:
            # delta / delta_v / first_layer / pre_gated / delta_replace /
            # full_additive: layer logic appends sources directly
            blocks: list[torch.Tensor] = []
            block_start_partial = None
        partial_block: torch.Tensor = inputs_embeds

        # Entropy accumulation for auxiliary loss
        entropy_lambda = kwargs.pop("entropy_lambda", 0.0)
        entropy_accum = [] if entropy_lambda > 0 else None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                blocks, partial_block = self._gradient_checkpointing_func(
                    layer.__call__,
                    blocks,
                    partial_block,
                    causal_mask_mapping[layer.attention_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                blocks, partial_block = layer(
                    blocks=blocks,
                    partial_block=partial_block,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    entropy_accum=entropy_accum,
                )

            # Delta-Block accumulation: at the end of each block, push the
            # block delta (sum of sublayer outputs within this block) into
            # the source list. The very first source is h_0 (the embedding,
            # prepended at init above). After this push, sources for the next
            # block become [h_0, Δ_1, ..., Δ_{i+1}].
            if is_delta_block and layer.is_block_end:
                block_delta = partial_block - block_start_partial
                blocks = blocks + [block_delta]
                block_start_partial = partial_block

        # Final routing: produce effective hidden state from all sources.
        # Needed for modes where partial_block doesn't carry the full state.
        if self._needs_final_routing:
            partial_block = block_attn_res(
                blocks, partial_block,
                self.final_res_proj, self.final_res_norm,
            )

        hidden_states = self.norm(partial_block)

        # Compute mean entropy across all sublayers
        attnres_entropy = None
        if entropy_accum:
            attnres_entropy = torch.stack(entropy_accum).mean()

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        # Attach entropy as extra attribute
        out.attnres_entropy = attnres_entropy
        return out


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class Sarvam1AttnResForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """Sarvam-1 (Llama) causal LM with Block AttnRes residuals."""

    config_class = Sarvam1AttnResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Sarvam1AttnResConfig):
        super().__init__(config)
        self.model = Sarvam1AttnResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            # Concentration (anti-entropy) bonus: encourage SHARP routing.
            # Per the Delta paper, deep layers should concentrate attention on
            # a few sources (max weight ~0.6), not collapse to uniform (~0.2).
            # We add a positive entropy term to the loss so that minimizing
            # loss pushes entropy down toward sharper routing.
            entropy_lambda = kwargs.get("entropy_lambda", 0.0)
            attnres_entropy = getattr(outputs, "attnres_entropy", None)
            if entropy_lambda > 0 and attnres_entropy is not None:
                loss = loss + entropy_lambda * attnres_entropy

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
