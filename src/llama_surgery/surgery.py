import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .layer import RotaryPositionEmbedding, apply_rotary_pos_emb
from .topology import DynamicTopologyRouter, get_dynamic_ultrametric_mask

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x):
        x_f32 = x.float()
        gamma = x_f32.abs().mean(dim=-1, keepdim=True)
        threshold = 0.5 * gamma
        
        x_ternary = torch.zeros_like(x_f32)
        x_ternary[x_f32 > threshold] = 1.0
        x_ternary[x_f32 < -threshold] = -1.0
        
        ctx.save_for_backward(x_f32, threshold)
        return x_ternary * gamma

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        return grad_output

def pack_ternary(x_ternary: torch.Tensor) -> torch.Tensor:
    """
    Packs a float tensor containing {-1.0, 0.0, 1.0} into int32.
    """
    assert x_ternary.shape[-1] % 16 == 0, "Last dimension must be multiple of 16 for packing"
    
    mapped = torch.zeros_like(x_ternary, dtype=torch.int32)
    mapped[x_ternary == -1.0] = 2
    mapped[x_ternary == 1.0] = 1
    
    shape = list(mapped.shape)
    shape[-1] = shape[-1] // 16
    shape.append(16)
    
    mapped = mapped.view(shape)
    packed = torch.zeros(shape[:-1], dtype=torch.int32, device=x_ternary.device)
    
    for i in range(16):
        packed |= (mapped[..., i] << (2 * i))
        
    return packed

def unpack_ternary(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Unpacks an int32 tensor back to float tensor containing {-1.0, 0.0, 1.0}.
    """
    shape = list(packed.shape)
    shape.append(16)
    
    unpacked = torch.zeros(shape, dtype=torch.int32, device=packed.device)
    for i in range(16):
        val = (packed >> (2 * i)) & 3
        unpacked[..., i] = val
        
    unpacked = unpacked.view(original_shape).float()
    
    res = torch.zeros_like(unpacked)
    res[unpacked == 2.0] = -1.0
    res[unpacked == 1.0] = 1.0
    
    return res

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equiv to torch.repeat_interleave(x, dim=1, repeats=n_rep)
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class SurgeryLossRamp(nn.Module):
    """
    Computes auxiliary loss for Llama Surgery to encourage sparsity.
    """
    def __init__(self, lambda_init: float = 0.0, lambda_max: float = 1.0, ramp_steps: int = 1000):
        super().__init__()
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.ramp_steps = ramp_steps

    def get_lambda(self, step: int) -> float:
        if step >= self.ramp_steps:
            return self.lambda_max
        return self.lambda_init + (self.lambda_max - self.lambda_init) * (step / self.ramp_steps)

    def forward(self, g_h: torch.Tensor, step: int) -> torch.Tensor:
        """
        g_h: Tensor of shape (num_heads,)
        Loss penalizes dense execution (g_h \approx 0).
        """
        lambda_t = self.get_lambda(step)
        # Average over heads
        sparsity_penalty = (1.0 - g_h).mean()
        return lambda_t * sparsity_penalty

class SurgicalLlamaAttention(nn.Module):
    """
    Llama Attention modified for Continuous Sparsification (Surgery).
    Integrates the per-head routing gate to interpolate between dense and sparse topology.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.embed_dim // self.num_heads)
        
        # GQA compliance
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Get surgery-specific config params, default to some values if not present
        self.p = getattr(config, "surgical_p", 2)
        self.alpha = getattr(config, "surgical_alpha", 10000.0)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=getattr(config, "attention_bias", False))

        max_pos_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_pos_embeddings)
        self.attn_dropout = nn.Dropout(getattr(config, "attention_dropout", 0.0))
        
        init_mode = getattr(config, "surgical_init_mode", "collapse")
        self.router = DynamicTopologyRouter(
            embed_dim=self.embed_dim,
            seq_len=max_pos_embeddings,
            num_heads=self.num_heads,
            p=self.p,
            init_mode=init_mode
        )

    # ------------------------------------------------------------------
    # Opt-in fast prefill backend (config.surgical_attention_backend = "block_list")
    # ------------------------------------------------------------------
    def _block_list_eligible(self, hidden_states, attention_mask, past_key_value) -> bool:
        """True if this call can use the block-list prefill path.

        Requires: backend opt-in, inference (no grad), a fresh prefill (no cached
        keys yet), CUDA fp16/bf16, no mask override, and no padding (batch 1 or no
        attention_mask), since the kernel applies only the causal mask.
        """
        if getattr(self.config, "surgical_attention_backend", "eager") != "block_list":
            return False
        if torch.is_grad_enabled() or hidden_states.shape[1] <= 1:
            return False
        if not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return False
        if getattr(self.config, "surgical_mask_override", None) is not None:
            return False
        if attention_mask is not None and hidden_states.shape[0] != 1:
            return False
        if past_key_value is not None:
            past_len = 0
            if hasattr(past_key_value, "get_seq_length"):
                try:
                    past_len = int(past_key_value.get_seq_length(self.layer_idx))
                except TypeError:
                    past_len = int(past_key_value.get_seq_length())
            elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                past_len = past_key_value[0].shape[-2]
            if past_len != 0:
                return False
        return True

    def _block_list_prefill(self, q, k, v, assignments, r):
        """Causal prefill with BLOCK-granular routing via the Triton block-list kernel.

        Each route block (config.surgical_route_block tokens, default 128) gets one
        routing vector: the per-level majority vote of its tokens' argmax branches
        (first r levels). Block i attends to block j <= i iff their vectors agree on
        all r levels, plus block 0 (sink) and ceil(surgical_local_window / RB) previous
        blocks (local window). This is coarser than the token-level eager mask, so
        quality must be measured separately; it is not an exact re-implementation.
        """
        from ultrametric.block_list import block_list_attention, build_block_lists

        RB = int(getattr(self.config, "surgical_route_block", 128))
        B, H, S, _ = q.shape
        NB = -(-S // RB)
        idx = assignments[:, :, :S, :r, :].argmax(dim=-1)                 # (B, H, S, r)
        votes = F.one_hot(idx, self.p).to(torch.float32)                  # (B, H, S, r, p)
        if NB * RB != S:
            votes = F.pad(votes, (0, 0, 0, 0, 0, NB * RB - S))
        block_route = votes.view(B, H, NB, RB, r, self.p).sum(3).argmax(-1).to(torch.int32)  # (B, H, NB, r)

        local_window = int(getattr(self.config, "surgical_local_window", 16))
        local_blocks = -(-local_window // RB) if local_window > 0 else 0
        bl = build_block_lists(block_route, r, is_causal=True, sink_block=True,
                               local_blocks=local_blocks, arity=self.p)

        if getattr(self.config, "surgical_collect_stats", False):
            # Exact allowed-key count under the block mask: for query block i with
            # c_i active blocks (all earlier ones full), token t of the block sees
            # (c_i - 1) * RB + t + 1 keys.
            lens = torch.full((NB,), RB, dtype=torch.float32, device=q.device)
            lens[-1] = S - (NB - 1) * RB
            c = bl.counts.to(torch.float32)
            allowed = ((c - 1) * RB * lens + lens * (lens + 1) / 2).sum(-1)  # (B, H)
            self.last_mean_allowed_keys = (allowed.mean() / S).item()
            self.last_mean_causal_keys = (S + 1) / 2
            self.last_block_density = (c.sum() / (B * H * NB * (NB + 1) / 2)).item()

        return block_list_attention(q, k, v, bl, route_block=RB, is_causal=True, sm_scale=self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        past_key_value = kwargs.get("past_key_value", past_key_values)

        use_block_list = self._block_list_eligible(hidden_states, attention_mask, past_key_value)
        req_depth_cfg = getattr(self.config, "surgical_req_depth", None)
        if use_block_list and req_depth_cfg == 0:
            # Dense prefill under the block_list backend = plain SDPA; the router is
            # not needed, so skip it (this is the "unmodified model" reference).
            self.current_penalty = hidden_states.new_zeros(())
            self._cached_assignments = None
            assignments = None
        else:
            tau = getattr(self.config, "surgical_tau", 1.0)
            curr_assignments, load_balance_loss = self.router(hidden_states, tau_override=tau)
            self.current_penalty = load_balance_loss

            if past_key_value is None or seq_len > 1:
                self._cached_assignments = curr_assignments
                assignments = curr_assignments
            else:
                if hasattr(self, '_cached_assignments') and self._cached_assignments is not None:
                    assignments = torch.cat([self._cached_assignments, curr_assignments], dim=2)
                    self._cached_assignments = assignments
                else:
                    assignments = curr_assignments
                    self._cached_assignments = assignments

        # Accumulate sparsity penalty during forward pass (REMOVED for gradient checkpointing safety)

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply KV cache fake quantization if enabled (QAT)
        if getattr(self.config, "quantize_kv_cache", False):
            from .qat import FakeQuantizeSTE
            kv_bits = getattr(self.config, "kv_cache_bits", 4)
            q_min = - (2 ** (kv_bits - 1))
            q_max = (2 ** (kv_bits - 1)) - 1
            # Channel-wise scaling along the head dimension (dim=-1)
            k_scale = k.abs().max(dim=-1, keepdim=True).values / q_max
            v_scale = v.abs().max(dim=-1, keepdim=True).values / q_max
            k = FakeQuantizeSTE.apply(k, kv_bits, k_scale, q_min, q_max)
            v = FakeQuantizeSTE.apply(v, kv_bits, v_scale, q_min, q_max)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # HF position_embeddings can be 3D or 4D depending on version
            if cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            def rotate_half(x):
                x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        else:
            cos, sin = self.rope(q)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {"sin": getattr(self, "_dummy", None), "cos": getattr(self, "_dummy", None)}
                k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            elif isinstance(past_key_value, tuple):
                # tuple format: (past_k, past_v)
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)
                past_key_value = (k, v)

        # Broadcast KV for GQA before attention computation
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # [Opt-in] Fast prefill backend: block-granular routing + Triton block-list
        # kernel (r > 0) or SDPA/FlashAttention (r = 0). Decode steps are unchanged.
        if use_block_list:
            if req_depth_cfg == 0:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                r = req_depth_cfg if req_depth_cfg is not None else assignments.shape[-2] // 2
                out = self._block_list_prefill(q, k, v, assignments, min(r, assignments.shape[-2]))
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.o_proj(out), None

        use_triton = getattr(self.config, "use_triton_sparse_attention", False)
        req_depth = getattr(self.config, "surgical_req_depth", 2)
        if use_triton and seq_len > 1 and q.dtype == torch.float16 and not q.requires_grad:
            from .kernel import routing_to_block_indices, ultrametric_attention_triton
            router_indices = routing_to_block_indices(assignments, seq_len=seq_len, block_size=128)
            out = ultrametric_attention_triton(q, k, v, router_indices, local_window=128, req_depth=req_depth, p=self.p)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            out = self.o_proj(out)
            return out, None

        # Raw attention scores
        # CRITICAL FIX: Cast to float32 BEFORE matmul to prevent float16 overflow (inf) which causes NaN gradients
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * self.scale

        # Base causal masking (simplified, since HF usually passes attention_mask)
        # HF passes attention_mask as (batch_size, 1, tgt_len, src_len) with -inf for masked
        if attention_mask is not None:
            # Handle possible broadcast shapes
            scores = scores + attention_mask
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Dynamic Sparsification
        L = k.shape[-2]
        local_window = getattr(self.config, "surgical_local_window", 16)
        levels = assignments.shape[-2]
        surgical_req_depth = getattr(self.config, "surgical_req_depth", None)

        # FAST INFERENCE PATH (Zero memory churn, O(1) integer prefix comparison)
        is_inference = (not self.training) or (not torch.is_grad_enabled())
        if is_inference:
            if surgical_req_depth == 0:
                # r = 0: Dense baseline (no sparse masking)
                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
                attn_weights = torch.nan_to_num(attn_weights, 0.0)
                attn_weights = attn_weights.to(v.dtype)
                attn_weights = self.attn_dropout(attn_weights)
                out = torch.matmul(attn_weights, v)
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                return self.o_proj(out), attn_weights

            # r > 0: Direct discrete prefix equality
            r = min(surgical_req_depth if surgical_req_depth is not None else (levels // 2), levels)
            indices = assignments.argmax(dim=-1)[..., :r]  # (B, H, S_total, r)
            powers = (self.p ** torch.arange(r, device=indices.device))
            branch_id = (indices * powers).sum(dim=-1)      # (B, H, S_total)

            if seq_len == 1 and L > 1:
                curr_branch = branch_id[:, :, -1:].unsqueeze(-1)    # (B, H, 1, 1)
                past_branches = branch_id[:, :, :L].unsqueeze(-2)  # (B, H, 1, L)
                um_mask_bool = (curr_branch == past_branches)      # (B, H, 1, L)
                if local_window > 0:
                    um_mask_bool[..., :, max(0, L - local_window):] = True
            else:
                curr_b = branch_id[:, :, :seq_len]                 # (B, H, seq_len)
                past_b = branch_id[:, :, :L]                       # (B, H, L)
                um_mask_bool = (curr_b.unsqueeze(-1) == past_b.unsqueeze(-2)) # (B, H, seq_len, L)
                if local_window > 0:
                    idx_q = torch.arange(seq_len, device=hidden_states.device)
                    idx_k = torch.arange(L, device=hidden_states.device)
                    band = torch.abs(idx_q.unsqueeze(1) - idx_k.unsqueeze(0)) <= local_window
                    um_mask_bool = um_mask_bool | band.unsqueeze(0).unsqueeze(0)

            # Attention Sink (Token 0)
            um_mask_bool[..., :, 0] = True

            # [Eval hook, opt-in] Matched-budget baseline: replace the routed mask
            # with a causal sliding window of W keys + sink (StreamingLLM-style).
            if getattr(self.config, "surgical_mask_override", None) == "window":
                W = int(getattr(self.config, "surgical_window_size", 256))
                idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                idx_k = torch.arange(L, device=hidden_states.device)
                band = (idx_k.unsqueeze(0) > idx_q.unsqueeze(1) - W)  # (seq_len, L)
                um_mask_bool = band.unsqueeze(0).unsqueeze(0).expand_as(um_mask_bool).clone()
                um_mask_bool[..., :, 0] = True

            # [Eval hook, opt-in] Record the mean number of causal keys each query
            # may attend to (prefill only), so budgets are measured, not assumed.
            if getattr(self.config, "surgical_collect_stats", False) and seq_len > 1:
                idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                idx_k = torch.arange(L, device=hidden_states.device)
                causal = idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)
                allowed = (um_mask_bool & causal).sum(-1).float()   # (B, H, seq_len)
                self.last_mean_allowed_keys = allowed.mean().item()
                self.last_mean_causal_keys = causal.sum(-1).float().mean().item()

            sparse_scores = scores.masked_fill(~um_mask_bool, float('-inf'))
            is_all_neg_inf = (sparse_scores == float('-inf')).all(dim=-1, keepdim=True)
            sparse_scores = sparse_scores.masked_fill(is_all_neg_inf, 0.0)

            attn_weights = F.softmax(sparse_scores, dim=-1, dtype=torch.float32)
            attn_weights = torch.nan_to_num(attn_weights, 0.0)
            attn_weights = attn_weights.to(v.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            return self.o_proj(out), attn_weights

        # TRAINING PATH (Differentiable Straight-Through Estimator)
        max_dist = getattr(
            self.config,
            "surgical_max_dist",
            (levels - surgical_req_depth) if surgical_req_depth is not None else None
        )
        full_mask = get_dynamic_ultrametric_mask(
            assignments, p=self.p, max_dist=max_dist, local_window=local_window
        ).to(hidden_states.device)
        um_mask_bool = full_mask > 0.5  # Shape: (B, H, S_full, L) or (B, H, L, L)
        
        if seq_len == 1 and L > 1:
            # Decode phase: extract the row for the current absolute token index
            um_mask_bool = um_mask_bool[:, :, -1:, :]
            full_mask = full_mask[:, :, -1:, :]
            
        sparse_scores = scores.masked_fill(~um_mask_bool, float('-inf'))
        
        # CRITICAL FIX: Prevent softmax NaN from all -inf rows (e.g. padded tokens getting fully masked)
        is_all_neg_inf = (sparse_scores == float('-inf')).all(dim=-1, keepdim=True)
        sparse_scores = sparse_scores.masked_fill(is_all_neg_inf, 0.0)
        
        attn_weights = F.softmax(sparse_scores, dim=-1, dtype=torch.float32)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Multiply by the differentiable soft mask for training
        attn_weights = attn_weights * full_mask
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        attn_weights = attn_weights.to(v.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        out = self.o_proj(out)
        
        return out, attn_weights

