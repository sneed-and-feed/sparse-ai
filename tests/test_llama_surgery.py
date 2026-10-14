"""
Comprehensive Unit Tests for Milestone 2 (Track 1: QK-Mode, Multi-Prime, Swin Shift,
Route Stride, Block-Level KL + Dual Lagrangian, Frontier Model Patching) and
Milestone 3 (Track 3: SurprisalAdelicCache, Non-Power-of-Two ultrametric_decode,
and Self-Speculative Ultrametric Decoding).
"""

import os
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_surgery import (
    DynamicTopologyRouter,
    MultiPrimeTopologyRouter,
    compute_expected_block_distance,
    SurgicalLlamaAttention,
    patch_llama_model,
    inject_surgery,
    SurprisalAdelicCache,
    TemporalAdelicCache,
    SpeculativeSurgicalGenerator,
)
from ultrametric.kernel_decode import ultrametric_decode, _ultrametric_decode_reference


# ============================================================================
# 1. DynamicTopologyRouter: QK-Mode, Multi-Prime, Expected Block Distance
# ============================================================================


def test_router_qk_mode_and_collapse_init():
    B, H, NB, D = 2, 4, 6, 16
    router = DynamicTopologyRouter(
        embed_dim=H * D,
        seq_len=512,
        num_heads=H,
        p=2,
        init_mode="collapse",
        levels=4,
        qk_mode=True,
        head_dim=D,
    )
    router.eval()
    qk_desc = torch.randn(B, H, NB, 2 * D)
    assign, lb_loss = router(qk_desc)
    assert assign.shape == (B, H, NB, 4, 2)
    # Collapse init guarantees all blocks choose branch 0 at step 0
    assert torch.all(assign[..., 0] == 1.0)
    assert torch.all(assign[..., 1] == 0.0)

    # Distance between all blocks at step 0 must be 0
    dist = compute_expected_block_distance(assign)
    assert dist.shape == (B, H, NB, NB)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-6)


def test_multi_prime_topology_router_and_ultrametric_distance():
    B, H, NB, D = 2, 4, 8, 16
    router = MultiPrimeTopologyRouter(
        embed_dim=H * D,
        seq_len=512,
        num_heads=H,
        prime_arities=[2, 3],
        init_mode="random",
        levels=4,
        qk_mode=True,
        head_dim=D,
    )
    assert router.head_arities == [2, 2, 3, 3]
    assert router.p == 3

    router.train()
    qk_desc = torch.randn(B, H, NB, 2 * D, requires_grad=True)
    assign, lb_loss = router(qk_desc, tau_override=0.8)
    assert assign.shape == (B, H, NB, 4, 3)

    # Head group 0 (heads 0, 1) has arity 2 -> branch index 2 must have exact 0.0 probability
    assert torch.all(assign[:, :2, :, :, 2] == 0.0)
    # Probabilities across branches sum to 1.0 for every head
    assert torch.allclose(assign.sum(dim=-1), torch.ones(B, H, NB, 4), atol=1e-5)

    # Differentiable expected block distance
    dist = compute_expected_block_distance(assign)
    assert dist.shape == (B, H, NB, NB)
    loss = dist.mean() + lb_loss
    loss.backward()
    assert qk_desc.grad is not None
    assert not torch.isnan(qk_desc.grad).any()

    # Check strong triangle inequality on hard eval assignments
    router.eval()
    with torch.no_grad():
        hard_assign, _ = router(qk_desc)
        d_mat = compute_expected_block_distance(hard_assign)[0, 2]  # head 2 (3-ary)
        for i in range(NB):
            assert abs(d_mat[i, i].item()) < 1e-5
            for j in range(NB):
                for k in range(NB):
                    assert d_mat[i, k].item() <= max(d_mat[i, j].item(), d_mat[j, k].item()) + 1e-5


# ============================================================================
# 2. SurgicalLlamaAttention: QK-Mode Pooling, Swin Shift, Route Stride, KL + Lagrangian
# ============================================================================


def test_valid_length_masked_pooling_no_padding_dilution():
    """Verify Trap 3 fix: a 1-token tail block or decode step is never diluted by zero padding."""
    B, H, D = 1, 2, 16
    RB = 8
    # Single token tensor (S = 1) vs RB = 8
    x_single = torch.randn(B, H, 1, D)
    pooled_single = SurgicalLlamaAttention._pool_tensor_blocks(x_single, route_block=RB)
    assert pooled_single.shape == (B, H, 1, D)
    assert torch.allclose(pooled_single, x_single, atol=1e-6)

    # Sequence of length RB + 1 = 9: the second block has exactly 1 valid token (index 8)
    x_nine = torch.randn(B, H, 9, D)
    pooled_nine = SurgicalLlamaAttention._pool_tensor_blocks(x_nine, route_block=RB)
    assert pooled_nine.shape == (B, H, 2, D)
    # Tail block (index 1) has 1 valid token -> both masked mean and peak-norm equal x_nine[:, :, 8, :]
    assert torch.allclose(pooled_nine[:, :, 1, :], x_nine[:, :, 8, :], atol=1e-6)


def test_surgical_attention_qk_mode_swin_shift_and_kl_lagrangian():
    cfg = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        surgical_p=2,
        surgical_prime_arities=[2, 3],
        surgical_qk_mode=True,
        surgical_swin_shift=True,
        surgical_route_block=8,
        surgical_local_window=4,
        surgical_tree_depth=3,
        surgical_distill_kl=True,
        surgical_target_density=0.20,
        surgical_lambda_lr=0.1,
        surgical_init_mode="random",
    )
    attn_even = SurgicalLlamaAttention(cfg, layer_idx=0)
    attn_odd = SurgicalLlamaAttention(cfg, layer_idx=1)

    attn_even.train()
    attn_odd.train()

    x = torch.randn(2, 20, 64, requires_grad=True)
    lambda_before = attn_odd.density_lambda.item()

    out_even, _ = attn_even(x)
    out_odd, _ = attn_odd(x)
    assert out_even.shape == (2, 20, 64)
    assert out_odd.shape == (2, 20, 64)
    assert attn_odd.current_distill_loss is not None
    assert not torch.isnan(attn_odd.current_distill_loss)

    total_loss = out_odd.sum() + attn_odd.current_distill_loss + attn_odd.current_penalty
    total_loss.backward()

    assert attn_odd.router.backbone.weight.grad is not None
    assert attn_odd.router.route_heads.weight.grad is not None
    assert abs(attn_odd.density_lambda.item() - lambda_before) > 0.0


# ============================================================================
# 3. Multi-Architecture Patching (Frontier Models + Route Stride)
# ============================================================================


class _DummyRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1) * self.weight


class _DummyFrontierAttn(nn.Module):
    """Simulates Qwen3.5/Gemma4 attention with non-matching head_dim * num_heads != hidden_size,
    gated 2x q_proj, q_norm/k_norm/v_norm, and optional k_eq_v (v_proj=None)."""

    def __init__(self, hidden_size=48, num_heads=4, num_kv_heads=2, head_dim=16, gated_q=True, k_eq_v=False, is_sliding=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / (head_dim ** 0.5)
        self.is_sliding = is_sliding
        q_out = (2 * num_heads * head_dim) if gated_q else (num_heads * head_dim)
        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = None if k_eq_v else nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = _DummyRMSNorm(head_dim)
        self.k_norm = _DummyRMSNorm(head_dim)
        self.v_norm = _DummyRMSNorm(head_dim)


class _DummyLinearDeltaNetLayer(nn.Module):
    """Simulates a hybrid linear-attention layer without q_proj/k_proj."""

    def __init__(self, hidden_size=48):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)


class _DummyTransformerLayer(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.self_attn = attn_module

    def forward(self, x):
        out, _ = self.self_attn(x)
        return x + out


class _DummyVLMWrapper(nn.Module):
    """Simulates Gemma 4 / Qwen 3.5 VLM wrapper where layers live at model.model.language_model.layers."""

    def __init__(self):
        super().__init__()
        text_cfg = SimpleNamespace(
            hidden_size=48,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        self.config = SimpleNamespace(text_config=text_cfg, hidden_size=48, num_attention_heads=4)
        layers = nn.ModuleList([
            _DummyTransformerLayer(_DummyFrontierAttn(gated_q=True, k_eq_v=False, is_sliding=False)),
            _DummyTransformerLayer(_DummyFrontierAttn(gated_q=True, k_eq_v=True, is_sliding=True)),
            _DummyLinearDeltaNetLayer(hidden_size=48),
            _DummyTransformerLayer(_DummyFrontierAttn(gated_q=False, k_eq_v=False, is_sliding=False)),
        ])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=layers))


def test_frontier_vlm_patcher_and_route_stride():
    vlm = _DummyVLMWrapper()
    patch_llama_model(
        vlm,
        tree_depth=3,
        arity=2,
        qk_mode=True,
        prime_arities=[2, 3],
        swin_shift=True,
        route_stride=4,
        route_block=4,
    )
    layers = vlm.model.language_model.layers
    assert isinstance(layers[0].self_attn, SurgicalLlamaAttention)
    assert isinstance(layers[1].self_attn, SurgicalLlamaAttention)
    # Layer 2 is a linear DeltaNet layer -> must be skipped untouched!
    assert isinstance(layers[2], _DummyLinearDeltaNetLayer)
    assert not isinstance(layers[2].self_attn, SurgicalLlamaAttention)
    assert isinstance(layers[3].self_attn, SurgicalLlamaAttention)

    # Check q_norm, k_norm, v_norm and k_eq_v preservation
    assert layers[0].self_attn.q_norm is not None
    assert layers[1].self_attn.v_proj is None

    # Forward pass through patched layers and check route_stride cache sharing
    x = torch.randn(1, 12, 48)
    vlm.eval()
    with torch.no_grad():
        h = layers[0](x)
        assert 0 in vlm._surgical_route_cache
        h = layers[1](h)
        h = layers[3](h)
        assert h.shape == (1, 12, 48)
        # Layers 1 and 3 reused anchor layer 0's cached block assignments
        assert torch.equal(
            layers[0].self_attn._cached_block_assignments,
            layers[1].self_attn._cached_block_assignments,
        )


# ============================================================================
# 4. SurprisalAdelicCache: Needle Protection & Zero Dummy Padding
# ============================================================================


def test_surprisal_adelic_cache_needle_retention_and_zero_padding():
    B, H, D = 2, 4, 16
    max_cap = 32
    local_win = 12
    outlier_budget = 4
    condensed_budget = 16

    cache = SurprisalAdelicCache(
        max_capacity=max_cap,
        local_window=local_win,
        outlier_budget=outlier_budget,
        condensed_budget=condensed_budget,
        lambda_decay=0.05,
        k_outlier=2,
    )

    torch.manual_seed(42)
    # Create 40 tokens of correlated background "haystack" + 1 orthogonal semantic "needle" at pos 7
    base_dir = F.normalize(torch.randn(B, H, 1, D), p=2, dim=-1)
    noise = 0.05 * torch.randn(B, H, 48, D)
    keys = F.normalize(base_dir + noise, p=2, dim=-1) * 2.0
    values = F.normalize(base_dir + 0.05 * torch.randn(B, H, 48, D), p=2, dim=-1) * 2.0

    # Plant an orthogonal needle at token index 7 (well inside far-history [0..35])
    needle_k = torch.zeros(B, H, D)
    needle_v = torch.zeros(B, H, D)
    needle_k[..., 0] = 15.0
    needle_v[..., 1] = -15.0
    keys[:, :, 7, :] = needle_k
    values[:, :, 7, :] = needle_v

    k_out, v_out = cache.update(keys, values, layer_idx=0)

    # 1. Exact physical capacity across all batches and heads
    assert k_out.shape == (B, H, max_cap, D)
    assert v_out.shape == (B, H, max_cap, D)
    assert cache.get_seq_length(0) == max_cap
    assert cache.get_logical_seq_length() == 48

    # 2. ZERO dummy (K=0, V=0) padding anywhere in the cache (Trap 5 fix)
    k_norms = torch.norm(k_out, p=2, dim=-1)
    v_norms = torch.norm(v_out, p=2, dim=-1)
    assert torch.all(k_norms > 1e-4), "Found zero-padded dummy key vectors in condensed cache!"
    assert torch.all(v_norms > 1e-4), "Found zero-padded dummy value vectors in condensed cache!"

    # 3. Needle-in-a-Haystack retention: exact needle_k and needle_v must exist in Tier 1 [0..outlier_budget-1]
    tier1_k = k_out[:, :, :outlier_budget, :]
    tier1_v = v_out[:, :, :outlier_budget, :]
    for b in range(B):
        for h in range(H):
            k_diffs = (tier1_k[b, h] - needle_k[b, h].unsqueeze(0)).abs().max(dim=-1).values
            v_diffs = (tier1_v[b, h] - needle_v[b, h].unsqueeze(0)).abs().max(dim=-1).values
            assert k_diffs.min().item() < 1e-5, f"Needle K lost at b={b}, h={h}"
            assert v_diffs.min().item() < 1e-5, f"Needle V lost at b={b}, h={h}"

    # 4. Token 0 (Attention Sink) is also preserved in Tier 1
    assert torch.allclose(tier1_k[:, :, 0, :], keys[:, :, 0, :], atol=1e-5)


# ============================================================================
# 5. Non-Power-of-Two ultrametric_decode & Topology Parity
# ============================================================================


@pytest.mark.parametrize("tree_depth", [1, 3, 5, 6, 7, 10])
def test_ultrametric_decode_non_power_of_two_depths(tree_depth):
    torch.manual_seed(100 + tree_depth)
    B, H, D = 2, 4, 16
    num_blocks = 6
    block_size = 8
    req_depth = min(3, tree_depth)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(B, H, D, device=device, dtype=torch.float32)
    k_cache = torch.randn(num_blocks, H, block_size, D, device=device, dtype=torch.float32)
    v_cache = torch.randn(num_blocks, H, block_size, D, device=device, dtype=torch.float32)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    context_lens = torch.tensor([40, 35], device=device, dtype=torch.int32)

    router_indices = torch.randint(0, 2, (num_blocks, H, tree_depth), device=device, dtype=torch.int32)
    q_router_indices = torch.randint(0, 2, (B, H, tree_depth), device=device, dtype=torch.int32)

    out = ultrametric_decode(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        router_indices,
        q_router_indices,
        req_depth=req_depth,
        local_blocks=1,
        sink_block=True,
    )
    ref = _ultrametric_decode_reference(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        router_indices,
        q_router_indices,
        req_depth=req_depth,
        local_blocks=1,
        sink_block=True,
    )
    assert out.shape == (B, H, D)
    assert torch.allclose(out, ref, atol=1e-4)


# ============================================================================
# 6. SpeculativeSurgicalGenerator: Self-Speculative Decoding Equivalence
# ============================================================================


class _TinySurgicalCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            surgical_p=2,
            surgical_tree_depth=3,
            surgical_req_depth=0,
            surgical_local_window=2,
            surgical_init_mode="random",
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.attn = SurgicalLlamaAttention(self.config, layer_idx=0)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, use_cache=False, **kwargs):
        x = self.embed(input_ids)
        attn_out, _ = self.attn(x)
        logits = self.lm_head(x + attn_out)
        return SimpleNamespace(logits=logits)


def test_speculative_surgical_generator_exact_greedy_and_stochastic():
    torch.manual_seed(77)
    model = _TinySurgicalCausalLM(vocab_size=24, hidden_size=32)
    model.eval()

    prompt = torch.tensor([[1, 5, 3, 8]], dtype=torch.long)
    max_new = 7

    # 1. Greedy baseline with dense verifier (req_depth = 0)
    model.config.surgical_req_depth = 0
    expected_seq = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(expected_seq).logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            expected_seq = torch.cat([expected_seq, next_tok], dim=-1)

    # 2. Self-speculative greedy generation (draft_req_depth=3, verify_req_depth=0)
    spec_gen = SpeculativeSurgicalGenerator(
        model, gamma=3, draft_req_depth=3, verify_req_depth=0
    )
    spec_seq, stats = spec_gen.generate(
        prompt, max_new_tokens=max_new, temperature=0.0, return_stats=True
    )

    assert spec_seq.shape == (1, prompt.shape[1] + max_new)
    # Greedy speculative decoding is mathematically guaranteed to match greedy target decoding!
    assert torch.equal(spec_seq, expected_seq)
    assert stats["total_drafted"] > 0
    assert 0.0 <= stats["acceptance_rate"] <= 1.0

    # 3. Stochastic speculative rejection sampling (temperature = 0.7)
    spec_gen.reset_stats()
    stoch_seq, stoch_stats = spec_gen.generate(
        prompt, max_new_tokens=5, temperature=0.7, return_stats=True
    )
    assert stoch_seq.shape == (1, prompt.shape[1] + 5)
    assert stoch_stats["total_steps"] >= 1


# ============================================================================
# 7. V4 Upgrades: TriadTopologyRouter, BeltramiAdelicCache, ZenoSpeculativeGenerator
# ============================================================================


def test_v4_triad_topology_router_conservation_and_deep_gradients():
    from llama_surgery import TriadTopologyRouter

    B, H, NB, D = 2, 4, 8, 16
    router = TriadTopologyRouter(
        embed_dim=H * D,
        seq_len=512,
        num_heads=H,
        p=2,
        init_mode="collapse",
        levels=5,
        qk_mode=True,
        head_dim=D,
        d_shell=8,
    )
    assert router.triad_mode is True

    # 1. Step-0 collapse homotopy: exact dense mask (branch 0 = 1.0 across all 5 levels)
    router.eval()
    qk_desc = torch.randn(B, H, NB, 2 * D)
    assign_eval, _ = router(qk_desc)
    assert assign_eval.shape == (B, H, NB, 5, 2)
    assert torch.all(assign_eval[..., 0] == 1.0)
    assert torch.all(assign_eval[..., 1] == 0.0)
    assert router.last_triad_flux is not None
    assert router.last_triad_flux.shape == (B, H, NB)

    # 2. Skew-symmetric bilinear triad orthogonality check: <f_ell, B_W(parent_ctx, f_ell)> == 0
    ctx_dim = router.levels * router.p + router.d_shell
    u_parent = torch.randn(B, H, NB, ctx_dim)
    f_ell = torch.randn(B, H, NB, router.d_shell)
    W_raw = router.triad_weight[:, 0]  # (H, ctx_dim, M, M)
    W_skew = W_raw - W_raw.transpose(-1, -2)
    triad_pump = torch.einsum("bhni,bhnj,hijm->bhnm", u_parent, f_ell, W_skew)
    inner_prod = (f_ell * triad_pump).sum(dim=-1)
    assert torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=1e-5)

    # 3. Non-starved gradient flow through all levels l=1..4 (no cummin tie-breaking starvation)
    router_rand = TriadTopologyRouter(
        embed_dim=H * D,
        seq_len=512,
        num_heads=H,
        p=2,
        init_mode="random",
        levels=5,
        qk_mode=True,
        head_dim=D,
        d_shell=8,
    )
    router_rand.train()
    qk_train = torch.randn(B, H, NB, 2 * D, requires_grad=True)
    assign_train, lb_loss = router_rand(qk_train, tau_override=1.0)
    dist = compute_expected_block_distance(assign_train)
    target_sim = torch.eye(NB).unsqueeze(0).unsqueeze(0)
    loss = (dist * target_sim).mean() + lb_loss
    loss.backward()

    assert router_rand.triad_weight.grad is not None
    assert router_rand.triad_readout.grad is not None
    for ell in range(4):
        assert router_rand.triad_weight.grad[:, ell].abs().max().item() > 0.0
        assert router_rand.triad_readout.grad[:, ell].abs().max().item() > 0.0


def test_v4_patcher_float32_router_and_stride_sidecar_sync(tmp_path):
    from llama_surgery import save_router_sidecar, load_router_sidecar

    vlm = _DummyVLMWrapper()
    # Cast backbone to bfloat16 to verify router stays in float32
    for p in vlm.parameters():
        p.data = p.data.to(torch.bfloat16)

    patch_llama_model(
        vlm,
        tree_depth=4,
        arity=2,
        qk_mode=True,
        prime_arities=[2, 3],
        swin_shift=True,
        route_stride=2,
        route_block=4,
        triad_mode=True,
        d_shell=8,
    )
    layers = vlm.model.language_model.layers
    # Router parameters must remain float32 even when model is bfloat16
    assert layers[0].self_attn.router.backbone.weight.dtype == torch.float32
    assert layers[0].self_attn.router.triad_weight.dtype == torch.float32
    assert layers[0].self_attn.router.shell_inject.dtype == torch.float32

    # Modify anchor layer 0 router weights, leave non-anchor layer 1 untouched
    with torch.no_grad():
        layers[0].self_attn.router.backbone.weight.fill_(0.1234)
        layers[1].self_attn.router.backbone.weight.zero_()

    sidecar_file = str(tmp_path / "v4_router_sidecar.safetensors")
    save_router_sidecar(vlm, sidecar_file)

    # Verify save_router_sidecar synchronized anchor layer 0 weights to non-anchor layer 1
    assert torch.allclose(
        layers[1].self_attn.router.backbone.weight,
        layers[0].self_attn.router.backbone.weight,
        atol=1e-6,
    )

    # Verify reload into a fresh model
    vlm2 = _DummyVLMWrapper()
    patch_llama_model(
        vlm2,
        tree_depth=4,
        arity=2,
        qk_mode=True,
        prime_arities=[2, 3],
        swin_shift=True,
        route_stride=2,
        route_block=4,
        triad_mode=True,
        d_shell=8,
    )
    meta = load_router_sidecar(vlm2, sidecar_file)
    assert meta["format"] == "sparse-ai-router-sidecar-v4"
    assert meta["surgical_triad_mode"] == "True"
    assert torch.allclose(
        vlm2.model.language_model.layers[1].self_attn.router.backbone.weight,
        layers[0].self_attn.router.backbone.weight,
        atol=1e-6,
    )


def test_v4_beltrami_adelic_cache_passcode_and_banal_needle_shielding():
    from llama_surgery import BeltramiAdelicCache

    B, H, D = 1, 2, 16
    half_d = D // 2
    S_far = 64
    local_win = 8
    outlier_budget = 8
    condensed_budget = 16
    max_cap = outlier_budget + condensed_budget + local_win

    torch.manual_seed(2026)
    # Build coherent laminar background where coarse and fine shells are collinear (curl = 0)
    u_coarse_bg = F.normalize(torch.randn(B, H, 1, half_d), dim=-1)
    noise_half = 0.05 * torch.randn(B, H, S_far + local_win, half_d)
    bg_half = F.normalize(u_coarse_bg + noise_half, dim=-1)
    keys = torch.cat([bg_half, bg_half], dim=-1) * 2.5
    values = keys.clone()

    # Inject 8 high-norm laminar attention sinks at positions 2..9 (norm = 18.0, collinear coarse/fine shells)
    for idx, pos in enumerate(range(2, 10)):
        sink_half = torch.zeros(B, H, half_d)
        sink_half[..., idx % half_d] = 1.0
        # Collinear coarse & fine shells -> zero cross-shell curl, but huge static norm & isolation!
        sink_vec = torch.cat([sink_half, sink_half], dim=-1) * 18.0
        keys[:, :, pos, :] = sink_vec
        values[:, :, pos, :] = sink_vec

    # Inject a low-norm (||k|| = 1.8) 4-token banal needle / passcode at positions 28..31:
    # Coarse shell blends with background topic, while fine shell carries orthogonal Beltrami curl & innovation!
    passcode_positions = [28, 29, 30, 31]
    passcode_vecs = []
    for idx, pos in enumerate(passcode_positions):
        fine_vec = torch.zeros(B, H, half_d)
        fine_vec[..., (idx * 2 + 1) % half_d] = 1.0
        # Orthogonal to u_coarse_bg in fine shell -> maximal cross-shell curl and fine-shell innovation
        k_vec = torch.cat([u_coarse_bg.squeeze(2), fine_vec], dim=-1) * 1.2
        v_vec = torch.cat([-u_coarse_bg.squeeze(2), fine_vec * 2.0], dim=-1) * 1.5
        keys[:, :, pos, :] = k_vec
        values[:, :, pos, :] = v_vec
        passcode_vecs.append(v_vec)

    # 1. Static Euclidean norm shield (beltrami_mode=False) gets hijacked by the 8 collinear laminar sinks
    static_cache = SurprisalAdelicCache(
        max_capacity=max_cap,
        local_window=local_win,
        outlier_budget=outlier_budget,
        condensed_budget=condensed_budget,
        beltrami_mode=False,
    )
    _, v_static = static_cache.update(keys.clone(), values.clone(), layer_idx=0)
    tier1_v_static = v_static[:, :, :outlier_budget, :]

    # 2. V4 Beltrami Triad-Flux shield (BeltramiAdelicCache) rescues the low-norm passcode
    beltrami_cache = BeltramiAdelicCache(
        max_capacity=max_cap,
        local_window=local_win,
        outlier_budget=outlier_budget,
        condensed_budget=condensed_budget,
    )
    _, v_beltrami = beltrami_cache.update(keys.clone(), values.clone(), layer_idx=0)
    tier1_v_beltrami = v_beltrami[:, :, :outlier_budget, :]

    static_retained = 0
    beltrami_retained = 0
    for target_v in passcode_vecs:
        if (tier1_v_static - target_v.unsqueeze(2)).abs().max(dim=-1).values.min().item() < 1e-4:
            static_retained += 1
        if (tier1_v_beltrami - target_v.unsqueeze(2)).abs().max(dim=-1).values.min().item() < 1e-4:
            beltrami_retained += 1

    assert beltrami_retained == len(passcode_positions), (
        f"BeltramiAdelicCache should retain all {len(passcode_positions)} low-norm passcode tokens, got {beltrami_retained}"
    )
    assert beltrami_retained > static_retained


def test_v4_zeno_speculative_generator_dynamic_depth_and_greedy_equivalence():
    from llama_surgery import ZenoSpeculativeGenerator

    torch.manual_seed(99)
    model = _TinySurgicalCausalLM(vocab_size=24, hidden_size=32)
    # Upgrade model's attention router to TriadTopologyRouter (triad_mode=True)
    model.config.surgical_triad_mode = True
    model.attn = SurgicalLlamaAttention(model.config, layer_idx=0)
    model.eval()

    prompt = torch.tensor([[2, 4, 6, 8]], dtype=torch.long)
    max_new = 6

    # 1. Dense greedy reference (req_depth = 0)
    model.config.surgical_req_depth = 0
    expected_seq = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(expected_seq).logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            expected_seq = torch.cat([expected_seq, next_tok], dim=-1)

    # 2. ZenoSpeculativeGenerator greedy verification
    zeno_gen = ZenoSpeculativeGenerator(
        model,
        gamma=3,
        draft_req_depth=2,
        verify_req_depth=0,
        min_draft_depth=1,
        max_draft_depth=3,
        flux_threshold=0.25,
    )
    zeno_seq, stats = zeno_gen.generate(
        prompt, max_new_tokens=max_new, temperature=0.0, return_stats=True
    )

    assert torch.equal(zeno_seq, expected_seq)
    assert len(stats["draft_depths_used"]) == stats["total_drafted"]
    assert all(1 <= d <= 3 for d in stats["draft_depths_used"])

