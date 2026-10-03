"""
Surgery integration of the block-list kernel (config.surgical_attention_backend).

Checks SurgicalLlamaAttention._block_list_prefill against an independent
construction: per-block majority routing computed with Python loops, block mask
from the pairwise oracle, token mask, fp32 masked attention. Also checks that the
reported measured budget equals the true number of allowed keys.

Kernel tests need CUDA or TRITON_INTERPRET=1 (set before pytest starts).
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from ultrametric.block_list import (
    HAS_TRITON,
    block_mask_reference,
    masked_reference_attention,
    token_mask_from_blocks,
)

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
CAN_RUN_KERNEL = HAS_TRITON and (torch.cuda.is_available() or INTERPRET)
DEVICE = "cuda" if (torch.cuda.is_available() and not INTERPRET) else "cpu"


def make_attn(route_block=32, local_window=16, backend="block_list"):
    from llama_surgery.surgery import SurgicalLlamaAttention
    cfg = SimpleNamespace(
        hidden_size=64, num_attention_heads=2, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=1024, surgical_p=2, surgical_init_mode="random",
        surgical_route_block=route_block, surgical_local_window=local_window,
        surgical_attention_backend=backend, surgical_collect_stats=True,
    )
    return SurgicalLlamaAttention(cfg, layer_idx=0)


def majority_route_reference(assign, r, rb):
    """Python-loop per-level majority vote of token argmax branches per block."""
    B, H, S, _, p = assign.shape
    nb = math.ceil(S / rb)
    idx = assign[..., :r, :].argmax(-1)
    out = torch.zeros(B, H, nb, r, dtype=torch.int32)
    for b in range(B):
        for h in range(H):
            for i in range(nb):
                seg = idx[b, h, i * rb:(i + 1) * rb]               # (len, r)
                for lvl in range(r):
                    cnt = torch.bincount(seg[:, lvl], minlength=p)
                    out[b, h, i, lvl] = int(torch.argmax(cnt))      # ties -> lowest id
    return out


def test_eligibility_gates():
    attn = make_attn(backend="eager")
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        assert not attn._block_list_eligible(x, None, None)          # backend off
    attn = make_attn()
    with torch.no_grad():
        if not torch.cuda.is_available():
            assert not attn._block_list_eligible(x, None, None)      # CPU tensor
    assert not attn._block_list_eligible(x, None, None)              # grad enabled


@pytest.mark.skipif(not CAN_RUN_KERNEL, reason="needs Triton + CUDA, or TRITON_INTERPRET=1")
@pytest.mark.parametrize("S,rb,lw,r", [(200, 32, 16, 2), (256, 64, 0, 3), (96, 32, 40, 1)])
def test_block_list_prefill_matches_reference(S, rb, lw, r):
    torch.manual_seed(S + r)
    attn = make_attn(route_block=rb, local_window=lw).to(DEVICE)
    B, H, D, L = 1, 2, 32, attn.router.levels
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    q, k, v = (torch.randn(B, H, S, D, device=DEVICE, dtype=dtype) for _ in range(3))
    assign = torch.softmax(torch.randn(B, H, S, L, 2), dim=-1)

    with torch.no_grad():
        out = attn._block_list_prefill(q, k, v, assign.to(DEVICE), r)

    route = majority_route_reference(assign, r, rb)
    local_blocks = math.ceil(lw / rb) if lw > 0 else 0
    bm = block_mask_reference(route, r, is_causal=True, sink_block=True, local_blocks=local_blocks)
    tm = token_mask_from_blocks(bm, rb, S, is_causal=True).to(DEVICE)
    ref = masked_reference_attention(q.float(), k.float(), v.float(), tm, sm_scale=attn.scale)

    tol = 2e-3 if dtype == torch.float16 else 1e-4
    err = (out.float() - ref).abs().max().item()
    assert err < tol, f"max abs err {err:.2e}"

    true_allowed = tm.sum(-1).float().mean().item()
    assert abs(attn.last_mean_allowed_keys - true_allowed) < 1e-3, (attn.last_mean_allowed_keys, true_allowed)
    assert abs(attn.last_mean_causal_keys - (S + 1) / 2) < 1e-9
