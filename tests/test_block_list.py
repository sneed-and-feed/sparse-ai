"""
Tests for ultrametric.block_list (block-list sparse attention).

* Builder tests: pure PyTorch, run anywhere (CPU).
* Kernel tests: need Triton plus either CUDA or the Triton CPU interpreter
  (set TRITON_INTERPRET=1 *before* pytest starts, e.g.
  ``TRITON_INTERPRET=1 pytest tests/test_block_list.py``). Skipped otherwise.
"""

import math
import os

import pytest
import torch

from ultrametric.block_list import (
    HAS_TRITON,
    block_list_attention,
    block_mask_reference,
    build_block_lists,
    masked_reference_attention,
    token_mask_from_blocks,
)

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
CAN_RUN_KERNEL = HAS_TRITON and (torch.cuda.is_available() or INTERPRET)
DEVICE = "cuda" if (torch.cuda.is_available() and not INTERPRET) else "cpu"


def random_router(Z, H, NB, TD, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, p, (Z, H, NB, TD), generator=g, dtype=torch.int32)


def tree_router(Z, H, NB, TD):
    """Natural binary path of each block index (MSB first), padded to TD levels."""
    depth = max(1, math.ceil(math.log2(max(NB, 2))))
    r = torch.zeros(Z, H, NB, TD, dtype=torch.int32)
    for d in range(depth):
        r[:, :, :, d] = (torch.arange(NB) >> (depth - 1 - d)) % 2
    return r


# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("sink_block,local_blocks", [(False, 0), (True, 0), (False, 1), (True, 2)])
def test_builder_matches_pairwise_reference(p, is_causal, sink_block, local_blocks):
    Z, H, NB, TD = 2, 3, 7, 4
    router = random_router(Z, H, NB, TD, p, seed=p)
    depth = torch.tensor([0, 2, 4])  # per-head depths incl. dense and full
    kw = dict(is_causal=is_causal, sink_block=sink_block, local_blocks=local_blocks)

    bl = build_block_lists(router, depth, **kw)
    ref = block_mask_reference(router, depth, **kw)

    assert bl.lists.dtype == torch.int32 and bl.counts.dtype == torch.int32
    assert torch.equal(bl.counts, ref.sum(-1).to(torch.int32))
    assert bl.max_count == int(ref.sum(-1).max())
    for z in range(Z):
        for h in range(H):
            for i in range(NB):
                c = int(bl.counts[z, h, i])
                expected = torch.nonzero(ref[z, h, i]).flatten().to(torch.int32)
                assert torch.equal(bl.lists[z, h, i, :c], expected), (z, h, i)


def test_diagonal_always_present():
    router = random_router(1, 2, 9, 3, 2, seed=1)
    for causal in (False, True):
        bl = build_block_lists(router, 3, is_causal=causal)
        for h in range(2):
            for i in range(9):
                c = int(bl.counts[0, h, i])
                assert i in bl.lists[0, h, i, :c].tolist()


def test_depth_zero_is_dense():
    router = random_router(1, 2, 5, 3, 2)
    bl = build_block_lists(router, 0)
    assert torch.all(bl.counts == 5)
    blc = build_block_lists(router, 0, is_causal=True)
    assert torch.equal(blc.counts[0, 0], torch.arange(1, 6, dtype=torch.int32))


def test_tree_routing_full_depth_is_block_diagonal():
    NB = 16
    router = tree_router(1, 1, NB, 4)
    bl = build_block_lists(router, 4)
    assert torch.all(bl.counts == 1)
    assert torch.equal(bl.lists[0, 0, :, 0], torch.arange(NB, dtype=torch.int32))
    # Depth r on a balanced tree leaves NB / 2^r blocks per row.
    for r in range(5):
        assert torch.all(build_block_lists(router, r).counts == NB // 2 ** r)


def test_invalid_depth_raises():
    router = random_router(1, 2, 4, 3, 2)
    with pytest.raises(ValueError):
        build_block_lists(router, 4)
    with pytest.raises(ValueError):
        build_block_lists(router, [1, 2, 3])
    with pytest.raises(ValueError):
        build_block_lists(router, 4, arity=2)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 3, [0, 2]])
def test_sync_free_builder_matches(p, is_causal, depth):
    router = random_router(2, 2, 9, 3, p, seed=7)
    a = build_block_lists(router, depth, is_causal=is_causal, sink_block=True)
    b = build_block_lists(router, depth, is_causal=is_causal, sink_block=True, arity=p)
    assert b.max_count == 9 and b.lists.shape[-1] == 9
    assert torch.equal(a.counts, b.counts)
    for z in range(2):
        for h in range(2):
            for i in range(9):
                c = int(a.counts[z, h, i])
                assert torch.equal(a.lists[z, h, i, :c], b.lists[z, h, i, :c])


# ----------------------------------------------------------------------------
# Kernel vs. block-masked fp32 reference
# ----------------------------------------------------------------------------

KERNEL_CASES = [
    # (N, route_block, block_m, block_n, depth, is_causal, sink, local)
    (200, 64, 32, 32, 1, False, False, 0),   # partial tail block
    (200, 64, 32, 32, 2, True, False, 0),    # causal + tail
    (256, 64, 64, 32, 2, True, True, 0),     # causal + sink, BLOCK_M == route block
    (192, 64, 32, 64, 1, False, False, 1),   # local band, BLOCK_N == route block
    (160, 32, 32, 32, 0, True, False, 0),    # dense causal
]


@pytest.mark.skipif(not CAN_RUN_KERNEL, reason="needs Triton + CUDA, or TRITON_INTERPRET=1")
@pytest.mark.parametrize("N,rb,bm,bn,depth,is_causal,sink,local", KERNEL_CASES)
def test_kernel_matches_masked_reference(N, rb, bm, bn, depth, is_causal, sink, local):
    torch.manual_seed(0)
    Z, H, D = 1, 2, 32
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    q, k, v = (torch.randn(Z, H, N, D, device=DEVICE, dtype=dtype) for _ in range(3))
    NB = math.ceil(N / rb)
    router = random_router(Z, H, NB, 3, 2, seed=N).to(DEVICE)
    kw = dict(is_causal=is_causal, sink_block=sink, local_blocks=local)

    bl = build_block_lists(router, depth, **kw)
    out = block_list_attention(q, k, v, bl, route_block=rb, is_causal=is_causal,
                               block_m=bm, block_n=bn)

    mask = token_mask_from_blocks(block_mask_reference(router, depth, **kw), rb, N, is_causal)
    ref = masked_reference_attention(q, k, v, mask)
    tol = 2e-3 if dtype == torch.float16 else 1e-4
    err = (out.float() - ref.float()).abs().max().item()
    assert err < tol, f"max abs err {err:.2e}"
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available() and not INTERPRET),
                    reason="autotuned path needs CUDA")
@pytest.mark.parametrize("is_causal", [False, True])
def test_autotuned_kernel_cuda(is_causal):
    torch.manual_seed(0)
    Z, H, N, D, rb = 2, 4, 1000, 64, 128
    q, k, v = (torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    router = random_router(Z, H, math.ceil(N / rb), 4, 2, seed=7).cuda()
    bl = build_block_lists(router, 2, is_causal=is_causal)
    out = block_list_attention(q, k, v, bl, route_block=rb, is_causal=is_causal)
    mask = token_mask_from_blocks(block_mask_reference(router, 2, is_causal=is_causal), rb, N, is_causal)
    ref = masked_reference_attention(q, k, v, mask)
    assert (out.float() - ref.float()).abs().max().item() < 2e-3
