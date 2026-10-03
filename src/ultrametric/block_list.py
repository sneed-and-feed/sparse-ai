"""
Block-list sparse attention (forward) for routed / ultrametric block sparsity
=============================================================================

Problem with the existing kernels
---------------------------------
* ``benchmarks/benchmark_triton.py`` (the kernel behind the README numbers) visits
  EVERY key block and tests its route before skipping, so loop overhead stays
  O((N/B)^2) even at 98% sparsity. Measured: 5.7x over its own dense path at
  98.4% sparsity, vs. an ideal of 64x.
* ``ultrametric.kernel.ultrametric_attention_triton`` uses block lists, but passes
  the maximum list length as a ``tl.constexpr`` (a fresh compile for every new
  value) and runs every query block to that maximum: padded iterations still load
  K/V and execute both matmuls, so cost follows the *longest* list, not each row's.

This module
-----------
1. ``build_block_lists`` turns per-block routing vectors into, for every
   (batch, head, query route-block), an ascending list of active key route-blocks
   plus its length. Pure PyTorch, vectorized, built once per forward and reusable.
2. ``block_list_attention`` loops over **only that row's own list length**
   (runtime loop bound), so cost scales with the number of active blocks per row.
   Each route block (default 128 tokens) is tiled into BLOCK_N sub-tiles;
   tile sizes are autotuned on CUDA. Causality and the sequence tail are masked at
   token granularity; an optional sink block and local block band mirror the
   LLaMA-surgery inference mask.

Matching rule (same as the existing kernels): query block i attends to key block j
iff their routing vectors agree on the first ``req_depth[h]`` levels.

Forward only. Testing: ``tests/test_block_list.py`` checks the list builder against
an independent pairwise construction, and the kernel against a block-masked fp32
reference (on CUDA, or on CPU via ``TRITON_INTERPRET=1``).
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple, Optional, Sequence, Union

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, RuntimeError):
    HAS_TRITON = False

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

DepthLike = Union[int, Sequence[int], torch.Tensor]

__all__ = [
    "HAS_TRITON",
    "BlockLists",
    "build_block_lists",
    "block_list_attention",
    "routed_block_attention",
    "block_mask_reference",
    "token_mask_from_blocks",
    "masked_reference_attention",
]


# ============================================================================
# Block-list construction (pure PyTorch)
# ============================================================================

class BlockLists(NamedTuple):
    """Active key route-blocks per query route-block.

    lists:     (Z, H, NB, max_count) int32. Row (z, h, i) holds the active key
               blocks for query block i in ascending order; entries at positions
               >= counts[z, h, i] are padding and are never read by the kernel.
    counts:    (Z, H, NB) int32, number of active key blocks per row.
    max_count: int, max over counts (row stride of ``lists``).
    """

    lists: torch.Tensor
    counts: torch.Tensor
    max_count: int


def _depth_tensor(req_depth: DepthLike, num_heads: int, tree_depth: int,
                  device: torch.device) -> torch.Tensor:
    """Normalize req_depth to an int64 (H,) tensor and validate its range."""
    if isinstance(req_depth, int):
        depth = torch.full((num_heads,), req_depth, dtype=torch.int64, device=device)
    else:
        depth = torch.as_tensor(req_depth, dtype=torch.int64, device=device).reshape(-1)
        if depth.numel() != num_heads:
            raise ValueError(f"req_depth has {depth.numel()} entries, expected {num_heads} (one per head)")
    if depth.numel() and (int(depth.min()) < 0 or int(depth.max()) > tree_depth):
        raise ValueError(f"req_depth must be in [0, {tree_depth}], got {depth.tolist()}")
    return depth


def _apply_structural_masks(match: torch.Tensor, *, is_causal: bool, sink_block: bool,
                            local_blocks: int) -> torch.Tensor:
    nb = match.shape[-1]
    idx = torch.arange(nb, device=match.device)
    if local_blocks > 0:
        match = match | ((idx.view(-1, 1) - idx.view(1, -1)).abs() <= local_blocks)
    if sink_block:
        match = match.clone()
        match[..., 0] = True
    if is_causal:
        match = match & (idx.view(-1, 1) >= idx.view(1, -1))
    return match


@torch.no_grad()
def build_block_lists(
    router_indices: torch.Tensor,
    req_depth: DepthLike,
    *,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
) -> BlockLists:
    """Build per-row active key-block lists from per-block routing vectors.

    Args:
        router_indices: (Z, H, NB, TD) integer branch ids (>= 0), one routing
            vector per route block (e.g. from ``routing_to_block_indices``).
        req_depth: required matching depth; int (all heads) or per-head (H,).
            0 means every block matches (dense).
        is_causal: drop key blocks after the query block (token-level causality
            inside the diagonal block is applied by the kernel).
        sink_block: always include key block 0 (attention sink).
        local_blocks: always include key blocks with |i - j| <= local_blocks.

    Returns:
        BlockLists(lists, counts, max_count)
    """
    if router_indices.dim() != 4:
        raise ValueError(f"router_indices must be (Z, H, NB, TD), got {tuple(router_indices.shape)}")
    Z, H, NB, TD = router_indices.shape
    device = router_indices.device
    depth = _depth_tensor(req_depth, H, TD, device)
    r = router_indices.to(torch.int64)

    if r.numel() and int(r.min()) < 0:
        raise ValueError("router_indices must be non-negative branch ids")
    base = max(2, int(r.max()) + 1) if r.numel() else 2
    keep = torch.arange(TD, device=device).view(1, 1, 1, TD) < depth.view(1, H, 1, 1)

    if TD * math.log2(base) < 62:
        # Encode each block's first-d levels as a mixed-radix integer; blocks match
        # iff their codes are equal. O(Z*H*NB^2) memory instead of O(Z*H*NB^2*TD).
        powers = torch.pow(torch.tensor(base, dtype=torch.int64, device=device),
                           torch.arange(TD, dtype=torch.int64, device=device))
        code = (r * keep * powers).sum(-1)                      # (Z, H, NB)
        match = code.unsqueeze(-1) == code.unsqueeze(-2)        # (Z, H, NB, NB)
    else:
        eq = (r.unsqueeze(3) == r.unsqueeze(2)) | ~keep.unsqueeze(2)
        match = eq.all(-1)

    match = _apply_structural_masks(match, is_causal=is_causal, sink_block=sink_block,
                                    local_blocks=local_blocks)

    counts = match.sum(-1, dtype=torch.int32)                   # (Z, H, NB)
    max_count = int(counts.max()) if counts.numel() else 0

    # Stable compaction: the k-th True in a row goes to slot k (ascending order);
    # non-matching entries go to a dump slot that is sliced off.
    pos = torch.cumsum(match, dim=-1, dtype=torch.int64) - 1
    pos = torch.where(match, pos, torch.full_like(pos, max_count))
    src = torch.arange(NB, dtype=torch.int32, device=device).expand(Z, H, NB, NB)
    lists = torch.zeros((Z, H, NB, max_count + 1), dtype=torch.int32, device=device)
    lists.scatter_(-1, pos, src)
    lists = lists[..., :max_count].contiguous()

    return BlockLists(lists, counts.contiguous(), max_count)


# ============================================================================
# Independent references (tests / benchmark correctness checks)
# ============================================================================

@torch.no_grad()
def block_mask_reference(
    router_indices: torch.Tensor,
    req_depth: DepthLike,
    *,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
) -> torch.Tensor:
    """(Z, H, NB, NB) bool block mask via direct pairwise prefix comparison.

    Deliberately shares no code path with ``build_block_lists`` (per-head loop,
    explicit ``all`` over levels) so it can serve as a test oracle.
    """
    Z, H, NB, TD = router_indices.shape
    depth = _depth_tensor(req_depth, H, TD, router_indices.device).tolist()
    out = torch.empty((Z, H, NB, NB), dtype=torch.bool, device=router_indices.device)
    for h, d in enumerate(depth):
        if d == 0:
            out[:, h] = True
        else:
            rh = router_indices[:, h, :, :d]
            out[:, h] = (rh[:, :, None, :] == rh[:, None, :, :]).all(-1)
    return _apply_structural_masks(out, is_causal=is_causal, sink_block=sink_block,
                                   local_blocks=local_blocks)


def token_mask_from_blocks(block_mask: torch.Tensor, route_block: int, seq_len: int,
                           is_causal: bool = False) -> torch.Tensor:
    """Expand a (..., NB, NB) block mask to a (..., N, N) token mask."""
    m = block_mask.repeat_interleave(route_block, dim=-2).repeat_interleave(route_block, dim=-1)
    m = m[..., :seq_len, :seq_len]
    if is_causal:
        m = m & torch.ones(seq_len, seq_len, dtype=torch.bool, device=m.device).tril()
    return m


def masked_reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               token_mask: torch.Tensor,
                               sm_scale: Optional[float] = None) -> torch.Tensor:
    """fp32 masked attention reference; returns q.dtype."""
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    s = s.masked_fill(~token_mask, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


# ============================================================================
# Triton kernel
# ============================================================================

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _block_list_fwd_kernel(
        Q, K, V, Out, Lists, Counts, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        H, N_CTX, NUM_RB, MAX_COUNT, COUNT_BUCKET,
        HEAD_DIM: tl.constexpr, ROUTE_BLOCK: tl.constexpr, IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # COUNT_BUCKET is unused in the body; it is an autotune key (next pow2 of
        # MAX_COUNT) so dense and very sparse launches may pick different tiles.
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        m_valid = offs_m < N_CTX

        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_valid[:, None], other=0.0)

        # BLOCK_M divides ROUTE_BLOCK, so this query tile lies in one route block.
        row = off_hz.to(tl.int64) * NUM_RB + (start_m * BLOCK_M) // ROUTE_BLOCK
        count = tl.load(Counts + row)
        list_ptr = Lists + row * MAX_COUNT

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.4426950408889634  # log2(e): use exp2

        for t in range(0, count):  # runtime bound: only this row's active blocks
            kb = tl.load(list_ptr + t)
            for s in tl.static_range(ROUTE_BLOCK // BLOCK_N):
                cols = kb * ROUTE_BLOCK + s * BLOCK_N + offs_n
                n_valid = cols < N_CTX
                k = tl.load(k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                            mask=n_valid[None, :], other=0.0)
                v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                            mask=n_valid[:, None], other=0.0)
                qk = tl.dot(q, k) * qk_scale
                keep = n_valid[None, :]
                if IS_CAUSAL:
                    keep = keep & (offs_m[:, None] >= cols[None, :])
                qk = tl.where(keep, qk, float("-inf"))

                # Online softmax, safe for rows whose keys are all masked so far.
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.math.exp2(m_i - m_ref)
                p = tl.math.exp2(qk - m_ref[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new

        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]
        tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc.to(Out.dtype.element_ty), mask=m_valid[:, None])

    def _prune_configs(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        rb = args.get("ROUTE_BLOCK", 128)
        ok = [c for c in configs
              if c.kwargs["BLOCK_M"] <= rb and c.kwargs["BLOCK_N"] <= rb
              and rb % c.kwargs["BLOCK_M"] == 0 and rb % c.kwargs["BLOCK_N"] == 0]
        return ok or [triton.Config({"BLOCK_M": rb, "BLOCK_N": rb}, num_warps=4, num_stages=2)]

    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=4),
    ]

    _block_list_fwd_tuned = triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=["N_CTX", "COUNT_BUCKET", "HEAD_DIM", "IS_CAUSAL", "ROUTE_BLOCK"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )(_block_list_fwd_kernel)


# ============================================================================
# Python wrappers
# ============================================================================

def block_list_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lists: BlockLists,
    *,
    route_block: int = 128,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Block-sparse attention forward over precomputed block lists.

    Args:
        q, k, v: (Z, H, N, D), same shape and dtype (fp16/bf16; fp32 allowed but
            uses TF32 matmuls on GPU). D must be a power of two >= 16. Expand GQA
            K/V to H heads before calling.
        block_lists: from ``build_block_lists`` with NB = ceil(N / route_block).
        route_block: tokens per routing block (power of two >= 16).
        is_causal: token-level causal masking (lists should also be built causal
            so blocks after the diagonal are never visited).
        block_m, block_n: fix tile sizes (bypasses autotuning). Must divide
            route_block.

    Returns:
        out: (Z, H, N, D), same dtype as q.
    """
    if not HAS_TRITON:
        raise RuntimeError("block_list_attention requires Triton")
    if not (q.shape == k.shape == v.shape) or q.dim() != 4:
        raise ValueError(f"q, k, v must share shape (Z, H, N, D); got {q.shape}, {k.shape}, {v.shape}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must share dtype")
    Z, H, N, D = q.shape
    if D < 16 or D & (D - 1):
        raise ValueError(f"head_dim must be a power of two >= 16, got {D}")
    if route_block < 16 or route_block & (route_block - 1):
        raise ValueError(f"route_block must be a power of two >= 16, got {route_block}")

    lists, counts, max_count = block_lists
    nb = triton.cdiv(N, route_block)
    if tuple(counts.shape) != (Z, H, nb) or tuple(lists.shape[:3]) != (Z, H, nb):
        raise ValueError(f"block_lists built for {tuple(counts.shape)}, expected {(Z, H, nb)} "
                         f"(N={N}, route_block={route_block})")
    if lists.shape[-1] != max_count:
        raise ValueError("lists.shape[-1] must equal max_count")
    lists = lists.to(torch.int32).contiguous()
    counts = counts.to(torch.int32).contiguous()

    out = torch.empty_like(q)
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    args = (
        q, k, v, out, lists, counts, scale,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H, N, nb, max_count, _next_pow2(max_count),
    )
    meta = dict(HEAD_DIM=D, ROUTE_BLOCK=route_block, IS_CAUSAL=bool(is_causal))

    use_autotune = block_m is None and block_n is None and q.is_cuda and not _INTERPRET
    if use_autotune:
        grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), Z * H)  # noqa: E731
        _block_list_fwd_tuned[grid](*args, **meta)
    else:
        bm = block_m or min(64, route_block)
        bn = block_n or min(64, route_block)
        if route_block % bm or route_block % bn:
            raise ValueError(f"block_m ({bm}) and block_n ({bn}) must divide route_block ({route_block})")
        launch = dict(BLOCK_M=bm, BLOCK_N=bn, **meta)
        if not _INTERPRET:
            launch.update(num_warps=num_warps, num_stages=num_stages)
        _block_list_fwd_kernel[(triton.cdiv(N, bm), Z * H)](*args, **launch)
    return out


def routed_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    router_indices: torch.Tensor,
    req_depth: DepthLike,
    *,
    route_block: int = 128,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Convenience wrapper: build the block lists, then run the kernel."""
    bl = build_block_lists(router_indices, req_depth, is_causal=is_causal,
                           sink_block=sink_block, local_blocks=local_blocks)
    return block_list_attention(q, k, v, bl, route_block=route_block,
                                is_causal=is_causal, sm_scale=sm_scale)
