import functools
import math
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


import re


def _call_with_mosaic_version_fallback(fn, *args, **kwargs):
    """Executes a Pallas TPU call, automatically adjusting `tpu_custom_call._FWD_COMPAT_VERSION`
    if the Colab TPU runtime's installed `libtpu` expects an older Mosaic serialization version
    than `jaxlib` emits by default (e.g. `Unsupported version: expected <= 7 but got 8`).
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e)
        m = re.search(r"Unsupported version:\s*expected\s*<=\s*(\d+)\s*but\s*got\s*(\d+)", msg)
        if m is not None:
            target_ver = int(m.group(1))
            try:
                import jax._src.tpu_custom_call as tcc
                tcc._FWD_COMPAT_VERSION = target_ver
                tcc.ir_version_override = lambda: target_ver
            except Exception:
                pass
            if hasattr(fn, "clear_cache"):
                fn.clear_cache()
            return fn(*args, **kwargs)
        raise


def _bruhat_tits_kernel(
    routing_vectors,
    req_depth,
    x_ref,
    weight_ref,
    out_ref,
):
    """
    Pallas kernel for nested hierarchical block-sparsity (Bruhat-Tits).
    Dynamically skips memory loads for blocks that don't share phylogenetic ancestor depth.
    """
    i = pl.program_id(0)
    j = pl.program_id(1)

    vec_i = routing_vectors[i]
    vec_j = routing_vectors[j]

    depth = req_depth[0]
    tree_depth = routing_vectors.shape[1]

    iota = jax.lax.iota(jnp.int32, tree_depth)
    mask = iota < depth

    match = jnp.all(jnp.logical_or(~mask, vec_i == vec_j))

    def _do_compute():
        x_val = x_ref[...]
        w_val = weight_ref[...]
        # x_val: (block_size, hidden_dim)
        # w_val: (hidden_dim, out_dim)
        # out_ref: (block_size, out_dim)
        out_ref[...] = jnp.dot(x_val, w_val)

    def _do_zero():
        out_ref[...] = jnp.zeros_like(out_ref[...])

    jax.lax.cond(match, _do_compute, _do_zero)


def lhs_index_map(i, j, routing_vectors, req_depth):
    depth = req_depth[0]
    tree_depth = routing_vectors.shape[1]
    iota = jax.lax.iota(jnp.int32, tree_depth)
    mask = iota < depth
    match = jnp.all(jnp.logical_or(~mask, routing_vectors[i] == routing_vectors[j]))

    safe_i = jax.lax.select(match, i, 0)
    return (safe_i, 0)


def rhs_index_map(i, j, routing_vectors, req_depth):
    depth = req_depth[0]
    tree_depth = routing_vectors.shape[1]
    iota = jax.lax.iota(jnp.int32, tree_depth)
    mask = iota < depth
    match = jnp.all(jnp.logical_or(~mask, routing_vectors[i] == routing_vectors[j]))

    safe_i = jax.lax.select(match, i, 0)
    safe_j = jax.lax.select(match, j, 0)
    return (safe_i, safe_j, 0, 0)


def out_index_map(i, j, routing_vectors, req_depth):
    # Output mapped unconditionally so we can initialize zeros
    return (i, j, 0, 0)


@functools.partial(jax.jit, static_argnames=['block_size', 'interpret'])
def _bruhat_tits_matmul_jit(x, weight, routing_vectors, req_depth, block_size=128, interpret=False):
    num_blocks = routing_vectors.shape[0]
    hidden_dim = x.shape[1]
    out_dim = weight.shape[3]

    if req_depth.shape == ():
        req_depth = req_depth[None]

    x_spec = pl.BlockSpec(
        (block_size, hidden_dim),
        lhs_index_map
    )

    weight_spec = pl.BlockSpec(
        (None, None, hidden_dim, out_dim),
        rhs_index_map
    )

    out_spec = pl.BlockSpec(
        (None, None, block_size, out_dim),
        out_index_map
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[x_spec, weight_spec],
        out_specs=out_spec,
        grid=(num_blocks, num_blocks),
    )

    out_shape = jax.ShapeDtypeStruct(
        (num_blocks, num_blocks, block_size, out_dim),
        weight.dtype
    )

    return pl.pallas_call(
        _bruhat_tits_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
    )(routing_vectors, req_depth, x, weight)


def bruhat_tits_matmul(x, weight, routing_vectors, req_depth, block_size=128, interpret=False):
    """
    x: (num_blocks * block_size, hidden_dim)
    weight: (num_blocks, num_blocks, hidden_dim, out_dim)
    routing_vectors: (num_blocks, tree_depth)
    req_depth: scalar integer specifying phylogenetic ancestor depth required.
    """
    return _call_with_mosaic_version_fallback(
        _bruhat_tits_matmul_jit, x, weight, routing_vectors, req_depth,
        block_size=block_size, interpret=interpret
    )


# ============================================================================
# Block-list sparse attention in JAX / Pallas (Milestone 4)
# ============================================================================

def build_block_lists_jax(
    router_indices: jnp.ndarray,
    req_depth: Union[int, Sequence[int], jnp.ndarray],
    *,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build per-row active key-block lists from per-block routing vectors in pure JAX.

    Args:
        router_indices: (Z, H, NB, TD) integer branch ids (>= 0).
        req_depth: required matching depth; scalar int or per-head (H,).
            0 means every block matches (dense).
        is_causal: drop key blocks after the query block.
        sink_block: always include key block 0 (attention sink).
        local_blocks: always include key blocks with |i - j| <= local_blocks.

    Returns:
        lists: (Z, H, NB, NB) int32. Row (z, h, i) holds the active key blocks
            for query block i in ascending order in the first ``counts[z, h, i]``
            slots; padded with 0s thereafter.
        counts: (Z, H, NB) int32, number of active key blocks per row.
    """
    if router_indices.ndim != 4:
        raise ValueError(f"router_indices must be (Z, H, NB, TD), got {router_indices.shape}")
    Z, H, NB, TD = router_indices.shape
    r = router_indices.astype(jnp.int32)

    depth_arr = jnp.asarray(req_depth, dtype=jnp.int32)
    if depth_arr.ndim == 0:
        depth_arr = jnp.broadcast_to(depth_arr, (H,))
    else:
        depth_arr = depth_arr.reshape(-1)
        if depth_arr.shape[0] != H:
            raise ValueError(f"req_depth has {depth_arr.shape[0]} entries, expected {H}")

    levels = jnp.arange(TD, dtype=jnp.int32)[None, None, None, :]  # (1, 1, 1, TD)
    keep = levels < depth_arr[None, :, None, None]                 # (1, H, 1, TD)

    eq = (r[:, :, :, None, :] == r[:, :, None, :, :]) | (~keep[:, :, None, :, :])
    match = jnp.all(eq, axis=-1)                                   # (Z, H, NB, NB)

    idx = jnp.arange(NB, dtype=jnp.int32)
    if local_blocks > 0:
        local_mask = jnp.abs(idx[:, None] - idx[None, :]) <= local_blocks
        match = match | local_mask[None, None, :, :]
    if sink_block and NB > 0:
        sink_mask = (idx == 0)[None, None, None, :]
        match = match | sink_mask
    if is_causal:
        causal_mask = idx[:, None] >= idx[None, :]
        match = match & causal_mask[None, None, :, :]

    counts = jnp.sum(match.astype(jnp.int32), axis=-1)             # (Z, H, NB)

    # Stable ascending compaction without dynamic shapes: sort (~match, j) along last axis.
    # Or scatter into (Z, H, NB, NB + 1) using cumsum positions and slice to NB.
    pos = jnp.cumsum(match.astype(jnp.int32), axis=-1) - 1
    pos = jnp.where(match, pos, NB)
    # Using argsort on (~match) preserves ascending key index order because stable=True is default
    order = jnp.argsort(~match, axis=-1, stable=True)
    valid_slot = jnp.arange(NB, dtype=jnp.int32)[None, None, None, :] < counts[..., None]
    lists = jnp.where(valid_slot, order.astype(jnp.int32), jnp.int32(0))
    return lists, counts


def _pallas_tpu_block_list_attn_kernel(
    lists_ref,      # SMEM: (ZH, nb, nb) int32
    counts_ref,     # SMEM: (ZH, nb) int32
    q_ref,          # VMEM: (block_size, D)
    k_ref,          # VMEM: (block_size, D) for current active step
    v_ref,          # VMEM: (block_size, D) for current active step
    out_ref,        # VMEM: (block_size, D)
    m_scratch_ref,  # VMEM scratch: (block_size, 1) float32
    l_scratch_ref,  # VMEM scratch: (block_size, 1) float32
    acc_scratch_ref,# VMEM scratch: (block_size, D) float32
    *,
    seq_len: int,
    block_size: int,
    is_causal: bool,
    sm_scale: float,
    max_steps: int,
):
    zh = pl.program_id(0)
    qb = pl.program_id(1)
    step = pl.program_id(2)

    count = counts_ref[zh, qb]
    safe_step = jnp.minimum(step, jnp.maximum(count - 1, 0))
    kb = lists_ref[zh, qb, safe_step]

    offs_m = qb * block_size + jnp.arange(block_size, dtype=jnp.int32)
    m_valid = offs_m < seq_len

    @pl.when(step == 0)
    def _init():
        m_scratch_ref[...] = jnp.full((block_size, 1), -jnp.inf, dtype=jnp.float32)
        l_scratch_ref[...] = jnp.zeros((block_size, 1), dtype=jnp.float32)
        acc_scratch_ref[...] = jnp.zeros((block_size, q_ref.shape[-1]), dtype=jnp.float32)

    @pl.when(step < count)
    def _step():
        q = q_ref[...].astype(jnp.float32)
        k_tile = k_ref[...].astype(jnp.float32)
        v_tile = v_ref[...].astype(jnp.float32)

        cols = kb * block_size + jnp.arange(block_size, dtype=jnp.int32)
        n_valid = cols < seq_len
        keep = m_valid[:, None] & n_valid[None, :]
        if is_causal:
            keep = keep & (offs_m[:, None] >= cols[None, :])

        qk = jnp.dot(q, k_tile.T, precision=jax.lax.Precision.HIGHEST) * sm_scale
        qk = jnp.where(keep, qk, -jnp.inf)

        m_prev = m_scratch_ref[...]
        l_prev = l_scratch_ref[...]
        acc_prev = acc_scratch_ref[...]

        m_new = jnp.maximum(m_prev, jnp.max(qk, axis=-1, keepdims=True))
        m_ref = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
        alpha = jnp.exp(m_prev - m_ref)
        p = jnp.where(keep, jnp.exp(qk - m_ref), 0.0)

        l_scratch_ref[...] = l_prev * alpha + jnp.sum(p, axis=-1, keepdims=True)
        acc_scratch_ref[...] = acc_prev * alpha + jnp.dot(
            p, v_tile, precision=jax.lax.Precision.HIGHEST
        )
        m_scratch_ref[...] = m_new

    @pl.when(step == max_steps - 1)
    def _epilogue():
        l_final = l_scratch_ref[...]
        l_safe = jnp.where(l_final == 0.0, 1.0, l_final)
        out = (acc_scratch_ref[...] / l_safe) * m_valid[:, None].astype(jnp.float32)
        out_ref[...] = out.astype(out_ref.dtype)


def _pallas_block_list_attn_kernel(
    q_ref,
    k_ref,
    v_ref,
    lists_ref,
    counts_ref,
    q_idx_ref,
    out_ref,
    *,
    seq_len: int,
    block_size: int,
    is_causal: bool,
    sm_scale: float,
):
    q = q_ref[0].astype(jnp.float32)               # (block_size, D)
    qb = q_idx_ref[0, 0]                           # scalar int32
    count = counts_ref[0, 0]                       # scalar int32
    offs_m = qb * block_size + jnp.arange(block_size, dtype=jnp.int32)
    m_valid = offs_m < seq_len

    m_i = jnp.full((block_size,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_size,), dtype=jnp.float32)
    acc = jnp.zeros_like(q, dtype=jnp.float32)

    def body_fn(t, carry):
        m_prev, l_prev, acc_prev = carry
        kb = lists_ref[0, t]
        k_tile = k_ref[0, kb].astype(jnp.float32)  # (block_size, D)
        v_tile = v_ref[0, kb].astype(jnp.float32)  # (block_size, D)

        cols = kb * block_size + jnp.arange(block_size, dtype=jnp.int32)
        n_valid = cols < seq_len
        keep = m_valid[:, None] & n_valid[None, :]
        if is_causal:
            keep = keep & (offs_m[:, None] >= cols[None, :])

        qk = jnp.dot(q, k_tile.T, precision=jax.lax.Precision.HIGHEST) * sm_scale
        qk = jnp.where(keep, qk, -jnp.inf)

        m_new = jnp.maximum(m_prev, jnp.max(qk, axis=-1))
        m_ref = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
        alpha = jnp.exp(m_prev - m_ref)
        p = jnp.where(keep, jnp.exp(qk - m_ref[:, None]), 0.0)

        l_new = l_prev * alpha + jnp.sum(p, axis=-1)
        acc_new = acc_prev * alpha[:, None] + jnp.dot(
            p, v_tile, precision=jax.lax.Precision.HIGHEST
        )
        return (m_new, l_new, acc_new)

    m_i, l_i, acc = jax.lax.fori_loop(0, count, body_fn, (m_i, l_i, acc))
    l_safe = jnp.where(l_i == 0.0, 1.0, l_i)
    out = (acc / l_safe[:, None]) * m_valid[:, None].astype(jnp.float32)
    out_ref[0] = out.astype(out_ref.dtype)


def pallas_block_list_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    router_indices: jnp.ndarray,
    req_depth: Union[int, Sequence[int], jnp.ndarray],
    *,
    block_size: int = 128,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
    sm_scale: Optional[float] = None,
    interpret: bool = False,
) -> jnp.ndarray:
    """Block-list sparse attention using a Pallas kernel with online softmax.

    On Cloud TPU (`jax.default_backend() == "tpu"` and `not interpret`), uses
    `pltpu.PrefetchScalarGridSpec` to prefetch `(lists, counts)` into SMEM and
    dynamically DMA only the active `max_active` key/value blocks `(block_size, D)`
    into VMEM (never loading the full sequence into VMEM).

    Args:
        q, k, v: (Z, H, N, D) query, key, value tensors.
        router_indices: (Z, H, NB, TD) integer routing vectors with NB = ceil(N / block_size).
        req_depth: required ancestor matching depth (int or per-head (H,)).
        block_size: tokens per routing block (default 128).
        is_causal: whether to apply causal masking (both block-level and token-level).
        sink_block: whether to always attend to key block 0.
        local_blocks: local sliding window radius in route blocks.
        sm_scale: softmax scale (defaults to 1 / sqrt(D)).
        interpret: run Pallas in CPU interpret mode.

    Returns:
        out: (Z, H, N, D) attention output, same dtype as q.
    """
    if q.ndim != 4 or not (q.shape == k.shape == v.shape):
        raise ValueError(f"q, k, v must share shape (Z, H, N, D), got {q.shape}, {k.shape}, {v.shape}")
    Z, H, N, D = q.shape
    nb = math.ceil(N / block_size)
    if router_indices.shape[:3] != (Z, H, nb):
        raise ValueError(
            f"router_indices shape {router_indices.shape[:3]} does not match (Z={Z}, H={H}, NB={nb})"
        )

    scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D)
    lists, counts = build_block_lists_jax(
        router_indices,
        req_depth,
        is_causal=is_causal,
        sink_block=sink_block,
        local_blocks=local_blocks,
    )

    pad_len = nb * block_size - N
    if pad_len > 0:
        q_pad = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k_pad = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v_pad = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
    else:
        q_pad, k_pad, v_pad = q, k, v

    ZH = Z * H
    q_blocks = q_pad.reshape(ZH, nb, block_size, D)
    k_blocks = k_pad.reshape(ZH, nb, block_size, D)
    v_blocks = v_pad.reshape(ZH, nb, block_size, D)
    lists_flat = lists.reshape(ZH, nb, nb)

    use_interpret = interpret or (jax.default_backend() != "tpu")
    if not use_interpret:
        counts_2d = counts.reshape(ZH, nb)
        max_steps = max(1, int(jnp.max(counts_2d)))

        def kv_index_map(zh, qb, step, lists_ref, counts_ref):
            cnt = counts_ref[zh, qb]
            safe_step = jnp.minimum(step, jnp.maximum(cnt - 1, 0))
            kb = lists_ref[zh, qb, safe_step]
            return (zh, kb, 0, 0)

        tpu_kernel = functools.partial(
            _pallas_tpu_block_list_attn_kernel,
            seq_len=N,
            block_size=block_size,
            is_causal=bool(is_causal),
            sm_scale=scale,
            max_steps=max_steps,
        )
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec((None, None, block_size, D), lambda zh, qb, step, l_ref, c_ref: (zh, qb, 0, 0)),
                pl.BlockSpec((None, None, block_size, D), kv_index_map),
                pl.BlockSpec((None, None, block_size, D), kv_index_map),
            ],
            out_specs=pl.BlockSpec((None, None, block_size, D), lambda zh, qb, step, l_ref, c_ref: (zh, qb, 0, 0)),
            scratch_shapes=[
                pltpu.VMEM((block_size, 1), jnp.float32),
                pltpu.VMEM((block_size, 1), jnp.float32),
                pltpu.VMEM((block_size, D), jnp.float32),
            ],
            grid=(ZH, nb, max_steps),
        )
        pallas_fn = pl.pallas_call(
            tpu_kernel,
            out_shape=jax.ShapeDtypeStruct((ZH, nb, block_size, D), q.dtype),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary")
            ),
        )
        out_blocks = _call_with_mosaic_version_fallback(
            pallas_fn, lists_flat, counts_2d, q_blocks, k_blocks, v_blocks
        )
    else:
        counts_flat = counts.reshape(ZH, nb, 1)
        q_idx = jnp.broadcast_to(jnp.arange(nb, dtype=jnp.int32)[None, :, None], (ZH, nb, 1))

        kernel = functools.partial(
            _pallas_block_list_attn_kernel,
            seq_len=N,
            block_size=block_size,
            is_causal=bool(is_causal),
            sm_scale=scale,
        )

        out_blocks = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((ZH, nb, block_size, D), q.dtype),
            in_specs=[
                pl.BlockSpec((None, 1, block_size, D), lambda zh, qb: (zh, qb, 0, 0)),
                pl.BlockSpec((1, nb, block_size, D), lambda zh, qb: (zh, 0, 0, 0)),
                pl.BlockSpec((1, nb, block_size, D), lambda zh, qb: (zh, 0, 0, 0)),
                pl.BlockSpec((None, 1, nb), lambda zh, qb: (zh, qb, 0)),
                pl.BlockSpec((None, 1, 1), lambda zh, qb: (zh, qb, 0)),
                pl.BlockSpec((None, 1, 1), lambda zh, qb: (zh, qb, 0)),
            ],
            out_specs=pl.BlockSpec((None, 1, block_size, D), lambda zh, qb: (zh, qb, 0, 0)),
            grid=(ZH, nb),
            interpret=True,
        )(q_blocks, k_blocks, v_blocks, lists_flat, counts_flat, q_idx)

    out_padded = out_blocks.reshape(Z, H, nb * block_size, D)
    return out_padded[:, :, :N, :]
