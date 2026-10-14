import functools
import math
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


import re


def _apply_mosaic_target_version_patch(target_ver: int) -> None:
    """Patches `jax._src.tpu_custom_call` so `mosaic-serde` always serializes with `target-version={target_ver}`."""
    try:
        import jax._src.tpu_custom_call as tcc

        tcc._FWD_COMPAT_VERSION = target_ver
        tcc.ir_version_override = lambda: target_ver
        tcc.get_ir_version = lambda *a, **kw: target_ver

        orig_pm = getattr(tcc, "_orig_PassManager", tcc.PassManager)
        tcc._orig_PassManager = orig_pm

        class _CompatPassManager:
            @staticmethod
            def parse(pipeline_str: str, *args, **kwargs):
                if "mosaic-serde" in pipeline_str and "serialize=true" in pipeline_str:
                    if "target-version=" in pipeline_str:
                        pipeline_str = re.sub(
                            r"target-version=\d+",
                            f"target-version={target_ver}",
                            pipeline_str,
                        )
                    else:
                        pipeline_str = pipeline_str.replace(
                            "serialize=true",
                            f"serialize=true target-version={target_ver}",
                        )
                return orig_pm.parse(pipeline_str, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(orig_pm, name)

        tcc.PassManager = _CompatPassManager

        if hasattr(tcc, "_lower_mosaic_module_to_asm"):
            orig_lower = getattr(tcc, "_orig_lower_mosaic_module_to_asm", tcc._lower_mosaic_module_to_asm)
            tcc._orig_lower_mosaic_module_to_asm = orig_lower

            def _patched_lower_mosaic_module_to_asm(module, *args, **kwargs):
                if "ir_version" in kwargs or len(args) == 0:
                    try:
                        kwargs["ir_version"] = target_ver
                        return orig_lower(module, *args, **kwargs)
                    except TypeError:
                        kwargs.pop("ir_version", None)
                return orig_lower(module, *args, **kwargs)

            tcc._lower_mosaic_module_to_asm = _patched_lower_mosaic_module_to_asm
    except Exception:
        pass


def _call_with_mosaic_version_fallback(fn, *args, **kwargs):
    """Executes a Pallas TPU call, automatically adjusting `mosaic-serde` `target-version`
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
            _apply_mosaic_target_version_patch(target_ver)
            if hasattr(fn, "clear_cache"):
                fn.clear_cache()
            try:
                jax.clear_caches()
            except Exception:
                pass
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


_TPU_NUM_LANES = 128
_TPU_SMEM_BUDGET_BYTES = 512 * 1024  # Safe 512 KB budget inside 1 MB SMEM


def _tile_lanes_to_dim(x_lanes: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    """Broadcasts/slices a `(block_size, 128)` lane-replicated tensor to `(block_size, target_dim)`."""
    repeats, rem = divmod(target_dim, _TPU_NUM_LANES)
    if rem == 0:
        return jnp.tile(x_lanes, (1, repeats)) if repeats > 1 else x_lanes
    if repeats == 0:
        return x_lanes[:, :target_dim]
    return jnp.tile(x_lanes, (1, repeats + 1))[:, :target_dim]


def _pallas_tpu_block_list_attn_kernel(
    lists_ref,      # SMEM: (ZH * nb, max_steps) int32
    counts_ref,     # SMEM: (ZH * nb,) int32
    q_ref,          # VMEM: (block_size, D)
    k_ref,          # VMEM: (block_size, D) for current active step
    v_ref,          # VMEM: (block_size, D) for current active step
    out_ref,        # VMEM: (block_size, D)
    m_scratch_ref,  # VMEM scratch: (block_size, 128) float32
    l_scratch_ref,  # VMEM scratch: (block_size, 128) float32
    acc_scratch_ref,# VMEM scratch: (block_size, D) float32
    *,
    nb: int,
    seq_len: int,
    block_size: int,
    is_causal: bool,
    sm_scale: float,
    max_steps: int,
):
    zh = pl.program_id(0)
    qb = pl.program_id(1)
    step = pl.program_id(2)

    row_idx = zh * nb + qb
    count = counts_ref[row_idx]
    safe_step = jnp.minimum(step, jnp.maximum(count - 1, 0))
    kb = lists_ref[row_idx, safe_step]

    mask_shape = (block_size, block_size)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + qb * block_size

    @pl.when(step == 0)
    def _init():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        acc_scratch_ref[...] = jnp.zeros_like(acc_scratch_ref)

    @pl.when(step < count)
    def _step():
        in_dtype = q_ref.dtype
        dot_prec = (
            jax.lax.Precision.HIGHEST
            if in_dtype == jnp.float32
            else jax.lax.Precision.DEFAULT
        )
        q = q_ref[...].astype(jnp.float32) if in_dtype == jnp.float32 else q_ref[...]
        k_tile = k_ref[...].astype(jnp.float32) if in_dtype == jnp.float32 else k_ref[...]
        v_tile = v_ref[...].astype(jnp.float32) if in_dtype == jnp.float32 else v_ref[...]

        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + kb * block_size
        keep = jnp.logical_and(row_ids < seq_len, col_ids < seq_len)
        if is_causal:
            keep = jnp.logical_and(keep, row_ids >= col_ids)

        qk = (
            jax.lax.dot_general(
                q,
                k_tile,
                (((1,), (1,)), ((), ())),
                precision=dot_prec,
                preferred_element_type=jnp.float32,
            )
            * sm_scale
        )
        qk = jnp.where(keep, qk, -jnp.inf)

        m_prev = m_scratch_ref[...]
        l_prev = l_scratch_ref[...]
        acc_prev = acc_scratch_ref[...]

        m_curr = jnp.max(qk, axis=1)[:, None]
        m_new = jnp.maximum(m_prev, m_curr)
        m_ref = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
        alpha = jnp.exp(m_prev - m_ref)

        m_ref_bcast = _tile_lanes_to_dim(m_ref, block_size)
        p = jnp.where(keep, jnp.exp(qk - m_ref_bcast), 0.0)

        l_curr = jax.lax.broadcast_in_dim(jnp.sum(p, axis=1), l_prev.shape, (0,))
        l_scratch_ref[...] = l_prev * alpha + l_curr

        d_model = q.shape[-1]
        alpha_o = _tile_lanes_to_dim(alpha, d_model)
        p_dot = p if in_dtype == jnp.float32 else p.astype(v_tile.dtype)
        o_curr = jax.lax.dot(
            p_dot,
            v_tile,
            precision=dot_prec,
            preferred_element_type=jnp.float32,
        )
        acc_scratch_ref[...] = acc_prev * alpha_o + o_curr
        m_scratch_ref[...] = m_new

    @pl.when(step == max_steps - 1)
    def _epilogue():
        l_final = l_scratch_ref[...]
        l_safe = jnp.where(l_final == 0.0, 1.0, l_final)
        d_model = out_ref.shape[-1]
        l_bcast = _tile_lanes_to_dim(l_safe, d_model)
        row_valid_d = (
            jax.lax.broadcasted_iota(jnp.int32, (block_size, d_model), 0) + qb * block_size
        ) < seq_len
        out = jnp.where(row_valid_d, acc_scratch_ref[...] / l_bcast, 0.0)
        out_ref[...] = out.astype(out_ref.dtype)


@functools.lru_cache(maxsize=64)
def _get_cached_tpu_pallas_fn(
    zh_chunk: int,
    nb: int,
    block_size: int,
    D: int,
    seq_len: int,
    is_causal: bool,
    scale: float,
    max_steps: int,
    q_dtype,
):
    def kv_index_map(zh, qb, step, lists_ref, counts_ref):
        row_idx = zh * nb + qb
        cnt = counts_ref[row_idx]
        safe_step = jnp.minimum(step, jnp.maximum(cnt - 1, 0))
        kb = lists_ref[row_idx, safe_step]
        return (zh, kb, 0, 0)

    tpu_kernel = functools.partial(
        _pallas_tpu_block_list_attn_kernel,
        nb=nb,
        seq_len=seq_len,
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
            pltpu.VMEM((block_size, _TPU_NUM_LANES), jnp.float32),
            pltpu.VMEM((block_size, _TPU_NUM_LANES), jnp.float32),
            pltpu.VMEM((block_size, D), jnp.float32),
        ],
        grid=(zh_chunk, nb, max_steps),
    )
    return pl.pallas_call(
        tpu_kernel,
        out_shape=jax.ShapeDtypeStruct((zh_chunk, nb, block_size, D), q_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )


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
    precomputed_lists_counts: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    max_steps_static: Optional[int] = None,
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
        precomputed_lists_counts: optional `(lists, counts)` tuple from `build_block_lists_jax`
            to avoid recomputing block lists inside benchmark loops.
        max_steps_static: optional integer upper bound on active blocks per query block
            to avoid host-device sync (`int(jnp.max(counts))`) during execution.

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
    if precomputed_lists_counts is not None:
        lists, counts = precomputed_lists_counts
    else:
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
        if max_steps_static is not None:
            max_steps = max(1, min(nb, int(max_steps_static)))
        else:
            max_steps = max(1, int(jnp.max(counts_2d)))

        lists_trimmed = lists_flat[:, :, :max_steps]

        # On TPU SMEM, 2D arrays are padded to multiples of (8, 128).
        # Flattening (zh_chunk, nb) -> (zh_chunk * nb,) ensures only `max_steps` is padded to 128.
        padded_cols = math.ceil(max_steps / 128) * 128
        padded_bytes_per_head = nb * (padded_cols + 1) * 4
        max_heads_per_call = max(1, _TPU_SMEM_BUDGET_BYTES // max(1, padded_bytes_per_head))
        zh_chunk = ZH
        while zh_chunk > max_heads_per_call or (ZH % zh_chunk != 0):
            zh_chunk -= 1
            if zh_chunk <= 1:
                zh_chunk = 1
                break

        pallas_fn = _get_cached_tpu_pallas_fn(
            zh_chunk,
            nb,
            block_size,
            D,
            N,
            bool(is_causal),
            scale,
            max_steps,
            q.dtype,
        )
        if zh_chunk == ZH:
            out_blocks = _call_with_mosaic_version_fallback(
                pallas_fn,
                lists_trimmed.reshape(ZH * nb, max_steps),
                counts_2d.reshape(ZH * nb),
                q_blocks,
                k_blocks,
                v_blocks,
            )
        else:
            chunks = []
            for start_zh in range(0, ZH, zh_chunk):
                end_zh = start_zh + zh_chunk
                chunks.append(
                    _call_with_mosaic_version_fallback(
                        pallas_fn,
                        lists_trimmed[start_zh:end_zh].reshape(zh_chunk * nb, max_steps),
                        counts_2d[start_zh:end_zh].reshape(zh_chunk * nb),
                        q_blocks[start_zh:end_zh],
                        k_blocks[start_zh:end_zh],
                        v_blocks[start_zh:end_zh],
                    )
                )
            out_blocks = jnp.concatenate(chunks, axis=0)
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
