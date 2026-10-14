import math
import jax
import jax.numpy as jnp
import pytest

from src.ultrametric_jax.topology import DynamicTopologyRouter
from src.ultrametric_jax.layer import UltrametricAttention
from src.ultrametric_jax.model import UltrametricTransformerBlock
from src.ultrametric_jax.kernel import (
    bruhat_tits_matmul,
    build_block_lists_jax,
    pallas_block_list_attention,
)


def test_dynamic_topology_router():
    key = jax.random.PRNGKey(0)
    batch, seq_len, embed_dim = 2, 16, 64
    num_heads, p = 4, 2
    expected_levels = int(math.ceil(math.log(max(seq_len, 2), p)))

    router = DynamicTopologyRouter(embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads, p=p)
    x = jax.random.normal(key, (batch, seq_len, embed_dim))

    variables = router.init(key, x, prng_key=key)
    assignments, aux_loss = router.apply(variables, x, prng_key=key)

    assert assignments.shape == (batch, num_heads, seq_len, expected_levels, p)
    assert aux_loss.shape == ()


def test_transformer_block_compilation():
    key = jax.random.PRNGKey(1)
    batch, seq_len, embed_dim = 2, 16, 64
    num_heads = 4

    block = UltrametricTransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
    x = jax.random.normal(key, (batch, seq_len, embed_dim))

    @jax.jit
    def init_and_apply(k, inputs):
        var = block.init(k, inputs, deterministic=True, use_interior=True)
        return block.apply(var, inputs, deterministic=True, use_interior=True)

    out = init_and_apply(key, x)
    assert out.shape == (batch, seq_len, embed_dim)


def test_bruhat_tits_matmul_pallas():
    key = jax.random.PRNGKey(2)
    num_blocks, block_size, hidden_dim, out_dim = 4, 32, 16, 16
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (num_blocks * block_size, hidden_dim))
    weight = jax.random.normal(k2, (num_blocks, num_blocks, hidden_dim, out_dim))
    routing_vectors = jnp.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32
    )
    req_depth = jnp.array(1, dtype=jnp.int32)

    out = bruhat_tits_matmul(
        x, weight, routing_vectors, req_depth, block_size=block_size, interpret=True
    )
    assert out.shape == (num_blocks, num_blocks, block_size, out_dim)
    # Blocks 0 and 2 differ at level 0, so out[0, 2] must be all zeros
    assert jnp.all(out[0, 2] == 0.0)
    # Blocks 0 and 1 agree at level 0, so out[0, 1] equals x[0] @ weight[0, 1]
    expected_01 = jnp.dot(x[:block_size], weight[0, 1])
    assert jnp.allclose(out[0, 1], expected_01, atol=1e-4)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("sink_block,local_blocks", [(False, 0), (True, 0), (False, 1), (True, 2)])
def test_build_block_lists_jax_matches_reference(is_causal, sink_block, local_blocks):
    key = jax.random.PRNGKey(7)
    Z, H, NB, TD = 2, 3, 6, 4
    router = jax.random.randint(key, (Z, H, NB, TD), 0, 2, dtype=jnp.int32)
    req_depth = jnp.array([0, 2, 4], dtype=jnp.int32)

    lists, counts = build_block_lists_jax(
        router,
        req_depth,
        is_causal=is_causal,
        sink_block=sink_block,
        local_blocks=local_blocks,
    )
    assert lists.shape == (Z, H, NB, NB) and lists.dtype == jnp.int32
    assert counts.shape == (Z, H, NB) and counts.dtype == jnp.int32

    for z in range(Z):
        for h in range(H):
            d = int(req_depth[h])
            for qb in range(NB):
                expected = []
                for kb in range(NB):
                    match = True if d == 0 else bool(jnp.all(router[z, h, qb, :d] == router[z, h, kb, :d]))
                    if local_blocks > 0 and abs(qb - kb) <= local_blocks:
                        match = True
                    if sink_block and kb == 0:
                        match = True
                    if is_causal and qb < kb:
                        match = False
                    if match:
                        expected.append(kb)
                c = int(counts[z, h, qb])
                assert c == len(expected)
                assert lists[z, h, qb, :c].tolist() == expected


@pytest.mark.parametrize("N,block_size,depth,is_causal,sink,local", [
    (128, 32, 1, False, False, 0),
    (100, 32, 2, True, False, 0),     # partial tail block + causal
    (128, 32, 2, True, True, 0),      # causal + sink block
    (96, 32, 1, False, False, 1),     # local window
    (64, 32, 0, True, False, 0),      # dense causal
])
def test_pallas_block_list_attention_matches_reference(N, block_size, depth, is_causal, sink, local):
    key = jax.random.PRNGKey(N + depth)
    Z, H, D = 1, 2, 16
    NB = math.ceil(N / block_size)
    kq, kk, kv, kr = jax.random.split(key, 4)

    q = jax.random.normal(kq, (Z, H, N, D), dtype=jnp.float32)
    k = jax.random.normal(kk, (Z, H, N, D), dtype=jnp.float32)
    v = jax.random.normal(kv, (Z, H, N, D), dtype=jnp.float32)
    router = jax.random.randint(kr, (Z, H, NB, 3), 0, 2, dtype=jnp.int32)

    out = pallas_block_list_attention(
        q, k, v, router, depth,
        block_size=block_size,
        is_causal=is_causal,
        sink_block=sink,
        local_blocks=local,
        interpret=True,
    )

    # Build reference token-level mask from build_block_lists_jax
    lists, counts = build_block_lists_jax(
        router, depth, is_causal=is_causal, sink_block=sink, local_blocks=local
    )
    block_mask = jnp.zeros((Z, H, NB, NB), dtype=bool)
    for z in range(Z):
        for h in range(H):
            for qb in range(NB):
                c = int(counts[z, h, qb])
                for t in range(c):
                    kb = int(lists[z, h, qb, t])
                    block_mask = block_mask.at[z, h, qb, kb].set(True)

    token_mask = jnp.repeat(jnp.repeat(block_mask, block_size, axis=-2), block_size, axis=-1)[..., :N, :N]
    if is_causal:
        token_mask = token_mask & jnp.tril(jnp.ones((N, N), dtype=bool))

    scale = 1.0 / math.sqrt(D)
    scores = jnp.einsum("zhqd,zhkd->zhqk", q, k, precision=jax.lax.Precision.HIGHEST) * scale
    scores = jnp.where(token_mask, scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    ref = jnp.einsum("zhqk,zhkd->zhqd", attn, v, precision=jax.lax.Precision.HIGHEST)

    assert out.shape == (Z, H, N, D)
    assert jnp.allclose(out, ref, atol=1e-4)
