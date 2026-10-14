# Cloud TPU v6e (Trillium) Experiment Report: Pallas DMA-Skipping Block-List Attention

**Date**: October 13, 2026  
**Target Module**: [`src/ultrametric_jax/kernel.py`](../src/ultrametric_jax/kernel.py) (`build_block_lists_jax`, `pallas_block_list_attention`)  
**Benchmark Notebook**: [`notebooks/colab_tpu_pallas_ultrametric.ipynb`](../notebooks/colab_tpu_pallas_ultrametric.ipynb)  
**Commit**: `33e359b`

---

## 1. Executive Summary

This report logs the hardware validation and wall-clock scaling benchmarks of the Track 4 JAX/Pallas Block-List Sparse Attention kernel (`_pallas_tpu_block_list_attn_kernel`) on **Google Cloud TPU** via Google Colab.

Whereas the initial V1 prototype (`bruhat_tits_matmul_pallas`) evaluated an $O(NB^2)$ dense grid with `jax.lax.cond` inside the kernel body, the V3+ Pallas implementation decouples sparse schedule construction from TensorCore execution:
1. **Vectorized Block-List Compaction (`build_block_lists_jax`)**: Compiles a padded index table `lists` of shape $(Z, H, NB, \text{max\_steps})$ and active block count `counts` of shape $(Z, H, NB)$ on-device via stable boolean sorting over ultrametric prefix matches, causal bounds, attention sinks, and local sliding windows.
2. **SMEM Scalar Prefetch & Dynamic VMEM DMA (`pltpu.PrefetchScalarGridSpec`)**: Prefetches the flattened `(lists, counts)` lookup tables into TPU Scalar Memory (**SMEM**) and iterates strictly over $t \in [0, \text{max\_steps})$, dynamically issuing indirect DMA transfers from HBM into Vector Memory (**VMEM**) only for active $(B \times D)$ key/value blocks.

### Key Empirical Highlights
- **Exact Numerical Parity (`5.96e-07` Max Abs Error)**: Verified across prime bases $p \in \{2, 3\}$ and required ultrametric tree depths $d \in \{1, 2, 3, 4\}$ against an exact `float32` masked attention reference.
- **3.35x–4.10x Speedup over Dense XLA Attention ($N \in [4096, 8192]$)**: Achieves `2.630 ms` vs `10.780 ms` (**4.10x**) at $N = 4096$ and `6.204 ms` vs `20.799 ms` (**3.35x**) at $N = 8192$.
- **2.92x Speedup at $N = 16,384$ & Zero-OOM Scaling to $N = 32,768$**: Executes $N = 32,768$ ($Z=1, H=16, D=128$, `bfloat16`) in **44.121 ms** at $8.47\%$ active causal density where standard Dense XLA Attention runs out of HBM (**OOM**).

---

## 2. Numerical Parity Validation (`fp32` IEEE Mode)

**Configuration**:
- Batch size $Z = 1$, Heads $H = 8$, Sequence length $N = 1024$, Head dimension $D = 128$, Block size $B = 128$ ($NB = 8$ blocks), Tree depth $TD = 5$.
- Causal masking `is_causal=True`, Attention sink `sink_block=True`, Local window `local_blocks=1`.
- MXU Precision: `jax.lax.Precision.HIGHEST` (3-pass IEEE `float32` emulation on MXU).

| Prime Arity ($p$) | Required Depth ($d$) | Active Causal Blocks (%) | Max Absolute Error vs. Dense Reference | Status |
|---:|---:|---:|---:|:---:|
| $p = 2$ | $d = 1$ | $77.43\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 2$ | $d = 2$ | $67.01\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 2$ | $d = 3$ | $62.85\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 2$ | $d = 4$ | $60.42\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 3$ | $d = 1$ | $73.26\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 3$ | $d = 2$ | $62.50\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 3$ | $d = 3$ | $60.07\%$ | $5.96 \times 10^{-7}$ | PASS |
| $p = 3$ | $d = 4$ | $58.68\%$ | $5.96 \times 10^{-7}$ | PASS |

---

## 3. Wall-Clock Latency & HBM Scaling Benchmark (`bfloat16`)

**Configuration**:
- Batch size $Z = 1$, Heads $H = 16$, Head dimension $D = 128$, Block size $B = 128$, Tree depth $TD = 6$, Required depth $d = 4$ ($p = 2$).
- Causal masking `is_causal=True`, Attention sink `sink_block=True`, Local window `local_blocks=1`.
- Dtype: `jnp.bfloat16` inputs/outputs with `float32` online softmax accumulators in VMEM (`m_scratch`, `l_scratch`, `acc_scratch`).
- Timing methodology: Warmup compilation followed by 10 timed device-synchronized iterations (`block_until_ready()`).

| Sequence Length ($N$) | Route Blocks ($NB$) | Active Causal Blocks (%) | `pallas_block_list_attention` (ms) | Dense XLA Attention (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| **2,048** | 16 | $37.50\%$ | $1.604\text{ ms}$ | $1.104\text{ ms}$ | **0.69x** |
| **4,096** | 32 | $22.88\%$ | $2.630\text{ ms}$ | $10.780\text{ ms}$ | **4.10x** |
| **8,192** | 64 | $14.68\%$ | $6.204\text{ ms}$ | $20.799\text{ ms}$ | **3.35x** |
| **16,384** | 128 | $10.52\%$ | $16.327\text{ ms}$ | $47.691\text{ ms}$ | **2.92x** |
| **32,768** | 256 | $8.47\%$ | $44.121\text{ ms}$ | **OOM** | **$\infty$** |

---

## 4. TPU Compiler & Hardware Engineering Notes

During hardware bring-up on the Colab TPU runtime, four critical Mosaic/XLA hardware constraints were identified and permanently resolved in [`src/ultrametric_jax/kernel.py`](../src/ultrametric_jax/kernel.py):

1. **Mosaic IR Serialization Version Compatibility (`_call_with_mosaic_version_fallback`)**:
   - Colab's `jaxlib` emitted `stable_mosaic.version = 8` while the runtime's `libtpu.so` supported `<= 7`.
   - Resolved by intercepting `jax._src.tpu_custom_call.PassManager.parse` to inject `target-version=7` into `mosaic-serde{serialize=true ...}` and clearing JAX's JIT compilation cache on fallback retry.
2. **32-Bit Minor-Dimension Broadcast Constraint**:
   - TPU Mosaic forbids inserting minor dimensions via `vector.shape_cast` on 1-bit boolean vectors (`vector<128xi1> -> vector<128x1xi1>`).
   - Resolved by generating 2D `int32` coordinate tiles directly via `jax.lax.broadcasted_iota(jnp.int32, (block_size, block_size), dimension=0/1)` prior to boolean comparison.
3. **128-Lane VMEM Scratch Layout Alignment**:
   - Online softmax state buffers (`m_scratch_ref`, `l_scratch_ref`) are allocated as `(block_size, 128)` (`_TPU_NUM_LANES = 128`) in VMEM and reduced via `jax.lax.broadcast_in_dim(jnp.sum(p, axis=1), l_prev.shape, (0,))` so every VPU operation aligns natively with TPU's $(8 \times 128)$ vector registers.
4. **SMEM `(8, 128)` Tile-Padding & Head Chunking**:
   - XLA pads the minor dimension of 2D/3D SMEM arrays up to a multiple of 128 lanes. Passing a 3D `lists` array of shape `(16, 128, 19)` at $N=16,384$ caused XLA to pad the last dimension from `19` to `128`, allocating $16 \times 128 \times 128 \times 4 = 1.00\text{ MB}$ and overflowing the 1.00 MB SMEM ceiling by 9.1 KB.
   - Resolved by flattening `(zh_chunk, nb)` into a single 1D row axis `(zh_chunk * nb, max_steps)` for `lists_ref` and `(zh_chunk * nb,)` for `counts_ref`, and computing `zh_chunk` against `padded_cols = ceil(max_steps / 128) * 128` within a safe 512 KB SMEM budget.
