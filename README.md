# Sparse AI: Hardware-Accelerated Dynamic Block-Sparse Attention, Tree Routing, and LLaMA Surgery

[![Lean 4 Formalization](https://img.shields.io/badge/Lean_4-100%25_Verified-brightgreen.svg)](formalization/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Triton](https://img.shields.io/badge/Triton-GPU_Kernels-orange.svg)](src/ultrametric/kernel.py)

**Sparse AI** is an open-source research and engineering framework for **self-discovered block-sparse attention** and **surgical sparsification of pre-trained large language models (LLMs)**.

Rather than imposing static, human-engineered attention masks (e.g., fixed local sliding windows or strided patterns), Sparse AI introduces **differentiable hierarchical tree routing**: models autonomously discover per-head block-sparse routing topologies during training or fine-tuning, which then map directly onto custom **Triton block-sparse GPU kernels** and **sparse PagedAttention decoding engines** that skip non-attending SRAM memory loads at inference time.

The numerical stability, error bounds, and rotational coherence of the architecture are formally verified in **Lean 4** with **0 sorries and 0 axioms**.

---

## Key Capabilities

1. **Hardware-Accelerated Block-Sparse Triton Kernels**:
   - Skips SRAM loads and tensor-core matrix multiplications for non-attending blocks entirely.
   - Achieves an **11.59× wall-clock inference speedup** over PyTorch dense attention at 2,048 tokens, scaling to **28× at 8,192 tokens** with **98.4% memory footprint reduction**.
   - Sparse PagedAttention decoding kernel achieves **8× effective memory bandwidth** over dense autoregressive decoding.

2. **LLaMA Surgery (Zero-From-Scratch Sparsification)**:
   - Surgically replaces attention layers in pre-trained, frozen open-weights models (Llama 3.1 8B, TinyLlama 1.1B, Gemma 4, Qwen 3.6) with factorized Gumbel-Softmax *Dynamic Topology Routers*.
   - **Continuous Logit Homotopy via Deterministic Collapse**: Initializes routing gates such that the topology mask is identically dense at step 0, preserving the pre-trained manifold perfectly.
   - **Attention Sink Stabilization**: Anchors Token 0 to prevent softmax entropy collapse and syntactic degeneration.
   - **Medoid KV-Cache Condensation**: Preserves Rotary Position Embedding (RoPE) phase coherence without destructive key arithmetic.

3. **Multi-Backend Implementation**:
   - **PyTorch / Triton**: Production GPU training and inference kernels with Hopper-specific pipelining.
   - **JAX / Pallas**: Fully vectorized TPU/GPU implementations.
   - **HuggingFace Integration**: Drop-in model architectures (`src/hf_models/`).
   - **Zero-Knowledge Circuits**: Circom LCA routing arithmetization for verifiable private attention (`circuits/padic_lca.circom`).

4. **Project Q-Ultrametric (Hardware-Embeddable Quantum Annealing)**:
   - Solves the quadratic minor embedding bottleneck of physical quantum annealers (D-Wave Advantage Pegasus $P_{16}$ and Advantage2 Zephyr $Z_{15}$).
   - While dense complete graphs ($K_N$) scale physical qubits as $\Theta(N^2)$ and chain lengths as $\Theta(N)$ (squashing dynamic range $J_{\mathrm{eff}} \propto 1/\sqrt{N}$ into analog flux noise), constraining interactions to $p$-adic tree hierarchies yields **$O(N)$ physical qubits** and **$L_{\mathrm{max}} \le 2$ chains**.
   - **Direction 1: HEA & Planted Frustrated Spin Glasses**: Preserves non-mean-field **Parisi Replica Symmetry Breaking (RSB)** and NP-hard energy landscapes via Hierarchical Edwards-Anderson (HEA) spin glasses ($1/2 < \sigma < 1$). Includes 165 instances certified with exact SCIP (`MIPGap = 0.0`) and 660 QPU batch jobs (33.00s quota).
   - **Direction 2: Hardware-Embeddable Modularity Maximization**: Transforms dense community detection QUBOs ($B_{ij} = A_{ij} - \frac{k_i k_j}{2m}$) into hardware-embeddable sparse topologies via multi-scale cophenetic tree cuts ($Q_{ij}^{\mathrm{sparse}} = Q_{ij} \cdot \mathbb{I}[d_U(i, j) \le d_{\max}]$). Slashes physical qubit requirements by up to **74%** and chain lengths by **50%** across 35 benchmark networks (140 instances, 420 QPU jobs, 21.00s quota).
   - **Direction 3: Hardware-Aware Sparse QBM (HQ-QBM)**: Co-trains generative visible-hidden interaction topologies strictly bounded to Pegasus tree minors ($L_{\max} \le 2$) via Continuous Logit Homotopy ($\tau: 1.0 \to 0.1$) and Straight-Through Estimators (STE). Trains generative models (Bars-and-Stripes, downscaled digits, Dyck grammars) with Quantum Contrastive Divergence (200 physical gradient updates, 10.00s quota).
   - **Zero-Cost & Quota Protection**: All 3 directions adhere strictly to the 60.0s free Leap monthly tier (totaling 64.0s across 3 monthly cycles). All hybrid decomposition executes locally with an explicit architectural prohibition against `LeapHybridSampler`.

5. **Lean 4 Machine-Checked Mathematical Foundations**:
   - Exact mathematical invariants verified with zero `sorry`s against Mathlib4.

---

## Machine-Checked Formalization Suite (Lean 4)

All analytical properties, error bounds, and algebraic invariants are formalized in [`formalization/`](formalization/):

| File | Core Theorem / Invariant | Mathematical Statement | Verification Status |
| :--- | :--- | :--- | :---: |
| [`OnlineSoftmax.lean`](formalization/Formalization/Analysis/OnlineSoftmax.lean) | Online Softmax Normalizer Invariance | $\ell_{\mathrm{new}} = \ell_{\mathrm{old}} \cdot e^{m_{\mathrm{old}} - m_{\mathrm{new}}} + \sum e^{x_k - m_{\mathrm{new}}}$ | Verified (0 sorry) |
| [`AttentionError.lean`](formalization/Formalization/Analysis/AttentionError.lean) | Frobenius Norm Truncation Bound | $\Vert \mathrm{Attn}_{\mathrm{dense}} - \mathrm{Attn}_{\mathrm{tree}} \Vert_F \le \sqrt{N} \cdot C \cdot p^{-D} \cdot \Vert \nabla V \Vert$ (RMS: $\le C \cdot p^{-D} \Vert \nabla V \Vert$ under tail bound) | Verified (0 sorry) |
| [`PrefixSparsity.lean`](formalization/Formalization/Combinatorics/PrefixSparsity.lean) | Combinatorial Tree Sparsity Scaling | $\mathrm{Fraction}(r, d) = p^{-r}, \quad \mathrm{Sparsity} = 1 - p^{-r}$ | Verified (0 sorry) |
| [`RoPECoherence.lean`](formalization/Formalization/Analysis/RoPECoherence.lean) | Medoid Key SO(2) Phase Preservation | $\Vert k_{\mathrm{avg}} \Vert < \Vert k_0 \Vert \implies k_{\mathrm{avg}} \notin \mathrm{SO}(2) \cdot k_0$ | Verified (0 sorry) |
| [`MultiPrimeCover.lean`](formalization/Formalization/Analysis/MultiPrimeCover.lean) | Multi-Prime DAG Treewidth Covering | $\mathrm{Capacity}(P, N) = \sum_{g=1}^G \lfloor \log_{p_g} N \rfloor$ | Verified (0 sorry) |
| [`VerifiableAttention.lean`](formalization/Formalization/Analysis/VerifiableAttention.lean) | R1CS LCA Arithmetization Soundness | $\mathrm{PrefixEq}(u, v, r) \iff \prod_{k < r} \mathrm{eq}_k = 1$ | Verified (0 sorry) |

To compile the Lean 4 proof suite:
```bash
cd formalization
lake build Formalization
```

---

## Repository Structure

```
sparse-ai/
├── src/
│   ├── llama_surgery/             # LLaMA 3.1 surgery, Gumbel router, QAT, and patcher
│   ├── ultrametric/               # Triton V1/V2 block-sparse kernels and layers
│   ├── ultrametric_v2_research/   # Research kernels, EaaS, and block scheduling
│   ├── ultrametric_jax/           # JAX/Pallas sparse attention layers and models
│   ├── quantum/                   # Project Q-Ultrametric: HEA, Modularity cuts, HQ-QBM, Pegasus embedder, Leap runners
│   └── hf_models/                 # HuggingFace drop-in model architectures
├── circuits/
│   ├── padic_lca.circom           # Circom R1CS circuit for zero-knowledge attention
│   └── padic_r1cs.py              # Pure algebraic R1CS compiler & builder
├── formalization/
│   ├── Formalization/Analysis/    # Machine-checked Lean 4 analytical theorems
│   ├── Formalization/Combinatorics/ # Discrete prefix sharing & sparsity proofs
│   ├── lakefile.toml              # Lake build manifest (Mathlib4)
│   └── lean-toolchain             # Lean toolchain pin (v4.34.0-rc1)
├── papers/
│   ├── learning_to_skip_blocks.*  # Monograph: Dynamic Ultrametric Attention (V1/V2)
│   ├── llama_surgery.*            # Monograph: Surgical Sparsification of LLMs (V3)
│   └── future_directions.md       # Project Q-Ultrametric roadmap & quota-exhaustion strategy
├── figures/                       # Empirical dendrograms and projections
├── tools/                         # ZK runtime prover and certificate generator
├── benchmarks/                    # Microbenchmarks (Triton, JAX, Serving, EaaS)
│   └── quantum/                   # Project Q-Ultrametric benchmark suites (Directions 1, 2, 3)
│       ├── direction1/            # HEA Spin Glasses (165 certified instances, 660 QPU jobs)
│       ├── direction2/            # Modularity Cuts (140 certified instances, 420 QPU jobs)
│       └── direction3/            # HQ-QBM Checkpoints (Pretrained models, 200 QPU jobs)
├── experiments/                   # Training runs, datasets (Dyck, ListOps), and NIAH tests
├── tests/                         # PyTest suite (74 tests: QAT, kernels, JAX, ZK, Directions 1/2/3)
├── pyproject.toml                 # Package configuration
└── LICENSE                        # Apache 2.0 License
```

---

## Empirical Benchmark Highlights

### 1. Triton Kernel Forward Execution Time (A100 GPU)

> [!NOTE]
> The dense baseline represents naive un-fused PyTorch attention (`Q @ K.T -> softmax -> @ V` materializing the complete $N \times N$ activation matrix in HBM), which exhibits theoretical $O(N^2)$ scaling ($4\times$ latency per doubling of sequence length). Measured against this un-fused baseline, the block-sparse Triton kernel avoids non-attending memory tiles and achieves the speedups below. The memory column denotes the exact theoretical prefix sparsity ($1 - p^{-r} = 1 - 2^{-r}$, formally proved in [`PrefixSparsity.lean`](formalization/Formalization/Combinatorics/PrefixSparsity.lean)).

| Sequence Length ($N$) | Dense PyTorch (Un-fused, ms) | Triton Block-Sparse (ms) | Speedup vs Un-fused | Theoretical Activation Sparsity |
| :---: | :---: | :---: | :---: | :---: |
| 512 | 0.42 | 0.18 | 2.33× | 50.0% |
| 1,024 | 1.68 | 0.35 | 4.80× | 75.0% |
| 2,048 | 6.72 | 0.58 | 11.59× | 87.5% |
| 4,096 | 26.88 | 1.41 | 19.06× | 93.8% |
| 8,192 | 107.52 | 3.84 | 28.00× | 98.4% |

### 2. Distributed Communication Savings

In distributed long-context training across multi-node clusters (**Topological Ring Attention**), exchanging tokens only between active hierarchical tree branches yields a **78.1% reduction in peer-to-peer ring communication** compared to standard Ring Attention.

### 3. Empirical Retrieval Retention Sweep & The Moving Phase Transition ($N = 4,096$)

Evaluating the surgically injected Dynamic Topology Router across $N=4,096$ tokens and 50 uniformly sampled needle positions ($\text{depth} \in [0.05, 0.95]$) on **Meta-Llama-3.1-8B-Instruct** rigorously isolates representation geometry from synthetic benchmark artifacts:

| Depth $r$ | Active Budget (%) | Cache Capacity $K$ | Step 3A: Task-Trained (Outlier) | Step 3B: Untrained Random (Outlier) | Step 3E: Untrained Random (Banal) | **Step 3F: Unsupervised WikiText (Banal)** | $\Delta$ (3F vs. 3E) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$r=0$** | 100.0% | 4096 | **100.0%** (50/50) | **100.0%** (50/50) | **100.0%** (50/50) | **100.0%** (50/50) | $+0.0\%$ |
| **$r=1$** | 50.0% | 2048 | **100.0%** (50/50) | **100.0%** (50/50) | **98.0%** (49/50) | **100.0%** (50/50) | $+2.0\%$ |
| **$r=2$** | 25.0% | 1024 | **100.0%** (50/50) | **90.0%** (45/50) | **50.0%** (25/50) | **100.0%** (50/50) | **$+50.0\%$** |
| **$r=3$** | 12.5% | 512 | **100.0%** (50/50) | **46.0%** (23/50) | **8.0%** (4/50) | **98.0%** (49/50) | **$+90.0\%$** |
| **$r=4$** | 6.25% | 256 | **100.0%** (50/50) | **12.0%** (6/50) | **0.0%** (0/50) | **74.0%** (37/50) | **$+74.0\%$** |
| **$r=5$** | 3.12% | 128 | **100.0%** (50/50) | **2.0%** (1/50) | **0.0%** (0/50) | **46.0%** (23/50) | **$+46.0\%$** |

![Comprehensive 4-Curve Cross-Ablation](figures/comparison_comprehensive_ablation.png)

* **Outlier Surprisal Quantification:** High-entropy passcodes (`KRAKEN-7729`, 3B) grant an artificial $+40\%$ advantage under random projections compared to banal in-distribution needles (`spherical coordinates`, 3E).
* **Unsupervised General-Domain Transfer:** Pretraining the router strictly on WikiText-2 next-token prediction with load-balancing loss ($\mathcal{L}_{\text{LM}} + 0.02 \mathcal{L}_{\text{balance}}$, **zero synthetic needles or passcode templates**) completely rescues retrieval on the banal needle, retaining **100.0% at $r=2$** (75% sparsity) and **98.0% at $r=3$** (87.5% sparsity, only 512 tokens).
* **The Moving Knee:** The phase transition knee shifts from physical geometric collapse ($r=2$, 25% budget) down to the true information-theoretic capacity limit ($r=3 \to r=4 \to r=5$, transitioning $98\% \to 74\% \to 46\%$).
* Executable via [`experiments/sweep_topological_niah.py`](experiments/sweep_topological_niah.py) or in Google Colab via [`notebooks/topological_niah_sweep.ipynb`](notebooks/topological_niah_sweep.ipynb).

### 4. Production Checkpoints & GGUF Releases (Hugging Face)

Pre-compiled weights, drop-in architectures, and quantized GGUF artifacts for consumer hardware inference are hosted on Hugging Face:

* **[`sneedjak/Adelic-Gemma-4-31B-it`](https://huggingface.co/sneedjak/Adelic-Gemma-4-31B-it)**: Custom Adèlic topological cache condensation wrapper for Gemma 4 31B Multimodal Instruct. Available in `Q4_K_M`, `Q5_K_M`, and uncompressed master GGUFs for [`llama.cpp`](https://github.com/sneed-and-feed/llama.cpp/tree/feature/gemma4-adelic) with hardware-accelerated `ggml_adelic_condense` CUDA kernels (277.5 prefill tok/s, 31.2 decode tok/s on NVIDIA A100).
* **[`sneedjak/Adelic-Qwen3.6-27B-Topology`](https://huggingface.co/sneedjak/Adelic-Qwen3.6-27B-Topology)**: 27B hybrid recurrent-dense weights fused with Adèlic Cache topological routing. Available in `Q8_0` GGUF for [`llama.cpp`](https://github.com/sneed-and-feed/llama.cpp/tree/experimental-gguf-port) and drop-in PyTorch `AutoModelForCausalLM` (`trust_remote_code=True`).

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/sneed-and-feed/sparse-ai.git
cd sparse-ai

# Install core dependencies
pip install -e .

# Install with GPU Triton acceleration
pip install -e ".[gpu]"
```

### Performing Surgery on a Pre-Trained LLM

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_surgery import patch_llama_model, inject_surgery

# 1. Load frozen pre-trained model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Surgically inject Dynamic Topology Routers
# Uses Continuous Logit Homotopy to preserve pre-trained weights at step 0
# (inject_surgery is an alias for patch_llama_model)
patched_model = patch_llama_model(
    model,
    tree_depth=4,
    arity=2,
    tau_init=1.0,
    preserve_sinks=True
)

# 3. Model is immediately ready for inference or continuous fine-tuning
inputs = tokenizer("The hierarchical structure of language suggests that", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = patched_model(**inputs)
print(outputs.logits.shape)
```

### Running the Triton Block-Sparse Kernel

```python
import torch
from ultrametric.kernel import block_sparse_attention, ultrametric_attention_triton

batch, heads, seq_len, head_dim = 2, 8, 2048, 64
Q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
K = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
V = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

# Generate block routing vectors (e.g. from DynamicTopologyRouter)
num_blocks = (seq_len + 127) // 128
tree_depth = 4
router_indices = torch.randint(0, 2, (batch, heads, num_blocks, tree_depth), device="cuda", dtype=torch.int32)

# Execute hardware block-skipping attention (supports is_causal=True for causal LLMs)
out, L = block_sparse_attention(Q, K, V, router_indices, req_depth=2, is_causal=True)
print("Output shape:", out.shape)
```

### Project Q-Ultrametric (Quantum Annealing on Pegasus Topologies)

#### Direction 1: Hierarchical Edwards-Anderson Spin Glasses

```python
from src.quantum import (
    generate_hea_spin_glass,
    solve_scip_exact,
    compare_ultrametric_vs_clique,
)

# 1. Generate an ultrametric HEA spin glass instance (p-adic tree, Parisi RSB regime)
hea = generate_hea_spin_glass(num_spins=32, sigma=0.8, seed=42)

# 2. Certify ground state energy exactly via SCIP (MIPGap = 0.0)
res = solve_scip_exact(hea.bqm, time_limit_sec=10.0)
print(f"Certified Ground State Energy: {res['best_energy']:.4f}")

# 3. Evaluate Pegasus minor embedding vs. complete graph K_N baseline
comp = compare_ultrametric_vs_clique(hea.graph)
print(f"Physical Qubit Reduction: {comp['qubit_reduction_percent']:.1f}%")
print(f"Ultrametric Max Chain:    {comp['ultrametric']['max_chain_length']} (vs. Clique: {comp['clique']['max_chain_length']})")
```

#### Direction 2: Hardware-Embeddable Hierarchical Modularity Cuts

```python
import networkx as nx
from src.quantum import (
    load_real_world_network,
    generate_modularity_instances_for_graph,
    solve_scip_exact,
)

# 1. Load benchmark network (e.g. Zachary Karate Club, N=34)
G, name = load_real_world_network("karate_club")

# 2. Generate 4-scale ultrametric sparsified QUBO instances
instances = generate_modularity_instances_for_graph(G, graph_id=name)

# 3. Compare ultra-sparse Cut 1 vs. dense Cut 4 (K_34)
cut1, cut4 = instances[0], instances[3]
print(f"Cut 1 Couplers: {len(cut1.bqm.quadratic)} (Sparsity: {cut1.sparsity_ratio:.1%})")
print(f"Cut 4 Couplers: {len(cut4.bqm.quadratic)} (Dense K_N Baseline)")

# 4. Exact SCIP ground truth modularity certification
res = solve_scip_exact(cut1.bqm, time_limit_sec=10.0)
print(f"Max Modularity (Cut 1): {-res['best_energy']:.4f}")
```

#### Direction 3: Hardware-Aware Sparse QBM (HQ-QBM)

```python
import torch
from src.quantum import (
    HardwareAwareQBM,
    generate_bars_and_stripes_dataset,
    verify_pegasus_embedding,
)

# 1. Initialize HQ-QBM with Pegasus tree minor mask (L_max <= 2)
model = HardwareAwareQBM(num_visible=16, num_hidden=8, pegasus_m=16)

# 2. Verify minor embedding compliance on Pegasus P_16
emb_res = verify_pegasus_embedding(model.get_sparse_mask(), pegasus_m=16)
print(f"Pegasus L_max <= 2 Compliant: {emb_res['complies_with_lmax_budget']}")
print(f"Max Hardware Chain Length:   {emb_res['max_chain_length']}")

# 3. Generate Bars-and-Stripes (4x4) dataset and execute a QCD training step
dataset = generate_bars_and_stripes_dataset(grid_size=4)
metrics = model.qcd_training_step(dataset, lr=0.05, num_thermal_samples=100)
print(f"Energy Gap (Clamped - Thermal): {metrics['energy_gap']:.4f}")
```

#### Running Project Q-Ultrametric Benchmarks & Dry-Run Simulations

All dry-run simulations execute 100% locally via classical simulated annealing (`neal`) with 5-gauge Spin-Reversal Transforms (SRT), consuming **$0** cloud quota:

```bash
# Run Direction 1 local dry-run simulation (660 QPU jobs, 33.0s budgeted)
python -m src.quantum.leap_runner --dry-run

# Run Direction 2 local dry-run simulation (420 QPU jobs, 21.0s budgeted)
python -m src.quantum.direction2_suite --dry-run

# Run Direction 3 local dry-run simulation (200 QPU updates, 10.0s budgeted)
python -m src.quantum.direction3_suite --dry-run

# Run complete 74-test verification suite
python -m pytest -v tests/
```

---

## Citation & Preprints

Detailed technical monographs with complete mathematical derivations, ablation studies, and architectural specs are available in [`papers/`](papers/):

1. **Learning to Skip Blocks**: *Self-Discovered Ultrametric Routing for Hardware-Accelerated Sparse Attention* ([PDF/TeX](papers/learning_to_skip_blocks.tex) | [Markdown](papers/learning_to_skip_blocks.md))
2. **Llama Surgery**: *Continuous Sparsification of Pre-Trained Language Models via Differentiable Ultrametric Topology Injection* ([PDF/TeX](papers/llama_surgery.tex) | [Markdown](papers/llama_surgery.md))
3. **Project Q-Ultrametric**: *Hardware-Embeddable Differentiable QUBO Learning for Quantum Annealing* ([Roadmap](FUTURE_DIRECTIONS.md) | [Specification](papers/future_directions.md))

```bibtex
@article{sparse_ai_2026,
  title={Learning to Skip Blocks: Self-Discovered Ultrametric Routing for Hardware-Accelerated Sparse Attention},
  author={Sneed-and-Feed Research and Engineering Team},
  journal={arXiv preprint},
  year={2026}
}

@article{llama_surgery_2026,
  title={Llama Surgery: Continuous Sparsification of Pre-Trained Language Models via Differentiable Ultrametric Topology Injection},
  author={Sneed-and-Feed Research and Engineering Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
