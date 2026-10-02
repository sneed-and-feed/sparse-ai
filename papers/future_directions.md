# Project Q-Ultrametric: Future Directions & Quota-Exhaustion Roadmap

> **Sequel Strategy to:** *Learning to Skip Blocks* and *Llama Surgery*  
> **Target Hardware:** D-Wave Advantage™ (Pegasus $P_{16}$, 5,000+ qubits) & Advantage2™ Prototype (Zephyr $Z_{15}$, 7,000+ qubits)  
> **Economic Constraint:** \$0 Total Budget / 60-Second Monthly Leap™ Developer Tier Quota

---

## Executive Strategy: Maximum Density Quota Depletion

To extract maximum scientific and publication yield from a zero-dollar budget, we adhere to the **Single-Project Quota Exhaustion Model**:
* **Monthly Quota Allocation:** 60.0 seconds ($60,000\text{ ms}$) of pure QPU time per calendar month (renewable indefinitely by connecting an active GitHub developer profile).
* **QPU Call Structure:** A standard batch query of $N_{\text{reads}} = 500$ on the D-Wave Advantage QPU consumes:
  $$t_{\text{call}} = t_{\text{programming}} + N_{\text{reads}} \times (t_{\text{anneal}} + t_{\text{readout}} + t_{\text{delay}}) \approx 15\text{ ms} + (500 \times 0.07\text{ ms}) \approx 50\text{ ms} = 0.05\text{ seconds}$$
* **Monthly Submission Capacity:** 60.0 seconds provides **$1,200$ independent batch submissions** (at 500 reads each, yielding $600,000$ hardware samples/month).
* **The Hybrid Trap Warning:** Proprietary cloud hybrid endpoints (e.g., `LeapHybridSampler`, `LeapHybridCQMSampler`) enforce a strict minimum billing charge of **3 to 5 seconds per query**, which burns an entire monthly allocation in only 12 to 20 calls. **All hybrid decomposition, partitioning, and classical metaheuristics are executed 100% locally on CPU/GPU.** Pure `DWaveSampler` is used exclusively for hardware calls.
* **Execution Rule:** We tackle one flagship angle per monthly quota cycle. Every millisecond of QPU time is consumed on systematic parameter sweeps, physical annealing time variations ($t_a \in [1, 200]\,\mu\text{s}$), and 5-gauge Spin-Reversal Transforms (SRT).

Below, three strategic angles of entry are defined and ranked (1–3) by **Projected Return on Investment (ROI)**, balancing publication impact, theoretical novelty, experimental risk, and QPU packing density.

---

```
                       PROJECT Q-ULTRAMETRIC ROADMAP
                      
   Month 1 Cycle                   Month 2 Cycle                   Month 3 Cycle
   [ DIRECTION 1 ]                 [ DIRECTION 2 ]                 [ DIRECTION 3 ]
   HEA & Frustrated Spin Glasses   Hierarchical Modularity Cuts    Hardware-Aware QBMs
   ROI: ★★★★★                      ROI: ★★★★☆                      ROI: ★★★☆☆
   • 1,000+ QPU instances          • 300+ network instances        • 50+ training checkpoints
   • Exact Gurobi certification    • Classical OR comparison       • Generative ML benchmarks
   • PRX Quantum / PRB target      • ACM TOC / INFORMS target      • NeurIPS / ICLR target
```

---

## Direction 1: Hierarchical Edwards-Anderson (HEA) & Frustrated Ultrametric Spin Glasses
* **Projected ROI:** **★★★★★ (Highest ROI / Lowest Risk / Fastest to Publication)**
* **Target Venues:** *Physical Review X Quantum*, *Physical Review B*, *IEEE Transactions on Quantum Engineering (TQE)*
* **Mathematical Core:** $p$-Adic Tree Geometries, Quenched Hierarchical Variance ($1/2 < \sigma < 1$), Replica Symmetry Breaking (RSB)
* **Local Baselines:** Gurobi 12 (`MIPGap = 0.0`), `dwave-neal` Simulated Annealing, `dwave-tabu`, GPU-accelerated Simulated Bifurcation (SBM)

### 1. Scientific Thesis & Novelty
Dense spin glasses (such as the Sherrington-Kirkpatrick model) collapse on physical quantum annealers because their complete graph structure ($K_N$) forces minor embedding chains of length $L \propto N$, causing:
1. Quadratic physical qubit blowup ($Q = \Theta(N^2)$), capping capacity at $N \approx 180$ variables.
2. Coupler downscaling ($J_{\text{eff}} \propto 1/\sqrt{N}$) that drives signals beneath the QPU's 1–2% analog flux noise floor.
3. Rapid proliferation of chain breaks ($P_{\text{break}} \to 1$).

Direction 1 investigates the **Hierarchical Edwards-Anderson (HEA) model** (Franz, Parisi, Virasoro 1992; Castellana & Parisi 2011), where spins are leaves of a $p$-adic tree and interactions have independent quenched Gaussian disorder whose variance decays with hierarchical tree distance:
$$\mathbb{E}[(J_{ij}^{(l)})^2] = 2^{-2\sigma l}, \quad l = d_U(i, j) = \text{LCA depth}$$
In the critical regime $1/2 < \sigma < 1$, the model retains **full non-mean-field Parisi Replica Symmetry Breaking (RSB)** and worst-case NP-hardness (Branching Overlap Gap Property), yet its tree topology maps into Pegasus/Zephyr with **$O(N)$ physical qubits and $L_{\text{max}} \le 2$**.

### 2. Experimental Execution & QPU Quota Packing (Month 1)
* **Problem Instance Set:**
  * 50 instances of HEA spin glasses across sizes $N \in \{16, 32, 48, 64, 96, 128\}$.
  * 3 decay exponents: $\sigma = 0.6$ (deep glassy phase), $\sigma = 0.8$ (critical boundary), and $\sigma = 1.2$ (short-range droplet phase).
  * 15 Planted Frustrated Cluster Loops (Hen et al. 2015) for absolute tunneling verification.
  * Total unique problem instances: $N_{\text{inst}} = 165$.
* **QPU Execution Strategy:**
  * Annealing time sweep: $t_a \in \{1, 5, 20, 100\}\,\mu\text{s}$ (4 points per instance).
  * Spin-Reversal Transforms: 5 independent random gauges (100 reads per gauge = 500 reads).
  * Total submissions: $165 \times 4 = 660$ calls.
  * **QPU Quota Consumed:** $660 \text{ calls} \times 0.05\text{ s} = \mathbf{33.0\text{ seconds}}$ (leaving 27 seconds for verification sweeps).
* **Offline Verification:**
  * Solve all 165 instances to global optimality on local CPU via Gurobi to certify true ground states $E_{\text{gs}}$.
  * Measure Time-to-Solution ($TTS_{99}$), Ground State Probability ($P_{\text{gs}}$), and Chain Break Frequency (CBF).

---

## Direction 2: Hardware-Embeddable Hierarchical Graph Partitioning & Modularity Maximization
* **Projected ROI:** **★★★★☆ (High Practical Appeal / Broad Operations Research Impact)**
* **Target Venues:** *ACM Transactions on Quantum Computing*, *INFORMS Journal on Computing*, *Quantum Science and Technology*
* **Mathematical Core:** Multi-Prime Tree Separator Cuts, Newman Modularity, Differentiable Graph Partitioning
* **Local Baselines:** Gurobi (BQP formulation), D-Wave Tabu Multi-Start, Louvain Algorithm, Leiden Algorithm

### 1. Scientific Thesis & Novelty
Real-world networks (citation networks, biological interactomes, web graphs) naturally exhibit hierarchical modularity. The standard formulation of community detection via modularity maximization generates a **fully connected QUBO matrix**:
$$Q_{ij} = - \left( A_{ij} - \frac{k_i k_j}{2m} \right)$$
Because every node pair has an interaction, direct QPU embedding requires $K_N$ cliques, failing for $N > 100$.

Direction 2 deploys `sparse-ai`'s [`DynamicTopologyRouter`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/ultrametric/topology.py) to learn a **multi-scale tree partition** of the graph. The router assigns nodes to branches of an adèlic tree, converting the dense modularity matrix into an ultrametrically sparse, hardware-embeddable QUBO:
$$Q_{ij}^{\text{sparse}} = Q_{ij} \cdot \mathbb{I}[d_U(i, j) \le d_{\text{max}}]$$
By mapping inter-cluster cuts to collinear Pegasus buses and intra-cluster edges to native $K_4$ tiles, we solve community detection on graphs of **$N = 500$ to $2,000$ nodes** using a two-tier quantum-classical hybrid scheme with zero cloud solver charges.

### 2. Experimental Execution & QPU Quota Packing (Month 2)
* **Problem Instance Set:**
  * 20 synthetic LFR benchmark graphs (Lancichinetti-Fortunato-Radicchi) with hierarchical community structure ($N \in [100, 1000]$).
  * 15 real-world network datasets (Zachary Karate Club, Dolphin Social Network, Football, Coauthorship, Protein-Protein Interaction networks).
  * Total graph instances: 35 graphs $\times$ 4 scale cuts = 140 QUBO instances.
* **QPU Execution Strategy:**
  * 500 reads per instance with 5 SRT gauges.
  * Chain strength sweep: $\lambda \in \{0.8, 1.2, 1.6\} \times \text{RMS}(Q) \sqrt{L}$.
  * Submissions: $140 \times 3 = 420$ calls.
  * **QPU Quota Consumed:** $420 \text{ calls} \times 0.05\text{ s} = \mathbf{21.0\text{ seconds}}$.
* **Evaluation Metrics:**
  * Realized Newman modularity $Q_{\text{mod}}$, Normalized Mutual Information (NMI) against ground-truth clusters, and end-to-end wall-clock time vs. Gurobi and Louvain.

---

## Direction 3: Hardware-Aware Sparse Quantum Boltzmann Machines (HQ-QBM)
* **Projected ROI:** **★★★☆☆ (High Conceptual Upside / Higher Algorithmic Complexity)**
* **Target Venues:** *International Conference on Learning Representations (ICLR)*, *NeurIPS*, *Nature Machine Intelligence*
* **Mathematical Core:** Quantum Contrastive Divergence (QCD), Fenchel-Young Perturbed Optimizers, Straight-Through Topology Homotopy
* **Local Baselines:** Classical Restricted Boltzmann Machines (CD-$k$), Classical Deep Belief Networks, Native Chimera/Pegasus QBMs (Adachi & Henderson)

### 1. Scientific Thesis & Novelty
Quantum Boltzmann Machines (QBMs) offer non-classical Gibbs sampling distributions driven by quantum transverse fluctuations. However, existing QBMs face an impasse:
* **Fully Visible / Dense QBMs:** Require $K_N$ embeddings, suffering severe chain breaks and small variable counts ($N \le 32$).
* **Native Hardware QBMs:** Restrict connections strictly to physical couplers (e.g. flat Chimera grids), destroying the model's ability to learn hierarchical syntax or long-range semantics.

Direction 3 introduces the **Hardware-Aware Quantum Boltzmann Machine (HQ-QBM)**. A neural router co-trains alongside the QBM energy weights, enforcing that the bipartite visible-hidden interaction graph $J_{vh}$ adheres strictly to a Pegasus tree minor ($L_{\text{max}} \le 2$). By combining **Continuous Logit Homotopy** (annealing Gumbel temperature $\tau: 1.0 \to 0.1$) with **Perturbed Optimizer Gradient Estimation**, the model learns hierarchical data representations directly compatible with physical quantum annealers.

### 2. Experimental Execution & QPU Quota Packing (Month 3)
* **Training Protocol (Preserving QPU Quota):**
  * Early epochs (steps 0–1000): Pre-train the topology router and visible-hidden weights classically using GPU-accelerated Simulated Annealing (`dwave-neal` / SBM).
  * Fine-tuning phase (steps 1000–1200): Switch to physical QPU sampling for the negative (thermal) phase:
    $$\nabla_\theta \mathcal{L} \approx \langle \partial_\theta H \rangle_{\text{clamped}} - \langle \partial_\theta H \rangle_{\text{QPU}}$$
  * 200 physical gradient updates $\times$ 1 submission per update (500 reads) = 200 calls $\times$ 0.05 s = **$10.0\text{ seconds}$ QPU time**.
* **Generative Benchmarks:**
  * Synthetic Bars-and-Stripes (BAS), MNIST $8 \times 8$, and hierarchical Dyck language token embeddings.
  * Metrics: Negative Log-Likelihood (NLL via Annealed Importance Sampling), Fréchet Inception Distance (FID) on generated samples, and embedding chain break frequency.

---

## Comparative ROI Matrix

| Evaluation Dimension | Direction 1: HEA Spin Glasses | Direction 2: Hierarchical Modularity | Direction 3: Hardware-Aware QBM |
| :--- | :--- | :--- | :--- |
| **Projected ROI** | **★★★★★ (Rank 1)** | **★★★★☆ (Rank 2)** | **★★★☆☆ (Rank 3)** |
| **Scientific Venue** | *PRX Quantum*, *PRB*, *TQE* | *ACM TOC*, *INFORMS*, *QST* | *ICLR*, *NeurIPS*, *Nat. Mach. Intell.* |
| **QPU Execution Ease** | Pure batch runs (50 ms each) | Batch runs (50 ms each) | Iterative loop (in-the-loop training) |
| **Ground Truth Certification** | Exact ($E_{\text{gs}}$ via Gurobi 12) | Exact ($Q_{\text{max}}$ via Gurobi 12) | Approximate (AIS partition function) |
| **Experimental Risk** | **Extremely Low** | **Low to Moderate** | **Moderate to High** |
| **QPU Packing Density** | **600–1,000 runs / month** | **300–500 runs / month** | **150–250 runs / month** |
| **Theoretical Uniqueness** | Solves $K_N$ noise floor squashing | Bridges graph theory to QPU | First differentiable topology QBM |

---

## Immediate Next Steps: Initiating Direction 1

1. **Verify Local Environment:** Confirm installation of `dimod`, `dwave-neal`, `minorminer`, and `dwave-tabu`.
2. **Implement `src/quantum/` Scaffold:**
   * `src/quantum/hea_generator.py`: Generates synthetic Hierarchical Edwards-Anderson spin glasses across arbitrary branching factors $p$, tree depths $L$, and decay exponents $\sigma$.
   * `src/quantum/embeddings.py`: Implements deterministic tree and H-tree minor embedding routines for Pegasus and Zephyr.
   * `src/quantum/benchmarks.py`: Orchestrates local Gurobi 12, Simulated Annealing, and Tabu sweeps.
3. **Execute Local Zero-Cost Pilot:** Run 20 HEA instances ($N = 16, 32$) through the offline pipeline to certify ground truth energies and establish exact classical baseline runtimes before submitting to D-Wave Leap.
