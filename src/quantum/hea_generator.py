"""
Hierarchical Edwards-Anderson (HEA) Spin Glass Generator.

Implements the non-mean-field hierarchical spin glass model of
Franz, Parisi, and Virasoro (1992) and Castellana & Parisi (2011).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import math
import numpy as np
import networkx as nx
import dimod


@dataclass
class HEAInstance:
    """Represents a Hierarchical Edwards-Anderson spin glass instance."""
    num_spins: int
    p: int
    levels: int
    sigma: float
    seed: int
    bqm: dimod.BinaryQuadraticModel
    graph: nx.Graph
    lca_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Serializes instance to JSON-compatible dictionary."""
        return {
            "type": "hierarchical_edwards_anderson",
            "num_spins": self.num_spins,
            "p": self.p,
            "levels": self.levels,
            "sigma": float(self.sigma),
            "seed": self.seed,
            "linear": {int(k): float(v) for k, v in self.bqm.linear.items()},
            "quadratic": [
                {"u": int(u), "v": int(v), "weight": float(w)}
                for (u, v), w in self.bqm.quadratic.items()
            ],
            "lca_matrix": self.lca_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HEAInstance":
        """Deserializes instance from dictionary."""
        linear = {int(k): float(v) for k, v in data["linear"].items()}
        quadratic = {
            (int(item["u"]), int(item["v"])): float(item["weight"])
            for item in data["quadratic"]
        }
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)

        g = nx.Graph()
        g.add_nodes_from(range(data["num_spins"]))
        for (u, v), w in quadratic.items():
            g.add_edge(u, v, weight=w)

        lca_mat = np.array(data["lca_matrix"], dtype=int)

        return cls(
            num_spins=data["num_spins"],
            p=data["p"],
            levels=data["levels"],
            sigma=float(data["sigma"]),
            seed=data["seed"],
            bqm=bqm,
            graph=g,
            lca_matrix=lca_mat,
        )


def compute_p_adic_lca_distance(i: int, j: int, p: int = 2, levels: Optional[int] = None) -> int:
    """
    Computes the hierarchical lowest common ancestor (LCA) distance between two leaf indices i, j.
    Level 1 represents sharing immediate parent; higher level represents higher common ancestor.
    """
    if i == j:
        return 0
    # Find the highest level at which digits differ in base p
    val_i, val_j = i, j
    dist = 0
    max_divergence = 0
    while val_i > 0 or val_j > 0:
        dist += 1
        if (val_i % p) != (val_j % p):
            max_divergence = dist
        val_i //= p
        val_j //= p
    return max_divergence


def generate_hea_spin_glass(
    levels: Optional[int] = None,
    num_spins: Optional[int] = None,
    p: int = 2,
    sigma: float = 0.8,
    seed: Optional[int] = None,
    h_bias_std: float = 0.0,
    normalize_couplers: bool = False,
) -> HEAInstance:
    """
    Generates a Hierarchical Edwards-Anderson spin glass on N spins.

    Coupling between spins i and j at hierarchical LCA level l:
        J_{ij} ~ Normal(0, p^(-2 * sigma * l))

    Args:
        levels: Depth of hierarchical tree. If provided without num_spins, N = p^levels.
        num_spins: Total number of spins N. If provided, levels is computed as ceil(log_p(N)).
        p: Tree branching factor (p=2 is binary tree).
        sigma: Decay exponent.
               - 1/2 < sigma < 1: True spin glass phase with full Replica Symmetry Breaking.
               - sigma > 1: Short-range droplet phase.
        seed: Random seed for reproducibility.
        h_bias_std: Standard deviation of longitudinal local magnetic fields h_i.
        normalize_couplers: If True, rescale all couplers so max(|J_{ij}|) <= 1.0 (D-Wave DAC limit).

    Returns:
        HEAInstance containing the BQM, NetworkX graph, and metadata.
    """
    rng = np.random.default_rng(seed)

    if num_spins is not None:
        actual_num_spins = num_spins
        actual_levels = max(1, math.ceil(math.log(actual_num_spins, p)))
    elif levels is not None:
        actual_levels = levels
        actual_num_spins = p ** levels
    else:
        actual_levels = 5
        actual_num_spins = p ** actual_levels

    lca_mat = np.zeros((actual_num_spins, actual_num_spins), dtype=int)
    for i in range(actual_num_spins):
        for j in range(i + 1, actual_num_spins):
            dist = compute_p_adic_lca_distance(i, j, p=p, levels=actual_levels)
            lca_mat[i, j] = dist
            lca_mat[j, i] = dist

    linear = {}
    if h_bias_std > 0:
        for i in range(actual_num_spins):
            linear[i] = float(rng.normal(0.0, h_bias_std))
    else:
        for i in range(actual_num_spins):
            linear[i] = 0.0

    raw_quadratic = {}
    for i in range(actual_num_spins):
        for j in range(i + 1, actual_num_spins):
            l = lca_mat[i, j]
            std = float(p ** (-sigma * l))
            weight = float(rng.normal(0.0, std))
            raw_quadratic[(i, j)] = weight

    # Optional DAC normalization to [-1.0, 1.0]
    if normalize_couplers and raw_quadratic:
        max_j = max(abs(w) for w in raw_quadratic.values())
        scale = 1.0 / max(1e-12, max_j)
        quadratic = {edge: w * scale for edge, w in raw_quadratic.items()}
    else:
        quadratic = raw_quadratic

    g = nx.Graph()
    g.add_nodes_from(range(actual_num_spins))
    for (i, j), weight in quadratic.items():
        g.add_edge(i, j, weight=weight, lca_level=int(lca_mat[i, j]))

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)

    return HEAInstance(
        num_spins=actual_num_spins,
        p=p,
        levels=actual_levels,
        sigma=sigma,
        seed=seed if seed is not None else 0,
        bqm=bqm,
        graph=g,
        lca_matrix=lca_mat,
    )
