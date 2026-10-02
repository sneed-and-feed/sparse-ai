"""
Planted Frustrated Cluster Loop Generator.

Implements the planted-solution spin glass model of Hen et al. (2015)
(Physical Review A 92, 042325) with analytically certified ground truth energies
and planted spin configurations for quantum annealing benchmarks.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
import networkx as nx
import dimod


@dataclass
class PlantedLoopInstance:
    """Represents a planted frustrated cluster loop problem instance."""
    num_spins: int
    num_loops: int
    loop_size: int
    seed: int
    planted_state: Dict[int, int]
    analytical_ground_energy: float
    bqm: dimod.BinaryQuadraticModel
    graph: nx.Graph
    loops: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes instance to JSON-compatible dictionary."""
        return {
            "type": "planted_frustrated_loops",
            "num_spins": self.num_spins,
            "num_loops": self.num_loops,
            "loop_size": self.loop_size,
            "seed": self.seed,
            "analytical_ground_energy": float(self.analytical_ground_energy),
            "planted_state": {int(k): int(v) for k, v in self.planted_state.items()},
            "linear": {int(k): float(v) for k, v in self.bqm.linear.items()},
            "quadratic": [
                {"u": int(u), "v": int(v), "weight": float(w)}
                for (u, v), w in self.bqm.quadratic.items()
            ],
            "loops": [[int(node) for node in loop] for loop in self.loops],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlantedLoopInstance":
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

        return cls(
            num_spins=data["num_spins"],
            num_loops=data["num_loops"],
            loop_size=data["loop_size"],
            seed=data["seed"],
            planted_state={int(k): int(v) for k, v in data["planted_state"].items()},
            analytical_ground_energy=float(data["analytical_ground_energy"]),
            bqm=bqm,
            graph=g,
            loops=data["loops"],
        )


def generate_planted_frustrated_loops(
    num_spins: int = 16,
    num_loops: Optional[int] = None,
    loop_size: int = 4,
    seed: Optional[int] = None,
    topology: str = "cluster",
) -> PlantedLoopInstance:
    """
    Generates a planted frustrated cluster loop instance following Hen et al. (2015).

    Methodology:
    1. Sample a random planted ground state s* in {-1, +1}^N.
    2. Construct a set of elementary loops (cycles of length loop_size, default 4).
    3. In each loop C_m, exactly one edge is chosen to be frustrated (+1 relative to s*),
       and the remaining (k - 1) edges are satisfied (-1 relative to s*).
    4. By construction, each loop has frustration signature prod sign(K_e) = -1.
    5. The minimum possible energy of a loop of length k with unit bonds is -(k - 2).
    6. For non-interfering or cluster-decomposable loops, s* attains this global lower bound,
       certifying that E_gs = sum_m -(k_m - 2).

    Args:
        num_spins: Total number of spins N.
        num_loops: Number of frustrated loops to place. Default is num_spins // 2.
        loop_size: Size of elementary loops (typically 4 for planar/Pegasus tiles).
        seed: Random seed for reproducibility.
        topology: 'cluster' (hierarchical tile-compatible) or 'ring'.

    Returns:
        PlantedLoopInstance with certified ground state energy.
    """
    rng = np.random.default_rng(seed)
    if num_loops is None:
        num_loops = max(1, num_spins // loop_size)

    # 1. Sample planted state
    s_planted = {i: int(rng.choice([-1, 1])) for i in range(num_spins)}

    # 2. Form loops
    loops: List[List[int]] = []
    if topology == "cluster":
        # Group spins into local clusters of size loop_size (e.g. 4-spin clusters matching Pegasus K_4 tiles)
        num_clusters = num_spins // loop_size
        for c in range(min(num_loops, num_clusters)):
            start_idx = c * loop_size
            cluster_nodes = list(range(start_idx, start_idx + loop_size))
            # Permute order to form cycle
            perm = list(rng.permutation(cluster_nodes))
            loops.append([int(x) for x in perm])

        # If more loops are requested than disjoint clusters, create inter-cluster bridge loops
        for extra in range(num_clusters, num_loops):
            c1 = rng.integers(0, num_clusters)
            c2 = (c1 + 1) % num_clusters
            # pick 2 nodes from c1, 2 nodes from c2
            nodes_c1 = list(rng.choice(range(c1 * loop_size, (c1 + 1) * loop_size), size=2, replace=False))
            nodes_c2 = list(rng.choice(range(c2 * loop_size, (c2 + 1) * loop_size), size=2, replace=False))
            loop = [int(nodes_c1[0]), int(nodes_c1[1]), int(nodes_c2[0]), int(nodes_c2[1])]
            loops.append(loop)
    else:
        # Ring/sliding window cycles
        for i in range(num_loops):
            start = (i * 2) % num_spins
            loop = [(start + j) % num_spins for j in range(loop_size)]
            loops.append(loop)

    # 3. Assign couplings for each loop
    quadratic: Dict[Tuple[int, int], float] = {}
    total_analytical_energy = 0.0

    for loop in loops:
        k = len(loop)
        total_analytical_energy += -(k - 2)
        # Select exactly 1 bond to frustrate
        frustrated_bond = int(rng.integers(0, k))

        for idx in range(k):
            u = loop[idx]
            v = loop[(idx + 1) % k]
            edge = (min(u, v), max(u, v))

            # Frustrated bond has sign +1 (antiferromagnetic relative to s*),
            # satisfied bonds have sign -1 (ferromagnetic relative to s*)
            sign = 1.0 if idx == frustrated_bond else -1.0
            j_val = sign * float(s_planted[u] * s_planted[v])

            quadratic[edge] = quadratic.get(edge, 0.0) + j_val

    # Construct BQM (Ising model: linear fields 0, vartype=SPIN)
    linear = {i: 0.0 for i in range(num_spins)}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)

    # Construct graph
    g = nx.Graph()
    g.add_nodes_from(range(num_spins))
    for (u, v), w in quadratic.items():
        g.add_edge(u, v, weight=w)

    return PlantedLoopInstance(
        num_spins=num_spins,
        num_loops=len(loops),
        loop_size=loop_size,
        seed=seed if seed is not None else 0,
        planted_state=s_planted,
        analytical_ground_energy=float(total_analytical_energy),
        bqm=bqm,
        graph=g,
        loops=loops,
    )
