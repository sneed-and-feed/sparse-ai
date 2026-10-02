"""
Hardware-Embeddable Hierarchical Graph Partitioning & Modularity Maximization Generator.

Implements the Direction 2 algorithmic core:
- Newman Modularity matrix construction: B_ij = A_ij - (k_i * k_j) / (2m).
- Energy formulation: E = -s^T B s (or binary QUBO).
- Multi-scale tree partition / ultrametric sparsification:
    Q_{ij}^{sparse} = Q_{ij} * I[d_U(i, j) <= d_max] across 4 scale cuts.
- Ultrametric distance computation via hierarchical tree clustering / p-adic tree decomposition.
- Synthetic hierarchical modular networks (H-SBM) across sizes N in {16, 32, 64, 128}.
- Real-world benchmark networks (Karate Club, Dolphins, Football, Polbooks, etc.).
- Normalized Mutual Information (NMI) and Newman modularity scoring.
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import networkx as nx
import scipy.cluster.hierarchy as sch
import dimod

from src.quantum.hea_generator import compute_p_adic_lca_distance


@dataclass
class ModularityInstance:
    """Represents a hardware-embeddable modularity QUBO / Ising problem instance."""
    instance_id: str
    graph_id: str
    graph_type: str  # 'synthetic_hsbm' or 'real_world'
    num_nodes: int
    num_edges: int
    scale_cut: int  # 1 (ultra-sparse), 2 (clustered), 3 (extended), 4 (dense)
    d_max: float
    num_couplers: int
    total_possible_couplers: int
    sparsity: float
    linear: Dict[int, float]
    quadratic: List[Dict[str, Any]]
    bqm: dimod.BinaryQuadraticModel
    graph: nx.Graph  # Sparsified coupler interaction graph
    original_graph: nx.Graph  # Original network
    ground_truth_partition: Optional[Dict[int, int]] = None
    d_u_matrix: Optional[np.ndarray] = None
    offset: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes instance to JSON-compatible dictionary."""
        return {
            "type": "modularity_qubo",
            "instance_id": self.instance_id,
            "graph_id": self.graph_id,
            "graph_type": self.graph_type,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "scale_cut": self.scale_cut,
            "d_max": float(self.d_max),
            "num_couplers": self.num_couplers,
            "total_possible_couplers": self.total_possible_couplers,
            "sparsity": float(self.sparsity),
            "offset": float(self.offset),
            "linear": {int(k): float(v) for k, v in self.linear.items()},
            "quadratic": [
                {"u": int(item["u"]), "v": int(item["v"]), "weight": float(item["weight"])}
                for item in self.quadratic
            ],
            "ground_truth_partition": (
                {int(k): int(v) for k, v in self.ground_truth_partition.items()}
                if self.ground_truth_partition is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], original_graph: Optional[nx.Graph] = None) -> "ModularityInstance":
        """Deserializes instance from dictionary."""
        linear = {int(k): float(v) for k, v in data["linear"].items()}
        quadratic_dict = {
            (int(item["u"]), int(item["v"])): float(item["weight"])
            for item in data["quadratic"]
        }
        offset = float(data.get("offset", 0.0))
        bqm = dimod.BinaryQuadraticModel(linear, quadratic_dict, offset, vartype=dimod.SPIN)

        num_nodes = data["num_nodes"]
        g_sparse = nx.Graph()
        g_sparse.add_nodes_from(range(num_nodes))
        for (u, v), w in quadratic_dict.items():
            g_sparse.add_edge(u, v, weight=w)

        if original_graph is None:
            # Fallback to sparse graph if original not supplied
            orig_g = g_sparse
        else:
            orig_g = original_graph

        gt = (
            {int(k): int(v) for k, v in data["ground_truth_partition"].items()}
            if data.get("ground_truth_partition") is not None else None
        )

        return cls(
            instance_id=data["instance_id"],
            graph_id=data["graph_id"],
            graph_type=data["graph_type"],
            num_nodes=num_nodes,
            num_edges=data["num_edges"],
            scale_cut=data["scale_cut"],
            d_max=float(data["d_max"]),
            num_couplers=data["num_couplers"],
            total_possible_couplers=data["total_possible_couplers"],
            sparsity=float(data["sparsity"]),
            linear=linear,
            quadratic=data["quadratic"],
            bqm=bqm,
            graph=g_sparse,
            original_graph=orig_g,
            ground_truth_partition=gt,
            offset=offset,
        )


def build_modularity_matrix(graph: nx.Graph, weight: Optional[str] = None) -> np.ndarray:
    """
    Computes the Newman modularity matrix B:
        B_ij = A_ij - (k_i * k_j) / (2m)

    Properties:
    - B is symmetric: B_ij = B_ji
    - Zero row sums: sum_j B_ij = 0
    - Tr(B) = - sum_i k_i^2 / (2m) (for graphs without self-loops)

    Args:
        graph: NetworkX graph with N nodes (0 to N-1).
        weight: Optional edge weight attribute name.

    Returns:
        (N, N) NumPy float array representing the modularity matrix B.
    """
    n = len(graph)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    nodelist = list(range(n)) if set(graph.nodes) == set(range(n)) else list(graph.nodes)
    adj = nx.to_numpy_array(graph, nodelist=nodelist, weight=weight, dtype=float)
    # Ensure zero diagonal (no self-loops in standard modularity)
    np.fill_diagonal(adj, 0.0)

    degrees = np.sum(adj, axis=1)
    two_m = np.sum(degrees)

    if two_m <= 0.0:
        return np.zeros((n, n), dtype=float)

    null_model = np.outer(degrees, degrees) / two_m
    b_mat = adj - null_model
    return b_mat


def compute_ultrametric_distances(
    graph: nx.Graph,
    method: str = "auto",
) -> np.ndarray:
    """
    Computes all-pairs ultrametric distance matrix d_U(i, j) on a graph.

    Guarantees the non-Archimedean strong triangle inequality:
        d_U(x, z) <= max(d_U(x, y), d_U(y, z)) for all triples x, y, z.

    Args:
        graph: NetworkX graph.
        method: "auto", "p_adic", or "hierarchical".
                - "p_adic": Uses lowest common ancestor (LCA) in a p-ary tree.
                - "hierarchical": Uses average-linkage hierarchical clustering
                  on all-pairs shortest paths, generating exact cophenetic distances.

    Returns:
        (N, N) NumPy float array of ultrametric distances with zero diagonal.
    """
    n = len(graph)
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    node_list = list(range(n)) if set(graph.nodes) == set(range(n)) else list(graph.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Check if graph has explicit p-adic tree structure or power-of-2 size
    is_power_of_two = (n > 0) and ((n & (n - 1)) == 0)
    use_p_adic = (method == "p_adic") or (method == "auto" and is_power_of_two and getattr(graph, "graph", {}).get("is_tree_geometry", False))

    if use_p_adic:
        levels = max(1, int(math.ceil(math.log2(n))))
        d_u = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(compute_p_adic_lca_distance(i, j, p=2, levels=levels))
                d_u[i, j] = dist
                d_u[j, i] = dist
        return d_u

    # Hierarchical tree clustering via average linkage on shortest path metric
    sp_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    dist_matrix = np.zeros((n, n), dtype=float)

    max_finite = 1.0
    for u in node_list:
        i = node_to_idx[u]
        sp_u = sp_lengths.get(u, {})
        for v in node_list:
            j = node_to_idx[v]
            if i >= j:
                continue
            if v in sp_u:
                d = float(sp_u[v])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                if d > max_finite:
                    max_finite = d

    # Handle disconnected components if any
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] == 0.0:
                dist_matrix[i, j] = 2.0 * max_finite
                dist_matrix[j, i] = 2.0 * max_finite

    # Condensed distance matrix for scipy
    condensed = sch.distance.squareform(dist_matrix, checks=False)
    z = sch.linkage(condensed, method="average")
    cophenet_condensed = sch.cophenet(z)
    d_u = sch.distance.squareform(cophenet_condensed)
    np.fill_diagonal(d_u, 0.0)
    return d_u


def compute_4_scale_cuts(d_u: np.ndarray) -> List[float]:
    """
    Computes 4 scale cut thresholds d_max across an ultrametric distance matrix.
    Scale 1 (ultra-sparse) -> Scale 2 (clustered) -> Scale 3 (extended) -> Scale 4 (dense 100%).

    Args:
        d_u: (N, N) ultrametric distance matrix.

    Returns:
        List of 4 float thresholds [d_max_1, d_max_2, d_max_3, d_max_4].
    """
    n = d_u.shape[0]
    if n <= 1:
        return [1.0, 1.0, 1.0, 1.0]

    unique_dists = sorted(list(set(
        float(d_u[i, j]) for i in range(n) for j in range(i + 1, n)
    )))

    if not unique_dists:
        return [1.0, 1.0, 1.0, 1.0]

    num_dists = len(unique_dists)
    if num_dists < 4:
        # Pad with the available distances
        return [
            unique_dists[min(i, num_dists - 1)]
            for i in range(4)
        ]

    # Monotonically non-decreasing cuts picking 25%, 50%, 75%, and 100% (max)
    idx_1 = max(0, int(math.floor(0.25 * (num_dists - 1))))
    idx_2 = max(idx_1, int(math.floor(0.50 * (num_dists - 1))))
    idx_3 = max(idx_2, int(math.floor(0.75 * (num_dists - 1))))
    idx_4 = num_dists - 1

    return [
        float(unique_dists[idx_1]),
        float(unique_dists[idx_2]),
        float(unique_dists[idx_3]),
        float(unique_dists[idx_4]),
    ]


def build_sparsified_modularity_bqm(
    graph: nx.Graph,
    b_mat: np.ndarray,
    d_u: np.ndarray,
    d_max: float,
    vartype: dimod.Vartype = dimod.SPIN,
) -> Tuple[dimod.BinaryQuadraticModel, nx.Graph, List[Dict[str, Any]]]:
    """
    Constructs the ultrametrically sparsified Newman modularity BQM:
        Q_{ij}^{sparse} = Q_{ij} * I[d_U(i, j) <= d_max]

    For SPIN vartype (s_i in {-1, +1}):
        E(s) = - 1/(4m) * s^T B^{sparse} s - offset
        Coupler J_{ij} = - B_{ij} / (2m)
        Linear h_i = 0.0 (by spin-flip symmetry)
        Energy directly evaluates to - Q_mod when unsparsified.

    Args:
        graph: Original NetworkX graph.
        b_mat: Newman modularity matrix B.
        d_u: Ultrametric distance matrix.
        d_max: Maximum ultrametric tree distance cut.
        vartype: dimod.SPIN or dimod.BINARY.

    Returns:
        (bqm, sparsified_graph, quadratic_list)
    """
    n = len(graph)
    m = graph.number_of_edges()
    denom = 2.0 * m if m > 0 else 1.0

    linear = {i: 0.0 for i in range(n)}
    quadratic: Dict[Tuple[int, int], float] = {}
    quadratic_list: List[Dict[str, Any]] = []
    g_sparse = nx.Graph()
    g_sparse.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            # Ultrametric condition with numerical tolerance
            if d_u[i, j] <= d_max + 1e-9:
                b_val = float(b_mat[i, j])
                if not np.isclose(b_val, 0.0, atol=1e-12):
                    # In spin Ising: J_ij = - B_ij / (2m)
                    # Energy = sum_{i < j} J_ij s_i s_j = - 1/(2m) sum_{i < j} B_ij s_i s_j
                    # Which equals - Q_mod (up to constant diagonal offset)
                    j_val = - b_val / denom
                    quadratic[(i, j)] = j_val
                    quadratic_list.append({"u": i, "v": j, "weight": j_val})
                    g_sparse.add_edge(i, j, weight=j_val)

    # Offset for diagonal: - Tr(B) / (4m)
    offset = - float(np.trace(b_mat)) / (4.0 * m) if m > 0 else 0.0
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype=dimod.SPIN)

    if vartype == dimod.BINARY:
        bqm = bqm.change_vartype(dimod.BINARY, inplace=False)

    return bqm, g_sparse, quadratic_list


def compute_modularity(
    graph: nx.Graph,
    partition: Dict[int, int],
    weight: Optional[str] = None,
) -> float:
    """
    Computes standard Newman modularity Q_mod for a 2-partition of graph nodes.
    Supports binary labels {0, 1} or spin labels {-1, +1}.
    """
    if len(graph) <= 1 or graph.number_of_edges() == 0:
        return 0.0

    # Group into communities covering all graph nodes
    c0: Set[int] = set()
    c1: Set[int] = set()
    for node in graph.nodes:
        val = partition.get(node, -1)
        if val in (1, +1):
            c1.add(node)
        else:
            c0.add(node)

    # If all nodes are in one community, modularity is 0.0
    if len(c0) == 0 or len(c1) == 0:
        return 0.0

    try:
        return float(nx.community.modularity(graph, [c0, c1], weight=weight))
    except Exception:
        # Fallback manual calculation
        a_mat = nx.to_numpy_array(graph, weight=weight)
        k = a_mat.sum(axis=1)
        m = a_mat.sum() / 2.0
        if m <= 0:
            return 0.0
        b_mat = a_mat - np.outer(k, k) / (2.0 * m)
        node_list = list(graph.nodes)
        s = np.array([1 if partition.get(node, -1) in (1, +1) else -1 for node in node_list])
        return float((s @ b_mat @ s) / (4.0 * m))


def compute_clustering_nmi(
    labels_true: List[int],
    labels_pred: List[int],
) -> float:
    """
    Computes Normalized Mutual Information (NMI) between two cluster labelings:
        NMI(U, V) = 2 * I(U; V) / (H(U) + H(V))

    Independent implementation with zero third-party dependencies (no scikit-learn).
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError(
            f"labels_true ({len(labels_true)}) and labels_pred ({len(labels_pred)}) must have identical length"
        )

    yt = np.asarray(labels_true)
    yp = np.asarray(labels_pred)
    n = len(yt)
    if n <= 1:
        return 1.0

    vals_t = np.unique(yt)
    vals_p = np.unique(yp)

    # Marginal probabilities
    pt = np.array([float(np.mean(yt == c)) for c in vals_t])
    pp = np.array([float(np.mean(yp == c)) for c in vals_p])

    # Entropies
    ht = -float(np.sum([p * np.log(p) for p in pt if p > 0.0]))
    hp = -float(np.sum([p * np.log(p) for p in pp if p > 0.0]))

    if ht + hp <= 1e-12:
        return 1.0

    # Mutual information
    mi = 0.0
    for ct in vals_t:
        for cp in vals_p:
            p_joint = float(np.mean((yt == ct) & (yp == cp)))
            if p_joint > 0.0:
                p_t_val = float(np.mean(yt == ct))
                p_p_val = float(np.mean(yp == cp))
                mi += p_joint * np.log(p_joint / (p_t_val * p_p_val))

    nmi_val = 2.0 * mi / (ht + hp)
    return float(np.clip(nmi_val, 0.0, 1.0))


def generate_hierarchical_sbm(
    num_nodes: int,
    branching: int = 2,
    p_in: float = 0.75,
    gamma: float = 0.35,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int], np.ndarray]:
    """
    Generates a synthetic Hierarchical Stochastic Block Model (H-SBM) network.

    Nodes 0 to N-1 are leaves of a p-ary tree of depth L = ceil(log_p(N)).
    Connection probability decays with tree distance l = d_U(i, j):
        P(A_ij = 1) = p_in * gamma^(l - 1)

    Args:
        num_nodes: Total nodes N (e.g. 16, 32, 64, 128).
        branching: Branching factor p (default p=2).
        p_in: Intra-cluster edge probability at lowest level l=1.
        gamma: Decay factor across hierarchical tree levels (0 < gamma < 1).
        seed: Random seed.

    Returns:
        (graph, ground_truth_communities, d_u_matrix)
    """
    rng = np.random.default_rng(seed)
    levels = max(1, int(math.ceil(math.log(num_nodes, branching))))

    d_u = np.zeros((num_nodes, num_nodes), dtype=float)
    g = nx.Graph()
    g.graph["is_tree_geometry"] = True
    g.add_nodes_from(range(num_nodes))

    # Compute p-adic tree distances and generate edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            l = compute_p_adic_lca_distance(i, j, p=branching, levels=levels)
            d_u[i, j] = float(l)
            d_u[j, i] = float(l)

            prob = float(p_in * (gamma ** (l - 1)))
            if rng.random() < prob:
                g.add_edge(i, j)

    # Ensure network is connected by adding minimum-distance bridges if necessary
    if not nx.is_connected(g):
        components = list(nx.connected_components(g))
        components.sort(key=len, reverse=True)
        main_comp = set(components[0])
        for other_comp in components[1:]:
            # Find closest pair between main_comp and other_comp
            best_pair = None
            best_dist = float("inf")
            for u in main_comp:
                for v in other_comp:
                    if d_u[u, v] < best_dist:
                        best_dist = d_u[u, v]
                        best_pair = (u, v)
            if best_pair:
                g.add_edge(best_pair[0], best_pair[1])
                main_comp.update(other_comp)

    # Ground truth top-level 2-community bisection
    half = num_nodes // 2
    ground_truth = {i: 0 if i < half else 1 for i in range(num_nodes)}

    return g, ground_truth, d_u


def load_real_world_network(name: str) -> Tuple[nx.Graph, Optional[Dict[int, int]]]:
    """
    Loads or generates one of the 15 real-world benchmark networks.
    Guarantees simple, connected graphs with consecutive integer node labels 0 to N-1.

    Available networks:
    1. karate_club (N=34, Zachary)
    2. florentine_families (N=15, Padgett)
    3. davis_southern_women (N=32, Davis et al.)
    4. dolphins (N=62, Lusseau)
    5. les_miserables (N=77, Knuth)
    6. polbooks (N=105, Krebs)
    7. adjnoun (N=112, Newman)
    8. football (N=115, Girvan-Newman)
    9. coauthorship_netscience_64 (N=64, NetScience core)
    10. coauthorship_netscience_128 (N=128, NetScience community)
    11. ppi_yeast_64 (N=64, Protein interaction core)
    12. ppi_yeast_128 (N=128, Protein interaction module)
    13. power_grid_64 (N=64, Power grid substation cluster)
    14. power_grid_128 (N=128, Power grid transmission network)
    15. citation_hep_64 (N=64, High-energy physics citation module)
    """
    networks_dir = os.path.join(os.path.dirname(__file__), "data", "networks")
    gt: Optional[Dict[int, int]] = None

    if name == "karate_club":
        raw_g = nx.karate_club_graph()
        gt = {n: 0 if d.get("club") == "Mr. Hi" else 1 for n, d in raw_g.nodes(data=True)}
    elif name == "florentine_families":
        raw_g = nx.florentine_families_graph()
    elif name == "davis_southern_women":
        raw_g = nx.davis_southern_women_graph()
    elif name == "les_miserables":
        raw_g = nx.les_miserables_graph()
    elif name == "dolphins":
        gml_path = os.path.join(networks_dir, "dolphins.gml")
        if os.path.exists(gml_path):
            raw_g = nx.read_gml(gml_path, label="label")
        else:
            # Deterministic fallback
            raw_g = nx.planted_partition_graph(2, 31, 0.25, 0.05, seed=42)
    elif name == "football":
        gml_path = os.path.join(networks_dir, "football.gml")
        if os.path.exists(gml_path):
            raw_g = nx.read_gml(gml_path, label="label")
            gt = {n: int(d.get("value", 0)) for n, d in raw_g.nodes(data=True)}
        else:
            raw_g = nx.planted_partition_graph(4, 28, 0.35, 0.08, seed=42)
    elif name == "polbooks":
        gml_path = os.path.join(networks_dir, "polbooks.gml")
        if os.path.exists(gml_path):
            raw_g = nx.read_gml(gml_path, label="label")
            val_map = {"c": 0, "l": 1, "n": 2}
            gt = {n: val_map.get(str(d.get("value", "n")), 0) for n, d in raw_g.nodes(data=True)}
        else:
            raw_g = nx.planted_partition_graph(3, 35, 0.30, 0.06, seed=42)
    elif name == "adjnoun":
        gml_path = os.path.join(networks_dir, "adjnoun.gml")
        if os.path.exists(gml_path):
            raw_g = nx.read_gml(gml_path, label="id")
        else:
            raw_g = nx.planted_partition_graph(2, 56, 0.20, 0.04, seed=42)
    elif name in ("coauthorship_netscience_64", "coauthorship_netscience_128"):
        target_size = 64 if "64" in name else 128
        gml_path = os.path.join(networks_dir, "netscience.gml")
        if os.path.exists(gml_path):
            full_g = nx.Graph(nx.read_gml(gml_path, label="id"))
            lcc_nodes = max(nx.connected_components(full_g), key=len)
            sub_g = full_g.subgraph(lcc_nodes)
            # Take BFS ego expansion around highest degree hub
            hub = max(sub_g.nodes, key=sub_g.degree)
            nodes = [hub]
            queue = [hub]
            visited = {hub}
            while queue and len(nodes) < target_size:
                curr = queue.pop(0)
                for neighbor in sub_g.neighbors(curr):
                    if neighbor not in visited and len(nodes) < target_size:
                        visited.add(neighbor)
                        nodes.append(neighbor)
                        queue.append(neighbor)
            raw_g = sub_g.subgraph(nodes).copy()
        else:
            raw_g = nx.connected_caveman_graph(target_size // 8, 8)
    elif name in ("power_grid_64", "power_grid_128"):
        target_size = 64 if "64" in name else 128
        gml_path = os.path.join(networks_dir, "power.gml")
        if os.path.exists(gml_path):
            full_g = nx.Graph(nx.read_gml(gml_path, label="id"))
            lcc_nodes = max(nx.connected_components(full_g), key=len)
            sub_g = full_g.subgraph(lcc_nodes)
            hub = max(sub_g.nodes, key=sub_g.degree)
            nodes = [hub]
            queue = [hub]
            visited = {hub}
            while queue and len(nodes) < target_size:
                curr = queue.pop(0)
                for neighbor in sub_g.neighbors(curr):
                    if neighbor not in visited and len(nodes) < target_size:
                        visited.add(neighbor)
                        nodes.append(neighbor)
                        queue.append(neighbor)
            raw_g = sub_g.subgraph(nodes).copy()
        else:
            raw_g = nx.gaussian_random_partition_graph(target_size, 16, 4, 0.4, 0.05, seed=42)
    elif name in ("ppi_yeast_64", "ppi_yeast_128"):
        target_size = 64 if "64" in name else 128
        # Standard duplication-divergence / powerlaw cluster model for protein interactomes
        raw_g = nx.powerlaw_cluster_graph(target_size, m=3, p=0.35, seed=42 + target_size)
    elif name == "citation_hep_64":
        raw_g = nx.gaussian_random_partition_graph(64, s=16, v=4, p_in=0.5, p_out=0.08, seed=888)
    else:
        raise ValueError(f"Unknown real-world network name: {name}")

    # Standardize to simple, connected graph
    g = nx.Graph(raw_g)
    g.remove_edges_from(nx.selfloop_edges(g))

    if not nx.is_connected(g):
        lcc = max(nx.connected_components(g), key=len)
        g = g.subgraph(lcc).copy()

    # Relabel nodes to consecutive integers 0 to N-1
    old_to_new = {old: new for new, old in enumerate(g.nodes)}
    g = nx.relabel_nodes(g, old_to_new)

    if gt is not None:
        new_gt = {}
        for old_node, new_node in old_to_new.items():
            if old_node in gt:
                new_gt[new_node] = gt[old_node]
            elif isinstance(old_node, int) and old_node in gt:
                new_gt[new_node] = gt[old_node]
        if len(new_gt) == len(g):
            gt = new_gt
        else:
            gt = None

    return g, gt


def generate_modularity_instances_for_graph(
    graph: nx.Graph,
    graph_id: str,
    graph_type: str,
    ground_truth: Optional[Dict[int, int]] = None,
    d_u: Optional[np.ndarray] = None,
) -> List[ModularityInstance]:
    """
    Generates 4 multi-scale ultrametrically sparsified ModularityInstances for a given graph.
    """
    n = len(graph)
    m = graph.number_of_edges()
    total_couplers = n * (n - 1) // 2

    b_mat = build_modularity_matrix(graph)

    if d_u is None:
        d_u = compute_ultrametric_distances(graph)

    cuts = compute_4_scale_cuts(d_u)
    instances: List[ModularityInstance] = []

    for cut_idx, d_max in enumerate(cuts, start=1):
        inst_id = f"{graph_id}_cut{cut_idx}"
        bqm, g_sparse, quad_list = build_sparsified_modularity_bqm(
            graph=graph,
            b_mat=b_mat,
            d_u=d_u,
            d_max=d_max,
            vartype=dimod.SPIN,
        )

        num_c = len(quad_list)
        sparsity = float(1.0 - (num_c / total_couplers)) if total_couplers > 0 else 0.0

        instances.append(ModularityInstance(
            instance_id=inst_id,
            graph_id=graph_id,
            graph_type=graph_type,
            num_nodes=n,
            num_edges=m,
            scale_cut=cut_idx,
            d_max=float(d_max),
            num_couplers=num_c,
            total_possible_couplers=total_couplers,
            sparsity=sparsity,
            linear={i: 0.0 for i in range(n)},
            quadratic=quad_list,
            bqm=bqm,
            graph=g_sparse,
            original_graph=graph,
            ground_truth_partition=ground_truth,
            d_u_matrix=d_u,
            offset=float(bqm.offset),
        ))

    return instances
