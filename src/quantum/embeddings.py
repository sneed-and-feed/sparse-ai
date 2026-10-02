"""
Hardware Embedding Analysis for D-Wave Pegasus and Zephyr Architectures.

Provides minor-embedding evaluation, comparing dense complete graphs against
sparse ultrametric tree graphs on Pegasus P_M and Zephyr Z_M hardware fabrics.
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import numpy as np
import minorminer
import minorminer.busclique as busclique


def get_pegasus_target_graph(m: int = 16) -> nx.Graph:
    """
    Constructs a Pegasus target graph P_M (default M=16, Advantage scale ~5640 qubits).
    Uses modern dwave.graphs.topologies or dwave_networkx.
    """
    try:
        from dwave.graphs.topologies import pegasus_graph
        return pegasus_graph(m)
    except ImportError:
        pass

    try:
        import dwave_networkx as dnx
        return dnx.pegasus_graph(m)
    except ImportError:
        pass

    raise RuntimeError("Neither dwave.graphs nor dwave_networkx is installed.")


def get_zephyr_target_graph(m: int = 15, t: int = 4) -> nx.Graph:
    """
    Constructs a Zephyr target graph Z_{M, t} (Advantage2 prototype scale ~7440 qubits).
    """
    try:
        from dwave.graphs.topologies import zephyr_graph
        return zephyr_graph(m, t)
    except ImportError:
        pass

    try:
        import dwave_networkx as dnx
        return dnx.zephyr_graph(m, t)
    except ImportError:
        pass

    raise RuntimeError("Neither dwave.graphs nor dwave_networkx is installed.")


def embed_graph_onto_pegasus(
    source_graph: nx.Graph,
    m: int = 16,
    target_graph: Optional[nx.Graph] = None,
    timeout_sec: float = 30.0,
    random_seed: int = 42,
) -> Dict[Any, List[int]]:
    """
    Finds a minor embedding of source_graph onto Pegasus P_M using minorminer.

    Args:
        source_graph: Logical problem graph.
        m: Pegasus grid dimension (M=16 corresponds to 5,000+ qubits; M=4 for fast tests).
        target_graph: Optional pre-constructed target graph.
        timeout_sec: Maximum solver timeout in seconds.
        random_seed: Random seed for minorminer heuristic search.

    Returns:
        Embedding mapping logical nodes to list of physical qubit indices.
    """
    if target_graph is None:
        target_graph = get_pegasus_target_graph(m)

    embedding = minorminer.find_embedding(
        source_graph,
        target_graph,
        timeout=timeout_sec,
        random_seed=random_seed,
        tries=5,
    )
    return embedding


def embed_clique_onto_pegasus(
    num_nodes: int,
    m: int = 16,
    target_graph: Optional[nx.Graph] = None,
) -> Optional[Dict[int, List[int]]]:
    """
    Embeds a complete graph K_N onto Pegasus P_M using minorminer.busclique.
    Serves as the theoretical dense baseline showing quadratic physical qubit scaling.
    """
    if target_graph is None:
        target_graph = get_pegasus_target_graph(m)

    try:
        emb = busclique.find_clique_embedding(num_nodes, target_graph)
        if emb:
            return {int(k): [int(q) for q in v] for k, v in emb.items()}
    except Exception:
        pass

    # Fallback to general minorminer
    kn = nx.complete_graph(num_nodes)
    return embed_graph_onto_pegasus(kn, m=m, target_graph=target_graph, timeout_sec=15.0)


def evaluate_embedding_quality(
    embedding: Optional[Dict[Any, List[int]]],
    source_graph: nx.Graph,
) -> Dict[str, Any]:
    """
    Computes rigorous physical metrics for a minor embedding.

    Returns:
        Dictionary with:
        - num_logical: Number of logical variables.
        - num_physical: Total physical qubits consumed.
        - dilation_ratio: Qubit expansion ratio N_phys / N_log.
        - max_chain_length: Length of longest chain (L_max).
        - mean_chain_length: Average chain length across logical nodes.
        - dynamic_range_factor: Estimated 1 / sqrt(L_max) dynamic range scaling factor.
        - is_valid: True if every logical node has at least one physical qubit.
    """
    num_logical = len(source_graph.nodes)
    if not embedding:
        return {
            "num_logical": num_logical,
            "num_physical": 0,
            "dilation_ratio": 0.0,
            "max_chain_length": 0,
            "mean_chain_length": 0.0,
            "chain_length_std": 0.0,
            "dynamic_range_factor": 0.0,
            "is_valid": False,
        }

    chain_lengths = [len(chain) for chain in embedding.values()]
    total_physical = sum(chain_lengths)
    max_l = int(max(chain_lengths)) if chain_lengths else 0
    mean_l = float(np.mean(chain_lengths)) if chain_lengths else 0.0
    dyn_range = 1.0 / np.sqrt(max_l) if max_l > 0 else 0.0

    return {
        "num_logical": num_logical,
        "num_physical": total_physical,
        "dilation_ratio": float(total_physical / max(1, num_logical)),
        "max_chain_length": max_l,
        "mean_chain_length": mean_l,
        "chain_length_std": float(np.std(chain_lengths)) if chain_lengths else 0.0,
        "dynamic_range_factor": float(dyn_range),
        "is_valid": len(embedding) == num_logical and all(l > 0 for l in chain_lengths),
    }


def compare_ultrametric_vs_clique(
    source_graph: nx.Graph,
    m: int = 16,
    target_graph: Optional[nx.Graph] = None,
) -> Dict[str, Any]:
    """
    Directly compares embedding quality of an ultrametric graph vs. complete graph K_N.
    Demonstrates the quadratic physical qubit reduction and chain length mitigation.
    """
    if target_graph is None:
        target_graph = get_pegasus_target_graph(m)

    n = len(source_graph.nodes)
    emb_ultra = embed_graph_onto_pegasus(source_graph, m=m, target_graph=target_graph, timeout_sec=20.0)
    emb_clique = embed_clique_onto_pegasus(n, m=m, target_graph=target_graph)

    metrics_ultra = evaluate_embedding_quality(emb_ultra, source_graph)
    kn_graph = nx.complete_graph(n)
    metrics_clique = evaluate_embedding_quality(emb_clique, kn_graph)

    qubit_reduction_pct = 0.0
    if metrics_clique["num_physical"] > 0:
        qubit_reduction_pct = 100.0 * (
            1.0 - metrics_ultra["num_physical"] / metrics_clique["num_physical"]
        )

    return {
        "num_spins": n,
        "ultrametric": metrics_ultra,
        "clique": metrics_clique,
        "qubit_reduction_percent": float(qubit_reduction_pct),
        "chain_length_ratio": (
            float(metrics_ultra["max_chain_length"] / max(1, metrics_clique["max_chain_length"]))
            if metrics_clique["max_chain_length"] > 0 else 1.0
        ),
    }
