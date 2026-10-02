"""
Project Q-Ultrametric: Hardware-Embeddable Differentiable QUBO & Quantum Annealing Modules.
"""

from .hea_generator import (
    generate_hea_spin_glass,
    compute_p_adic_lca_distance,
    HEAInstance,
)
from .planted_generator import (
    generate_planted_frustrated_loops,
    PlantedLoopInstance,
)
from .embeddings import (
    get_pegasus_target_graph,
    get_zephyr_target_graph,
    embed_graph_onto_pegasus,
    embed_clique_onto_pegasus,
    evaluate_embedding_quality,
    compare_ultrametric_vs_clique,
)
from .solvers import (
    solve_scip_exact,
    solve_locally_classical,
    compute_time_to_solution,
)
from .direction1_suite import (
    Direction1TestSuite,
    InstanceMetadata,
)
from .leap_runner import (
    run_dry_run_simulation,
    run_live_qpu_submission,
)

__all__ = [
    "generate_hea_spin_glass",
    "compute_p_adic_lca_distance",
    "HEAInstance",
    "generate_planted_frustrated_loops",
    "PlantedLoopInstance",
    "get_pegasus_target_graph",
    "get_zephyr_target_graph",
    "embed_graph_onto_pegasus",
    "embed_clique_onto_pegasus",
    "evaluate_embedding_quality",
    "compare_ultrametric_vs_clique",
    "solve_scip_exact",
    "solve_locally_classical",
    "compute_time_to_solution",
    "Direction1TestSuite",
    "InstanceMetadata",
    "run_dry_run_simulation",
    "run_live_qpu_submission",
]
