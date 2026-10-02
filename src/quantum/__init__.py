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
from .modularity_generator import (
    ModularityInstance,
    build_modularity_matrix,
    compute_ultrametric_distances,
    compute_4_scale_cuts,
    build_sparsified_modularity_bqm,
    compute_modularity,
    compute_clustering_nmi,
    generate_hierarchical_sbm,
    load_real_world_network,
    generate_modularity_instances_for_graph,
)
from .direction2_suite import (
    Direction2TestSuite,
    Direction2InstanceMetadata,
    run_direction2_dry_run,
)
from .direction3_qbm import (
    HardwareAwareQBM,
    generate_pegasus_tree_mask,
    verify_pegasus_embedding,
    generate_bars_and_stripes_dataset,
    generate_downscaled_digits_dataset,
    generate_dyck_sequence_dataset,
    compute_reconstruction_error,
    compute_total_variation_distance,
    evaluate_qbm,
)
from .direction3_suite import (
    Direction3TestSuite,
    QBMModelMetadata,
    run_dry_run_simulation as run_direction3_dry_run,
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
    "ModularityInstance",
    "build_modularity_matrix",
    "compute_ultrametric_distances",
    "compute_4_scale_cuts",
    "build_sparsified_modularity_bqm",
    "compute_modularity",
    "compute_clustering_nmi",
    "generate_hierarchical_sbm",
    "load_real_world_network",
    "generate_modularity_instances_for_graph",
    "Direction2TestSuite",
    "Direction2InstanceMetadata",
    "run_direction2_dry_run",
    "HardwareAwareQBM",
    "generate_pegasus_tree_mask",
    "verify_pegasus_embedding",
    "generate_bars_and_stripes_dataset",
    "generate_downscaled_digits_dataset",
    "generate_dyck_sequence_dataset",
    "compute_reconstruction_error",
    "compute_total_variation_distance",
    "evaluate_qbm",
    "Direction3TestSuite",
    "QBMModelMetadata",
    "run_direction3_dry_run",
]
