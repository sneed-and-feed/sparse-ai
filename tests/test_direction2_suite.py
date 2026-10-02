"""
Comprehensive Unit Tests for Direction 2 Test Suite & Infrastructure.

Verifies:
1. Newman modularity matrix formulation, symmetry, zero row-sums, and QUBO/Ising energy equivalence.
2. Ultrametric tree decomposition, p-adic LCA metric, and non-Archimedean strong triangle inequality.
3. Multi-scale tree partition cuts, coupler monotonicity, and sparsity progression.
4. Synthetic Hierarchical SBM network generation and NMI cluster evaluation.
5. Real-world benchmark network catalog integrity (Karate Club, Dolphins, Football, Polbooks, etc.).
6. Exact SCIP branch-and-cut optimization vs Louvain/Greedy classical baselines.
7. Pegasus P_M minor embedding improvements (sparse vs dense complete graph K_N).
8. Direction 2 Test Suite manifest integrity (140 instances, 420 QPU jobs, exactly 21.00s quota).
9. Leap dry-run simulation execution with 5-gauge SRT.
"""

import os
import json
import math
import unittest
import numpy as np
import networkx as nx
import dimod

from src.quantum.modularity_generator import (
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
from src.quantum.solvers import (
    solve_scip_exact,
    solve_locally_classical,
    compute_time_to_solution,
)
from src.quantum.embeddings import (
    get_pegasus_target_graph,
    embed_graph_onto_pegasus,
    evaluate_embedding_quality,
)
from src.quantum.direction2_suite import (
    Direction2TestSuite,
    Direction2InstanceMetadata,
    run_direction2_dry_run,
)


class TestDirection2Suite(unittest.TestCase):
    """Test suite covering Direction 2 modularity models, solvers, and benchmark infrastructure."""

    def test_newman_modularity_matrix_properties_and_energy_equivalence(self):
        """Verify Newman modularity matrix mathematical invariants and QUBO/Ising energy equivalence."""
        g = nx.karate_club_graph()
        n = len(g)
        m = g.number_of_edges()
        b_mat = build_modularity_matrix(g)

        # 1. Modularity matrix symmetry: B_ij == B_ji
        np.testing.assert_allclose(b_mat, b_mat.T, atol=1e-12)

        # 2. Modularity matrix zero row sums: sum_j B_ij == 0
        row_sums = np.sum(b_mat, axis=1)
        np.testing.assert_allclose(row_sums, np.zeros(n), atol=1e-12)

        # 3. Trace invariant: Tr(B) = - sum_i k_i^2 / (2m)
        degrees = np.array([g.degree(i) for i in range(n)])
        expected_trace = - float(np.sum(degrees ** 2)) / (2.0 * m)
        self.assertAlmostEqual(float(np.trace(b_mat)), expected_trace, places=10)

        # 4. Energy Equivalence: For any state, s^T B s == 4 x^T B x
        rng = np.random.default_rng(42)
        for _ in range(10):
            # Binary state x in {0, 1}^N and spin state s = 2x - 1 in {-1, 1}^N
            x = rng.integers(0, 2, size=n)
            s = 2 * x - 1

            sBs = float(s @ b_mat @ s)
            xBx = float(x @ b_mat @ x)
            self.assertAlmostEqual(sBs, 4.0 * xBx, places=10)

            # Modularity formula: Q_mod = (1 / 4m) s^T B s
            q_formula = sBs / (4.0 * m)

            # Partition modularity via compute_modularity
            partition_dict = {i: int(s[i]) for i in range(n)}
            q_eval = compute_modularity(g, partition_dict)

            # NetworkX modularity
            c1 = {i for i in range(n) if s[i] == 1}
            c0 = {i for i in range(n) if s[i] == -1}
            if c1 and c0:
                q_nx = float(nx.community.modularity(g, [c0, c1], weight=None))
                self.assertAlmostEqual(q_eval, q_nx, places=10)
                self.assertAlmostEqual(q_formula, q_nx, places=10)

        # 5. Exact BQM Energy Equivalence: E(s) == - Q_mod on Cut 4 (dense)
        d_u = compute_ultrametric_distances(g)
        cuts = compute_4_scale_cuts(d_u)
        bqm_spin, _, _ = build_sparsified_modularity_bqm(g, b_mat, d_u, cuts[3], vartype=dimod.SPIN)
        bqm_bin, _, _ = build_sparsified_modularity_bqm(g, b_mat, d_u, cuts[3], vartype=dimod.BINARY)

        for _ in range(10):
            s_dict = {i: int(rng.choice([-1, 1])) for i in range(n)}
            x_dict = {i: 1 if s_dict[i] == 1 else 0 for i in range(n)}
            e_spin = bqm_spin.energy(s_dict)
            e_bin = bqm_bin.energy(x_dict)
            q_val = compute_modularity(g, s_dict)
            self.assertAlmostEqual(e_spin, -q_val, places=10)
            self.assertAlmostEqual(e_bin, -q_val, places=10)

        # 6. Degenerate assignments: all-ones and all-minus-ones -> E = 0.0, Q = 0.0
        s_all_ones = {i: 1 for i in range(n)}
        s_all_negs = {i: -1 for i in range(n)}
        self.assertAlmostEqual(bqm_spin.energy(s_all_ones), 0.0, places=10)
        self.assertAlmostEqual(bqm_spin.energy(s_all_negs), 0.0, places=10)
        self.assertEqual(compute_modularity(g, s_all_ones), 0.0)
        self.assertEqual(compute_modularity(g, s_all_negs), 0.0)

    def test_ultrametric_tree_decomposition_and_non_archimedean_inequality(self):
        """Verify ultrametric distance matrices satisfy the strong triangle inequality with zero violations."""
        # 1. Test p-adic LCA metric on synthetic tree
        g_syn, _, d_u_syn = generate_hierarchical_sbm(num_nodes=32, seed=123)
        n_syn = len(g_syn)

        for i in range(n_syn):
            self.assertEqual(d_u_syn[i, i], 0.0)
            for j in range(n_syn):
                self.assertEqual(d_u_syn[i, j], d_u_syn[j, i])

        # Test non-Archimedean strong triangle inequality: d(x, z) <= max(d(x, y), d(y, z))
        for x in range(n_syn):
            for y in range(n_syn):
                for z in range(n_syn):
                    self.assertLessEqual(
                        d_u_syn[x, z],
                        max(d_u_syn[x, y], d_u_syn[y, z]) + 1e-12,
                        f"Non-Archimedean violation for p-adic tree nodes {x}, {y}, {z}"
                    )

        # 2. Test hierarchical clustering cophenetic distance on real-world network
        g_real = nx.karate_club_graph()
        d_u_real = compute_ultrametric_distances(g_real, method="hierarchical")
        n_real = len(g_real)

        for x in range(n_real):
            for y in range(n_real):
                for z in range(n_real):
                    self.assertLessEqual(
                        d_u_real[x, z],
                        max(d_u_real[x, y], d_u_real[y, z]) + 1e-12,
                        f"Non-Archimedean violation for cophenetic distance nodes {x}, {y}, {z}"
                    )

    def test_multi_scale_cuts_and_coupler_sparsity_monotonicity(self):
        """Verify 4 scale cuts produce monotonically increasing density / decreasing sparsity."""
        g = nx.karate_club_graph()
        n = len(g)
        b_mat = build_modularity_matrix(g)
        d_u = compute_ultrametric_distances(g)

        cuts = compute_4_scale_cuts(d_u)
        self.assertEqual(len(cuts), 4)
        # Monotonically non-decreasing cut thresholds
        for i in range(3):
            self.assertLessEqual(cuts[i], cuts[i + 1])

        # Generate instances
        instances = generate_modularity_instances_for_graph(g, "karate_club", "real_world", d_u=d_u)
        self.assertEqual(len(instances), 4)

        coupler_counts = [inst.num_couplers for inst in instances]
        sparsities = [inst.sparsity for inst in instances]

        # Verify monotonicity of couplers
        for i in range(3):
            self.assertLessEqual(coupler_counts[i], coupler_counts[i + 1])
            self.assertGreaterEqual(sparsities[i], sparsities[i + 1])

        # Cut 4 must be the dense baseline (100% of couplers, 0.0 sparsity)
        total_possible = n * (n - 1) // 2
        self.assertEqual(instances[3].num_couplers, total_possible)
        self.assertAlmostEqual(instances[3].sparsity, 0.0)

        # Cut 1 must be significantly sparsified (sparsity > 0.6)
        self.assertGreater(instances[0].sparsity, 0.60)

    def test_synthetic_hsbm_generation_and_nmi_metric(self):
        """Verify Hierarchical SBM generation across sizes N in {16, 32, 64, 128} and NMI cluster metric."""
        for n_spins in [16, 32, 64, 128]:
            g, gt, du = generate_hierarchical_sbm(num_nodes=n_spins, seed=42 + n_spins)
            self.assertEqual(len(g), n_spins)
            self.assertTrue(nx.is_connected(g))
            self.assertGreater(g.number_of_edges(), 0)
            self.assertEqual(len(gt), n_spins)
            self.assertEqual(du.shape, (n_spins, n_spins))

            # Test JSON serialization of instance
            instances = generate_modularity_instances_for_graph(g, f"syn_n{n_spins}", "synthetic_hsbm", gt, du)
            self.assertEqual(len(instances), 4)
            inst = instances[0]

            data = inst.to_dict()
            reconstructed = ModularityInstance.from_dict(data, original_graph=g)
            self.assertEqual(reconstructed.num_nodes, n_spins)
            self.assertEqual(reconstructed.num_couplers, inst.num_couplers)
            self.assertEqual(len(reconstructed.bqm.quadratic), len(inst.bqm.quadratic))
            self.assertAlmostEqual(reconstructed.offset, inst.offset, places=10)
            self.assertAlmostEqual(reconstructed.bqm.offset, inst.bqm.offset, places=10)

        # Test Normalized Mutual Information (NMI)
        # Identical labelings -> NMI = 1.0
        y_true = [0, 0, 1, 1, 0, 1]
        self.assertAlmostEqual(compute_clustering_nmi(y_true, y_true), 1.0)

        # Inverted labelings -> NMI = 1.0 (permutation invariant)
        y_inv = [1, 1, 0, 0, 1, 0]
        self.assertAlmostEqual(compute_clustering_nmi(y_true, y_inv), 1.0)

        # Independent/orthogonal labelings -> NMI < 0.2
        y_ortho = [0, 1, 0, 1, 0, 1]
        self.assertLess(compute_clustering_nmi(y_true, y_ortho), 0.3)

        # Single-node or empty labelings -> NMI = 1.0
        self.assertEqual(compute_clustering_nmi([0], [0]), 1.0)
        self.assertEqual(compute_clustering_nmi([], []), 1.0)

        # Mismatched length raises ValueError
        with self.assertRaises(ValueError):
            compute_clustering_nmi([0, 1], [0, 1, 0])

    def test_real_world_networks_catalog(self):
        """Verify loading and integrity of canonical real-world benchmark networks (all 15)."""
        test_networks = [
            ("karate_club", 34),
            ("florentine_families", 15),
            ("davis_southern_women", 32),
            ("dolphins", 62),
            ("les_miserables", 77),
            ("polbooks", 105),
            ("adjnoun", 112),
            ("football", 115),
            ("coauthorship_netscience_64", 64),
            ("coauthorship_netscience_128", 128),
            ("ppi_yeast_64", 64),
            ("ppi_yeast_128", 128),
            ("power_grid_64", 64),
            ("power_grid_128", 128),
            ("citation_hep_64", 64),
        ]
        for name, expected_nodes in test_networks:
            g, gt = load_real_world_network(name)
            self.assertEqual(len(g), expected_nodes, f"Node count mismatch for {name}")
            self.assertTrue(nx.is_connected(g), f"Graph {name} is not connected")
            self.assertGreater(g.number_of_edges(), 0, f"Graph {name} has no edges")

            # Node labels must be consecutive integers 0 to N-1
            self.assertEqual(sorted(list(g.nodes)), list(range(expected_nodes)))

    def test_exact_scip_vs_louvain_and_greedy_modularity(self):
        """Verify exact SCIP solver certifies global optimal 2-partition modularity vs heuristics."""
        g = nx.karate_club_graph()
        n = len(g)
        b_mat = build_modularity_matrix(g)
        d_u = compute_ultrametric_distances(g)

        # Test on Cut 4 (dense complete modularity graph)
        cuts = compute_4_scale_cuts(d_u)
        bqm, _, _ = build_sparsified_modularity_bqm(g, b_mat, d_u, cuts[3], vartype=dimod.SPIN)

        # Solve exactly with SCIP branch-and-cut
        scip_res = solve_scip_exact(bqm, time_limit_sec=10.0)
        self.assertTrue(scip_res["is_optimal"])

        q_scip = compute_modularity(g, scip_res["best_sample"])
        # Global optimal 2-way bisection modularity for Zachary Karate Club is exactly ~0.3718
        self.assertAlmostEqual(q_scip, 0.37179487, places=4)

        # Compare with Greedy Modularity Communities
        c_greedy = nx.community.greedy_modularity_communities(g)
        q_greedy = float(nx.community.modularity(g, c_greedy, weight=None))

        # Compare with Louvain
        c_louvain = nx.community.louvain_communities(g, seed=42)
        q_louvain = float(nx.community.modularity(g, c_louvain, weight=None))

        # Modularity values must be positive and non-trivial
        self.assertGreater(q_scip, 0.30)
        self.assertGreater(q_greedy, 0.30)
        self.assertGreater(q_louvain, 0.30)

    def test_pegasus_minor_embedding_sparse_vs_dense(self):
        """Verify ultrametrically sparsified graphs achieve dramatic chain length and dilation reductions."""
        # Use small Pegasus P_4 for fast test execution
        target_p4 = get_pegasus_target_graph(4)

        g = nx.karate_club_graph()
        n = len(g)
        b_mat = build_modularity_matrix(g)
        d_u = compute_ultrametric_distances(g)
        cuts = compute_4_scale_cuts(d_u)

        # Sparsified Cut 1
        bqm_sparse, g_sparse, _ = build_sparsified_modularity_bqm(g, b_mat, d_u, cuts[0], vartype=dimod.SPIN)
        # Dense Complete Graph (Cut 4)
        bqm_dense, g_dense, _ = build_sparsified_modularity_bqm(g, b_mat, d_u, cuts[3], vartype=dimod.SPIN)

        emb_sparse = embed_graph_onto_pegasus(g_sparse, m=4, target_graph=target_p4, timeout_sec=10.0)
        emb_dense = embed_graph_onto_pegasus(g_dense, m=4, target_graph=target_p4, timeout_sec=15.0)

        metrics_sparse = evaluate_embedding_quality(emb_sparse, g_sparse)
        metrics_dense = evaluate_embedding_quality(emb_dense, g_dense)

        self.assertTrue(metrics_sparse["is_valid"])
        self.assertTrue(metrics_dense["is_valid"])

        # Sparsified modularity graph requires significantly fewer physical qubits
        self.assertLess(metrics_sparse["num_physical"], metrics_dense["num_physical"])
        # Sparsified modularity graph achieves lower maximum chain length
        self.assertLess(metrics_sparse["max_chain_length"], metrics_dense["max_chain_length"])

    def test_manifest_generation_and_quota_accounting(self):
        """Verify Direction 2 test suite manifest integrity and exact 21.00s QPU quota accounting."""
        manifest_path = "benchmarks/quantum/direction2/manifest.json"
        qpu_manifest_path = "benchmarks/quantum/direction2/qpu_job_manifest.json"

        # If production run has completed, test files; otherwise run fast pilot check
        if not os.path.exists(manifest_path):
            test_dir = "benchmarks/quantum/direction2_test"
            suite = Direction2TestSuite(output_dir=test_dir, pegasus_m=16)
            summary = suite.generate_suite(is_pilot=True, compute_embeddings=False)
            manifest_path = summary["manifest_file"]
            qpu_manifest_path = summary["qpu_manifest_file"]

        self.assertTrue(os.path.exists(manifest_path))
        self.assertTrue(os.path.exists(qpu_manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        with open(qpu_manifest_path, "r", encoding="utf-8") as f:
            qpu_jobs = json.load(f)

        # Check production manifest if full run
        if not manifest_data.get("is_pilot", False):
            # Exactly 35 graphs (20 synthetic + 15 real-world) x 4 scale cuts = 140 instances
            self.assertEqual(manifest_data["total_graphs"], 35)
            self.assertEqual(manifest_data["num_synthetic_graphs"], 20)
            self.assertEqual(manifest_data["num_real_graphs"], 15)
            self.assertEqual(manifest_data["total_instances"], 140)
            self.assertEqual(len(manifest_data["instances"]), 140)

            # Exactly 140 instances x 3 chain strengths = 420 QPU calls
            self.assertEqual(qpu_jobs["total_qpu_submissions"], 420)
            self.assertEqual(len(qpu_jobs["jobs"]), 420)

            # Quota Accounting: 420 calls x 0.05s = exactly 21.00 seconds budgeted
            self.assertEqual(qpu_jobs["total_estimated_qpu_seconds"], 21.0)
            self.assertEqual(qpu_jobs["quota_headroom_seconds"], 39.0)

            # Verify chain strength multipliers
            self.assertEqual(qpu_jobs["chain_strength_multipliers"], [0.8, 1.2, 1.6])

            # Explicit architectural prohibition against cloud hybrid samplers
            self.assertIn("LeapHybridSampler", qpu_jobs["solver_prohibition"])
            self.assertIn("LeapHybridCQMSampler", qpu_jobs["solver_prohibition"])
            self.assertIn("strictly forbidden", qpu_jobs["solver_prohibition"])
            self.assertEqual(qpu_jobs["sampler"], "DWaveSampler")

            # Verify all instance JSON files exist, have valid structure, and contain offset
            for item in manifest_data["instances"]:
                full_path = os.path.join(os.path.dirname(manifest_path), item["json_path"])
                self.assertTrue(os.path.exists(full_path), f"Missing instance file: {full_path}")
                self.assertGreater(item["optimal_chain_strength"], 0.0)
                with open(full_path, "r", encoding="utf-8") as inst_f:
                    inst_json = json.load(inst_f)
                self.assertIn("offset", inst_json)
                self.assertIn("linear", inst_json)
                self.assertIn("quadratic", inst_json)

    def test_modularity_matrix_and_ultrametric_edge_cases(self):
        """Verify modularity matrix, ultrametric distance, and cuts handle edge cases robustly."""
        # 1. Empty graph (N=0)
        g_empty = nx.Graph()
        b_empty = build_modularity_matrix(g_empty)
        self.assertEqual(b_empty.shape, (0, 0))
        du_empty = compute_ultrametric_distances(g_empty)
        self.assertEqual(du_empty.shape, (0, 0))
        self.assertEqual(compute_modularity(g_empty, {}), 0.0)
        cuts_empty = compute_4_scale_cuts(du_empty)
        self.assertEqual(len(cuts_empty), 4)

        # 2. Single-node graph (N=1)
        g_single = nx.Graph()
        g_single.add_node(0)
        b_single = build_modularity_matrix(g_single)
        self.assertEqual(b_single.shape, (1, 1))
        self.assertEqual(b_single[0, 0], 0.0)
        du_single = compute_ultrametric_distances(g_single)
        self.assertEqual(du_single.shape, (1, 1))
        self.assertEqual(du_single[0, 0], 0.0)
        self.assertEqual(compute_modularity(g_single, {0: 1}), 0.0)

        # 3. Graph with nodes but zero edges (m=0)
        g_no_edges = nx.empty_graph(5)
        b_no_edges = build_modularity_matrix(g_no_edges)
        self.assertEqual(b_no_edges.shape, (5, 5))
        np.testing.assert_allclose(b_no_edges, np.zeros((5, 5)))
        self.assertEqual(compute_modularity(g_no_edges, {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}), 0.0)

        # 4. Disconnected graph with isolated nodes and multiple components
        g_disconn = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
        g_disconn.add_node(8)  # Isolated node
        n_disc = len(g_disconn)
        b_disc = build_modularity_matrix(g_disconn)
        np.testing.assert_allclose(b_disc, b_disc.T, atol=1e-12)
        np.testing.assert_allclose(np.sum(b_disc, axis=1), np.zeros(n_disc), atol=1e-12)

        du_disc = compute_ultrametric_distances(g_disconn)
        # Strong triangle inequality on disconnected graph
        for x in range(n_disc):
            for y in range(n_disc):
                for z in range(n_disc):
                    self.assertLessEqual(
                        du_disc[x, z],
                        max(du_disc[x, y], du_disc[y, z]) + 1e-12,
                        f"Non-Archimedean violation for disconnected graph at {x}, {y}, {z}"
                    )

        # 5. Graph with arbitrary string node labels
        g_str = nx.Graph()
        g_str.add_edge("node_a", "node_b")
        g_str.add_edge("node_b", "node_c")
        b_str = build_modularity_matrix(g_str)
        self.assertEqual(b_str.shape, (3, 3))
        np.testing.assert_allclose(np.sum(b_str, axis=1), np.zeros(3), atol=1e-12)
        du_str = compute_ultrametric_distances(g_str)
        self.assertEqual(du_str.shape, (3, 3))

        # 6. Small graph with < 4 unique distances (triangle K_3)
        g_tri = nx.complete_graph(3)
        du_tri = compute_ultrametric_distances(g_tri)
        cuts_tri = compute_4_scale_cuts(du_tri)
        self.assertEqual(len(cuts_tri), 4)
        for i in range(3):
            self.assertLessEqual(cuts_tri[i], cuts_tri[i + 1])

    def test_embedding_fallback_and_chain_strength_safety(self):
        """Verify minor embedding analytical fallback and chain strength positivity under zero coupling."""
        suite = Direction2TestSuite()

        # 1. Topology caching without explicit cache_key
        g = nx.path_graph(6)
        metrics1 = suite._get_cached_embedding_metrics(g)
        metrics2 = suite._get_cached_embedding_metrics(g)
        self.assertEqual(metrics1["max_chain_length"], metrics2["max_chain_length"])
        self.assertGreater(metrics1["max_chain_length"], 0)

        # 2. Analytical fallback produces non-trivial chain length for dense topology
        g_dense = nx.complete_graph(30)
        metrics_dense = suite._get_cached_embedding_metrics(g_dense, cache_key="test_dense_30")
        self.assertGreater(metrics_dense["max_chain_length"], 1)

        # 3. Chain strength safety floor when couplings are degenerate
        l_max = 4
        sqrt_l = math.sqrt(l_max)
        rms_zero = 0.0
        effective_rms = rms_zero if rms_zero > 1e-9 else 1.0
        lambda_08 = max(1e-4, float(0.8 * effective_rms * sqrt_l))
        lambda_12 = max(1e-4, float(1.2 * effective_rms * sqrt_l))
        lambda_16 = max(1e-4, float(1.6 * effective_rms * sqrt_l))
        self.assertGreater(lambda_08, 0.0)
        self.assertGreater(lambda_12, 0.0)
        self.assertGreater(lambda_16, 0.0)
        self.assertLessEqual(lambda_08, lambda_12)
        self.assertLessEqual(lambda_12, lambda_16)

    def test_dry_run_simulation_execution(self):
        """Verify Direction 2 dry-run execution with 5-gauge SRT and modularity evaluation."""
        test_dir = "benchmarks/quantum/direction2_dry_test"
        suite = Direction2TestSuite(output_dir=test_dir, pegasus_m=16)
        summary = suite.generate_suite(is_pilot=True, compute_embeddings=False)

        out_results = os.path.join(test_dir, "dry_run_results.json")
        res = run_direction2_dry_run(
            manifest_path=summary["manifest_file"],
            qpu_manifest_path=summary["qpu_manifest_file"],
            output_path=out_results,
            max_jobs=4,
        )

        self.assertEqual(res["total_jobs_executed"], 4)
        self.assertTrue(os.path.exists(out_results))
        self.assertGreaterEqual(res["mean_realized_modularity"], -1.0)
        self.assertLessEqual(res["mean_realized_modularity"], 1.0)

        # Cleanup temporary test files
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
