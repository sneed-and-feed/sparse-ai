"""
Comprehensive Unit Tests for Direction 1 Test Suite & Infrastructure.

Verifies:
1. HEA spin glass generation, p-adic LCA metric, and JSON serialization.
2. Planted Frustrated Cluster Loops (Hen et al. 2015), odd-frustration signature,
   and analytical ground state energy verification against exact SCIP solver.
3. SCIP exact MILP branch-and-cut with McCormick linearization envelopes.
4. Pegasus P_M embedding quality evaluation and clique comparisons.
5. Direction 1 Test Suite manifest integrity (165 instances, 150 HEA, 15 Planted, 660 QPU jobs).
6. Local Leap dry-run simulation and TTS_99 statistical metric reporting.
"""

import os
import json
import unittest
import numpy as np
import networkx as nx
import dimod

from src.quantum.hea_generator import (
    generate_hea_spin_glass,
    compute_p_adic_lca_distance,
    HEAInstance,
)
from src.quantum.planted_generator import (
    generate_planted_frustrated_loops,
    PlantedLoopInstance,
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
    compare_ultrametric_vs_clique,
)
from src.quantum.direction1_suite import Direction1TestSuite
from src.quantum.leap_runner import run_dry_run_simulation


class TestDirection1Suite(unittest.TestCase):
    """Test suite covering Direction 1 models, solvers, and benchmark infrastructure."""

    def test_p_adic_lca_distances_and_ultrametric_inequality(self):
        """Verify p-adic distance calculation and the strong triangle inequality."""
        # Non-Archimedean ultrametric inequality: d(x, z) <= max(d(x, y), d(y, z))
        p = 2
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    d_xy = compute_p_adic_lca_distance(x, y, p=p)
                    d_yz = compute_p_adic_lca_distance(y, z, p=p)
                    d_xz = compute_p_adic_lca_distance(x, z, p=p)
                    self.assertLessEqual(d_xz, max(d_xy, d_yz), f"Failed for {x}, {y}, {z}")

    def test_hea_generation_and_serialization(self):
        """Verify HEA instance generation across varying N and JSON roundtrip."""
        for n in [16, 32, 48]:
            inst = generate_hea_spin_glass(num_spins=n, sigma=0.8, seed=42)
            self.assertEqual(inst.num_spins, n)
            self.assertEqual(len(inst.bqm.variables), n)

            # Test JSON round-trip
            data = inst.to_dict()
            reconstructed = HEAInstance.from_dict(data)
            self.assertEqual(reconstructed.num_spins, n)
            self.assertEqual(len(reconstructed.bqm.quadratic), len(inst.bqm.quadratic))

            # Verify energies match on identical state
            test_state = {i: 1 for i in range(n)}
            self.assertAlmostEqual(
                inst.bqm.energy(test_state),
                reconstructed.bqm.energy(test_state),
                places=6,
            )

    def test_planted_frustrated_loops_analytical_certification(self):
        """Verify Hen et al. (2015) planted frustrated loops and exact ground state energy."""
        # Test N=16 with 4 loops of size 4
        planted_inst = generate_planted_frustrated_loops(
            num_spins=16,
            num_loops=4,
            loop_size=4,
            seed=123,
            topology="cluster",
        )

        self.assertEqual(planted_inst.num_spins, 16)
        self.assertEqual(len(planted_inst.loops), 4)

        # Expected energy: 4 loops * -(4 - 2) = -8.0
        expected_energy = -8.0
        self.assertAlmostEqual(planted_inst.analytical_ground_energy, expected_energy)

        # Verify energy of planted state on BQM
        e_planted = planted_inst.bqm.energy(planted_inst.planted_state)
        self.assertAlmostEqual(e_planted, expected_energy)

        # Verify exact solver certifies that no lower energy exists
        scip_res = solve_scip_exact(planted_inst.bqm, time_limit_sec=10.0)
        self.assertTrue(scip_res["is_optimal"])
        self.assertAlmostEqual(scip_res["best_energy"], expected_energy, places=5)

    def test_scip_exact_solver_vs_dimod_exact(self):
        """Verify SCIP MILP McCormick linearization matches dimod.ExactSolver on random spin glass."""
        rng = np.random.default_rng(999)
        n = 8
        h = {i: float(rng.normal(0, 0.5)) for i in range(n)}
        j = {(u, v): float(rng.normal(0, 1.0)) for u in range(n) for v in range(u + 1, n)}
        bqm = dimod.BinaryQuadraticModel(h, j, 0.0, vartype=dimod.SPIN)

        dimod_res = dimod.ExactSolver().sample(bqm).first
        scip_res = solve_scip_exact(bqm, time_limit_sec=10.0)

        self.assertTrue(scip_res["is_optimal"])
        self.assertAlmostEqual(scip_res["best_energy"], float(dimod_res.energy), places=5)

        # Verify evaluated energy of SCIP sample matches objective
        eval_e = bqm.energy(scip_res["best_sample"])
        self.assertAlmostEqual(eval_e, scip_res["best_energy"], places=5)

    def test_pegasus_embedding_and_clique_comparison(self):
        """Verify Pegasus embedding metrics and tree vs clique dilation contrast."""
        # Use small Pegasus P_4 for fast test execution
        target_p4 = get_pegasus_target_graph(4)

        # Ultrametric / cluster planted graph on 16 spins
        planted = generate_planted_frustrated_loops(num_spins=16, num_loops=4, seed=42)
        comp = compare_ultrametric_vs_clique(planted.graph, m=4, target_graph=target_p4)

        self.assertIn("ultrametric", comp)
        self.assertIn("clique", comp)
        self.assertTrue(comp["ultrametric"]["is_valid"])

        # Ultrametric max chain length should be small (<= 2 inside clusters)
        self.assertLessEqual(comp["ultrametric"]["max_chain_length"], 2)
        # Complete graph K_16 requires significantly longer chains and more physical qubits
        self.assertGreater(comp["clique"]["num_physical"], comp["ultrametric"]["num_physical"])

    def test_full_direction1_manifest_and_quota_accounting(self):
        """Verify integrity of the generated Direction 1 test suite and strict QPU quota bounds."""
        manifest_path = "benchmarks/quantum/direction1/manifest.json"
        qpu_manifest_path = "benchmarks/quantum/direction1/qpu_job_manifest.json"

        self.assertTrue(os.path.exists(manifest_path), "manifest.json missing")
        self.assertTrue(os.path.exists(qpu_manifest_path), "qpu_job_manifest.json missing")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        with open(qpu_manifest_path, "r", encoding="utf-8") as f:
            qpu_jobs = json.load(f)

        # Verify 165 instances
        self.assertEqual(manifest["total_instances"], 165)
        self.assertEqual(manifest["num_hea_instances"], 150)
        self.assertEqual(manifest["num_planted_instances"], 15)
        self.assertEqual(len(manifest["instances"]), 165)

        # Verify QPU submissions count: 165 * 4 = 660 calls
        self.assertEqual(qpu_jobs["total_qpu_submissions"], 660)
        self.assertEqual(len(qpu_jobs["jobs"]), 660)

        # Verify quota accounting: exactly 33.00 seconds consumed, leaving 27.00s headroom
        self.assertEqual(qpu_jobs["total_estimated_qpu_seconds"], 33.0)
        self.assertEqual(qpu_jobs["quota_headroom_seconds"], 27.0)

        # Verify all instances have positive optimal chain strength and valid energies
        for item in manifest["instances"]:
            self.assertGreater(item["optimal_chain_strength"], 0.0)
            self.assertLess(item["ground_state_energy"], 0.0)
            self.assertTrue(os.path.exists(os.path.join("benchmarks/quantum/direction1", item["json_path"])))

    def test_leap_runner_dry_run_simulation(self):
        """Verify Leap dry-run simulation execution and TTS_99 metric calculation."""
        out_file = "benchmarks/quantum/direction1/test_dry_run.json"
        res = run_dry_run_simulation(
            manifest_path="benchmarks/quantum/direction1/manifest.json",
            qpu_manifest_path="benchmarks/quantum/direction1/qpu_job_manifest.json",
            output_path=out_file,
            max_jobs=4,
        )

        self.assertEqual(res["total_jobs_executed"], 4)
        self.assertGreater(res["mean_p_gs"], 0.0)
        self.assertTrue(os.path.exists(out_file))

        # Clean up temporary test output
        if os.path.exists(out_file):
            os.remove(out_file)

    def test_tts_small_sample_edge_cases(self):
        """Verify TTS calculation handles small N and 0 successes without log domain errors."""
        # N=0 edge case
        res_0 = compute_time_to_solution(0.001, -10.0, np.array([]))
        self.assertEqual(res_0["num_reads"], 0)
        self.assertEqual(res_0["tts_99"], 0.0)

        # N=1, 2, 3 with 0 successes (previously caused log(-0.5) domain errors)
        for n in [1, 2, 3, 5]:
            energies = np.array([-5.0] * n)  # Ground state is -10.0 => 0 successes
            res = compute_time_to_solution(0.001, -10.0, energies)
            self.assertEqual(res["success_count"], 0)
            self.assertTrue(res["is_lower_bound"])
            self.assertGreater(res["tts_99"], 0.0)
            self.assertFalse(np.isnan(res["tts_99"]))

    def test_leap_runner_live_safety_guard_and_connection_check(self):
        """Verify Leap runner unconfirmed safety guard and connection check dictionary format."""
        from src.quantum.leap_runner import test_qpu_connection, run_live_qpu_submission

        # Safety guard blocks unconfirmed execution
        res_guard = run_live_qpu_submission(
            manifest_path="benchmarks/quantum/direction1/manifest.json",
            qpu_manifest_path="benchmarks/quantum/direction1/qpu_job_manifest.json",
            output_path="test_live.json",
            confirm_live=False,
        )
        self.assertEqual(res_guard["status"], "unconfirmed")

        # Connection check returns structured dict without throwing
        conn = test_qpu_connection()
        self.assertIn("connected", conn)
        self.assertIn("token_present", conn)


if __name__ == "__main__":
    unittest.main()
