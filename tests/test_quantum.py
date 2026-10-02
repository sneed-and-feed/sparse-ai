"""
Unit Tests for Project Q-Ultrametric Modules (src/quantum).
"""

import unittest
import numpy as np
import networkx as nx
import dimod

from src.quantum.hea_generator import (
    generate_hea_spin_glass,
    compute_p_adic_lca_distance,
    HEAInstance,
)
from src.quantum.embeddings import (
    evaluate_embedding_quality,
)
from src.quantum.solvers import (
    solve_locally_classical,
    compute_time_to_solution,
)


class TestQuantumUltrametric(unittest.TestCase):
    """Test suite for quantum annealing ultrametric generation and local solving."""

    def test_p_adic_lca_distance(self):
        """Test hierarchical LCA distance computation."""
        # Consecutive leaves share parent at level 1
        self.assertEqual(compute_p_adic_lca_distance(0, 1, p=2), 1)
        self.assertEqual(compute_p_adic_lca_distance(2, 3, p=2), 1)
        # Leaves 0 and 2 share parent at level 2
        self.assertEqual(compute_p_adic_lca_distance(0, 2, p=2), 2)
        # Identical leaves have distance 0
        self.assertEqual(compute_p_adic_lca_distance(5, 5, p=2), 0)

    def test_hea_spin_glass_generation(self):
        """Test generation of Hierarchical Edwards-Anderson spin glass."""
        levels = 4
        p = 2
        instance = generate_hea_spin_glass(levels=levels, p=p, sigma=0.8, seed=42)

        self.assertEqual(instance.num_spins, 16)
        self.assertEqual(len(instance.bqm.variables), 16)
        self.assertEqual(instance.graph.number_of_nodes(), 16)
        # Complete tree leaves have N*(N-1)/2 pairs
        self.assertEqual(instance.graph.number_of_edges(), 16 * 15 // 2)

    def test_local_classical_solvers(self):
        """Test local exact and simulated annealing solvers."""
        instance = generate_hea_spin_glass(levels=3, p=2, sigma=0.8, seed=123)  # N=8
        
        # Test Exact Solver
        res_exact = solve_locally_classical(instance.bqm, method="exact")
        self.assertIn("best_energy", res_exact)
        self.assertEqual(len(res_exact["best_sample"]), 8)

        # Test Simulated Annealing
        res_sa = solve_locally_classical(instance.bqm, method="sa", num_reads=50, num_sweeps=100)
        self.assertIn("best_energy", res_sa)
        self.assertGreaterEqual(res_sa["best_energy"], res_exact["best_energy"] - 1e-4)

        # Test Tabu Search
        res_tabu = solve_locally_classical(instance.bqm, method="tabu", num_reads=20)
        self.assertIn("best_energy", res_tabu)

    def test_time_to_solution(self):
        """Test TTS_99 statistical metric calculation."""
        energies = np.array([-10.0, -10.0, -8.0, -7.0, -10.0])
        tts_res = compute_time_to_solution(
            run_time_per_read=1e-3,
            ground_state_energy=-10.0,
            energies=energies,
        )
        self.assertAlmostEqual(tts_res["p_gs"], 0.6)
        self.assertEqual(tts_res["success_count"], 3)
        self.assertFalse(tts_res["is_lower_bound"])


if __name__ == "__main__":
    unittest.main()
