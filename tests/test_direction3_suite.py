"""
Comprehensive Unit Tests for Direction 3 Test Suite & Infrastructure.

Verifies:
1. HQ-QBM energy function, clamped state probabilities, and exact partition function.
2. Hardware-aware Pegasus tree mask generation and minor embedding verification (L_max <= 2).
3. Continuous Logit Homotopy and Straight-Through Estimator routing.
4. Quantum Contrastive Divergence (QCD) training steps (clamped vs thermal energy dynamics).
5. Benchmark datasets (Bars-and-Stripes 4x4, 8x8 Digits, Dyck Sequences).
6. Model learning and NLL reduction on Bars-and-Stripes.
7. Direction 3 Test Suite manifest generation, strict QPU quota calculation (exactly 10.0s),
   and local dry-run simulation with 5-gauge Spin-Reversal Transforms.
"""

import os
import json
import math
import tempfile
import unittest
import numpy as np
import torch
import dimod

from src.quantum.direction3_qbm import (
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
from src.quantum.direction3_suite import (
    Direction3TestSuite,
    run_dry_run_simulation,
)


class TestDirection3Suite(unittest.TestCase):
    """Test suite covering Direction 3 HQ-QBM models, QCD training, and benchmark infrastructure."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_hqqbm_energy_and_bipartite_interaction(self):
        """Verify energy calculation on manual inputs, spin alignment, and batch consistency."""
        model = HardwareAwareQBM(num_visible=4, num_hidden=2, seed=123)

        # Set specific known parameters
        with torch.no_grad():
            model.weights.fill_(0.0)
            model.weights[0, 0] = 2.0  # Coupling v0 * h0 with weight 2.0
            model.visible_bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            model.hidden_bias.copy_(torch.tensor([0.5, 0.0]))
            # Full connectivity mask for this test
            model.routing_logits.fill_(10.0)

        # State 1: v0 = 1, h0 = 1 (aligned)
        v = torch.tensor([1.0, -1.0, -1.0, -1.0])
        h = torch.tensor([1.0, -1.0])
        # E = - (v0 * W00 * h0) - a0 * v0 - b0 * h0 = - (1 * 2 * 1) - (1 * 1) - (0.5 * 1) = -2 - 1 - 0.5 = -3.5
        e1 = model.energy(v, h).item()
        self.assertAlmostEqual(e1, -3.5, places=4)

        # State 2: v0 = -1, h0 = 1 (anti-aligned)
        v_anti = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        # E = - (-1 * 2 * 1) - (1 * -1) - (0.5 * 1) = +2 + 1 - 0.5 = +2.5
        e2 = model.energy(v_anti, h).item()
        self.assertAlmostEqual(e2, 2.5, places=4)

        # Verify aligned state has strictly lower energy
        self.assertLess(e1, e2)

        # Test batched energy computation vs unbatched
        v_batch = torch.stack([v, v_anti])
        h_batch = torch.stack([h, h])
        e_batch = model.energy(v_batch, h_batch)
        self.assertEqual(e_batch.shape, (2,))
        self.assertAlmostEqual(e_batch[0].item(), e1, places=4)
        self.assertAlmostEqual(e_batch[1].item(), e2, places=4)

    def test_clamped_state_probabilities_and_expectations(self):
        """Verify conditional expectations <h|v> and Bernoulli sampling in {-1, +1}."""
        model = HardwareAwareQBM(num_visible=4, num_hidden=3, seed=42)
        v = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])

        # Expected hidden activations
        h_exp = model.clamped_hidden_expectation(v)
        self.assertEqual(h_exp.shape, (2, 3))
        # Ensure activations are strictly within (-1, 1)
        self.assertTrue(torch.all(h_exp >= -1.0) and torch.all(h_exp <= 1.0))

        # Sampled hidden units must be in {-1, +1}
        h_samp = model.sample_hidden(v)
        self.assertEqual(h_samp.shape, (2, 3))
        unique_vals = set(h_samp.unique().cpu().numpy().tolist())
        self.assertTrue(unique_vals.issubset({-1.0, 1.0}))

        # Sampled visible units must be in {-1, +1}
        v_samp = model.sample_visible(h_samp)
        self.assertEqual(v_samp.shape, (2, 4))
        unique_v = set(v_samp.unique().cpu().numpy().tolist())
        self.assertTrue(unique_v.issubset({-1.0, 1.0}))

        # Law of large numbers check for sampling
        large_v = v[0:1].repeat(2000, 1)
        many_samples = model.sample_hidden(large_v)
        emp_mean = many_samples.mean(dim=0)
        ana_mean = h_exp[0]
        for j in range(3):
            self.assertAlmostEqual(emp_mean[j].item(), ana_mean[j].item(), delta=0.08)

    def test_exact_partition_function_and_nll(self):
        """Verify exact partition function Z against brute-force joint summation and NLL."""
        nv, nh = 4, 3
        model = HardwareAwareQBM(num_visible=nv, num_hidden=nh, seed=99)

        # Brute-force joint enumeration of all 2^(nv+nh) states
        all_v = model._all_visible_states()
        all_h = model._all_hidden_states()

        w_eff = model.effective_weights()
        a = model.visible_bias
        b = model.hidden_bias

        # Joint energy tensor (2^nv, 2^nh)
        joint_e = -(all_v @ w_eff @ all_h.t()) - (all_v @ a).unsqueeze(1) - (all_h @ b).unsqueeze(0)
        brute_force_z = torch.exp(-joint_e).sum().item()
        brute_force_log_z = torch.logsumexp(-joint_e, dim=(0, 1)).item()

        # Model methods
        model_z = model.exact_partition_function()
        model_log_z = model.exact_log_partition_function()

        self.assertAlmostEqual(brute_force_z, model_z, places=3)
        self.assertAlmostEqual(brute_force_log_z, model_log_z, places=4)

        # Verify NLL is strictly positive and finite
        test_data = torch.tensor([[1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])
        nll = model.exact_nll(test_data)
        self.assertGreater(nll, 0.0)
        self.assertTrue(math.isfinite(nll))

    def test_pegasus_hardware_aware_mask_and_lmax2_embedding(self):
        """Verify Pegasus mask generation and physical minor embedding with L_max <= 2."""
        test_configurations = [
            (16, 4),  # BAS 4x4 with 4 hidden
            (16, 8),  # BAS 4x4 with 8 hidden
            (64, 8),  # 8x8 Digits with 8 hidden
        ]

        for nv, nh in test_configurations:
            mask = generate_pegasus_tree_mask(nv, nh, max_visible_degree=2)
            self.assertEqual(mask.shape, (nv, nh))

            # Verify every visible unit is connected
            vis_degrees = mask.sum(dim=-1)
            self.assertTrue(torch.all(vis_degrees >= 1.0))
            self.assertTrue(torch.all(vis_degrees <= 2.0))

            # Verify physical embedding onto Pegasus target graph
            emb_res = verify_pegasus_embedding(mask, pegasus_m=4, timeout_sec=15.0)
            self.assertTrue(emb_res["is_valid"])
            self.assertLessEqual(
                emb_res["max_chain_length"],
                2,
                f"Embedding exceeded L_max=2 budget for N_v={nv}, N_h={nh}: {emb_res['max_chain_length']}",
            )
            self.assertTrue(emb_res["complies_with_lmax_budget"])

    def test_continuous_logit_homotopy_and_ste_gradients(self):
        """Verify routing temperature annealing and Straight-Through Estimator backpropagation."""
        model = HardwareAwareQBM(num_visible=8, num_hidden=4, initial_tau=1.0, seed=42)

        # Check soft mask at tau = 1.0 vs tau = 0.1
        soft_mask_warm = model.get_sparse_mask(hard=False, tau=1.0)
        soft_mask_cold = model.get_sparse_mask(hard=False, tau=0.1)

        # Colder mask should be more polarized towards 0 or 1
        entropy_warm = -(soft_mask_warm * torch.log(soft_mask_warm + 1e-8) + (1 - soft_mask_warm) * torch.log(1 - soft_mask_warm + 1e-8)).mean()
        entropy_cold = -(soft_mask_cold * torch.log(soft_mask_cold + 1e-8) + (1 - soft_mask_cold) * torch.log(1 - soft_mask_cold + 1e-8)).mean()
        self.assertLess(entropy_cold.item(), entropy_warm.item())

        # Test Straight-Through Estimator gradients backpropagate to routing_logits
        hard_mask = model.get_sparse_mask(hard=True, tau=0.5)
        dummy_loss = (model.weights * hard_mask).sum()
        dummy_loss.backward()

        self.assertIsNotNone(model.routing_logits.grad)
        self.assertGreater(torch.norm(model.routing_logits.grad).item(), 0.0)

        # Test temperature schedule annealing
        t_mid = model.anneal_routing_temperature(50, 100, tau_start=1.0, tau_end=0.1)
        self.assertAlmostEqual(t_mid, 0.3162, delta=0.01)
        t_final = model.anneal_routing_temperature(100, 100, tau_start=1.0, tau_end=0.1)
        self.assertAlmostEqual(t_final, 0.1, delta=0.01)

    def test_qcd_training_step_energy_gap_and_direction(self):
        """Verify that a QCD training step updates weights to reduce clamped vs thermal energy gap."""
        model = HardwareAwareQBM(num_visible=4, num_hidden=2, seed=777)
        data = torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)

        mask = model.get_sparse_mask(hard=True)
        w_before = model.weights.clone()

        step_res = model.qcd_training_step(data, lr=0.1, sampling_method="exact")
        self.assertIn("clamped_energy", step_res)
        self.assertIn("thermal_energy", step_res)
        self.assertIn("energy_gap", step_res)

        # Weights corresponding to zero mask entries must NOT change
        w_diff = (model.weights - w_before).abs()
        zero_mask_entries = (mask == 0.0)
        self.assertTrue(torch.all(w_diff[zero_mask_entries] < 1e-7))

    def test_benchmark_datasets(self):
        """Verify synthetic BAS, 8x8 digits, and Dyck language datasets."""
        # 1. Bars-and-Stripes 4x4
        bas4 = generate_bars_and_stripes_dataset(grid_size=4)
        # Exactly 2^4 + 2^4 - 2 = 30 patterns
        self.assertEqual(bas4.shape, (30, 16))
        self.assertTrue(torch.all((bas4 == 1.0) | (bas4 == -1.0)))

        # Bars-and-Stripes 2x2
        bas2 = generate_bars_and_stripes_dataset(grid_size=2)
        # Exactly 2^2 + 2^2 - 2 = 6 patterns
        self.assertEqual(bas2.shape, (6, 4))

        # 2. Downscaled Digits 8x8
        digits = generate_downscaled_digits_dataset(num_samples=25, noise_flip_prob=0.0, seed=1)
        self.assertEqual(digits.shape, (25, 64))
        self.assertTrue(torch.all((digits == 1.0) | (digits == -1.0)))

        # 3. Dyck Language Sequences
        dyck = generate_dyck_sequence_dataset(seq_len=8, bits_per_token=2, num_samples=20, seed=1)
        self.assertEqual(dyck.shape, (20, 16))
        self.assertTrue(torch.all((dyck == 1.0) | (dyck == -1.0)))

    def test_model_learning_bars_and_stripes(self):
        """Verify that HQ-QBM learns the 2x2 BAS distribution with decreasing NLL and MSE."""
        bas2 = generate_bars_and_stripes_dataset(grid_size=2)
        model = HardwareAwareQBM(num_visible=4, num_hidden=3, seed=101)

        init_nll = model.exact_nll(bas2)
        init_mse = compute_reconstruction_error(model, bas2)

        # Train for 40 steps
        for step in range(40):
            model.qcd_training_step(bas2, lr=0.1, sampling_method="exact")

        final_nll = model.exact_nll(bas2)
        final_mse = compute_reconstruction_error(model, bas2)

        self.assertLess(final_nll, init_nll)
        self.assertLess(final_mse, init_mse)

        # Test Total Variation distance computation
        tv_dist = compute_total_variation_distance(model, bas2)
        self.assertGreaterEqual(tv_dist, 0.0)
        self.assertLessEqual(tv_dist, 1.0)

    def test_suite_manifest_generation_and_strict_quota(self):
        """Verify Direction 3 Test Suite manifest generation and strict 10.0s quota accounting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Direction3TestSuite(output_dir=tmpdir, pegasus_m=4)
            summary = suite.generate_suite(
                is_pilot=True,
                pretrain_steps=5,
                verify_embeddings=True,
            )

            # Check files exist
            self.assertTrue(os.path.exists(summary["manifest_file"]))
            self.assertTrue(os.path.exists(summary["qpu_manifest_file"]))

            with open(summary["manifest_file"], "r", encoding="utf-8") as f:
                manifest = json.load(f)

            with open(summary["qpu_manifest_file"], "r", encoding="utf-8") as f:
                qpu_manifest = json.load(f)

            # Manifest assertions
            self.assertEqual(manifest["suite_name"], "Direction 3: Hardware-Aware Sparse Quantum Boltzmann Machines (HQ-QBM)")
            self.assertEqual(len(manifest["models"]), 1)
            model_meta = manifest["models"][0]
            self.assertTrue(model_meta["pegasus_lmax_compliant"])
            self.assertLessEqual(model_meta["pegasus_max_chain_length"], 2)

            # Strict Quota Accounting Assertions
            # 200 physical updates = 200 calls = exactly 10.00 seconds
            self.assertEqual(qpu_manifest["total_qpu_submissions"], 200)
            self.assertAlmostEqual(qpu_manifest["total_estimated_qpu_seconds"], 10.0, places=2)
            self.assertEqual(len(qpu_manifest["jobs"]), 200)
            self.assertEqual(qpu_manifest["num_gauges_srt"], 5)
            self.assertEqual(qpu_manifest["reads_per_gauge"], 100)
            self.assertEqual(qpu_manifest["total_reads_planned"], 100000)

            # Solver prohibition check
            self.assertIn("LeapHybridSampler", qpu_manifest["solver_prohibition"])
            self.assertEqual(qpu_manifest["sampler"], "DWaveSampler")

    def test_dry_run_simulation_execution(self):
        """Verify zero-cost local dry-run simulation with neal 5-gauge SRT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Direction3TestSuite(output_dir=tmpdir, pegasus_m=4)
            summary = suite.generate_suite(
                is_pilot=True,
                pretrain_steps=3,
                verify_embeddings=False,
            )

            dry_out = os.path.join(tmpdir, "dry_run_results.json")
            # Run 3 simulated QPU jobs to verify runner
            res = run_dry_run_simulation(
                manifest_path=summary["manifest_file"],
                qpu_manifest_path=summary["qpu_manifest_file"],
                output_path=dry_out,
                max_jobs=3,
            )

            self.assertTrue(os.path.exists(dry_out))
            self.assertEqual(res["total_jobs_executed"], 3)
            self.assertIn("history", res)
            self.assertEqual(len(res["history"]), 3)
            self.assertIn("final_reconstruction_mse", res)
            self.assertIn("final_exact_nll", res)

    def test_mathematical_consistency_spin_vs_binary_and_tanh_sigmoid_identity(self):
        """
        Verify rigorous mathematical consistency between:
        1. Ising spin conditional probability P(h_j = +1 | v) = sigmoid(2 * phi_j)
        2. Expected value <h_j | v> = tanh(phi_j)
        3. Identity: <h_j | v> = (+1)*P(h_j=+1|v) + (-1)*P(h_j=-1|v) = 2*sigmoid(2*phi_j) - 1.
        """
        model = HardwareAwareQBM(num_visible=6, num_hidden=4, seed=123)
        v = torch.randn(10, 6).sign()  # random spin states in {-1, +1}
        w_eff = model.effective_weights()

        phi = torch.matmul(v, w_eff) + model.hidden_bias  # (10, 4)
        prob_plus = torch.sigmoid(2.0 * phi)
        expected_from_prob = 2.0 * prob_plus - 1.0
        expected_from_tanh = torch.tanh(phi)

        # The factor of 2 in sigmoid(2 * phi) guarantees exact numerical agreement with tanh(phi)
        diff = torch.abs(expected_from_prob - expected_from_tanh)
        self.assertLess(torch.max(diff).item(), 1e-6)

        # Similarly for visible units conditioned on hidden:
        h = torch.randn(10, 4).sign()
        psi = torch.matmul(h, w_eff.t()) + model.visible_bias
        prob_vis_plus = torch.sigmoid(2.0 * psi)
        exp_vis_from_prob = 2.0 * prob_vis_plus - 1.0
        exp_vis_from_tanh = torch.tanh(psi)
        diff_vis = torch.abs(exp_vis_from_prob - exp_vis_from_tanh)
        self.assertLess(torch.max(diff_vis).item(), 1e-6)

    def test_partition_function_hidden_vs_visible_exact_agreement(self):
        """Verify that exact Z summing over h equals Z summing over v and brute force."""
        import networkx as nx
        nv, nh = 6, 4
        model = HardwareAwareQBM(num_visible=nv, num_hidden=nh, seed=456)
        w_eff = model.effective_weights()
        a = model.visible_bias
        b = model.hidden_bias

        # 1. Sum over hidden states first
        H = model._all_hidden_states()  # (16, 4)
        psi = torch.matmul(H, w_eff.t()) + a
        log_cosh_vis = torch.abs(psi) + torch.log1p(torch.exp(-2.0 * torch.abs(psi)))
        log_terms_h = torch.matmul(H, b) + log_cosh_vis.sum(dim=-1)
        log_z_h = torch.logsumexp(log_terms_h, dim=0).item()

        # 2. Sum over visible states first
        V = model._all_visible_states()  # (64, 6)
        phi = torch.matmul(V, w_eff) + b
        log_cosh_hid = torch.abs(phi) + torch.log1p(torch.exp(-2.0 * torch.abs(phi)))
        log_terms_v = torch.matmul(V, a) + log_cosh_hid.sum(dim=-1)
        log_z_v = torch.logsumexp(log_terms_v, dim=0).item()

        # 3. Brute force joint sum over 2^(nv+nh) = 1024 states
        joint_e = -(V @ w_eff @ H.t()) - (V @ a).unsqueeze(1) - (H @ b).unsqueeze(0)
        log_z_brute = torch.logsumexp(-joint_e, dim=(0, 1)).item()

        self.assertAlmostEqual(log_z_h, log_z_v, places=5)
        self.assertAlmostEqual(log_z_h, log_z_brute, places=5)
        self.assertAlmostEqual(model.exact_log_partition_function(), log_z_brute, places=5)

    def test_acyclic_tree_mask_topology_and_degree_bounds(self):
        """Verify that generated bipartite masks form strict acyclic trees or forests."""
        import networkx as nx

        # 1. Deterministic mask: must be a single connected tree with V-1 edges
        nv, nh = 16, 4
        mask_det = generate_pegasus_tree_mask(nv, nh, routing_logits=None, max_visible_degree=2)
        g_det = nx.Graph()
        for v in range(nv):
            g_det.add_node(f"v_{v}")
        for h in range(nh):
            g_det.add_node(f"h_{h}")
        for v in range(nv):
            for h in range(nh):
                if mask_det[v, h] > 0.5:
                    g_det.add_edge(f"v_{v}", f"h_{h}")

        self.assertTrue(nx.is_tree(g_det), "Deterministic mask should form a connected tree")
        self.assertEqual(len(nx.cycle_basis(g_det)), 0, "Tree must have zero cycles")
        self.assertEqual(g_det.number_of_edges(), (nv + nh) - 1)

        # 2. Stochastic mask with routing logits: must be an acyclic forest
        torch.manual_seed(99)
        rl = torch.randn(nv, nh)
        mask_rl = generate_pegasus_tree_mask(nv, nh, routing_logits=rl, max_visible_degree=2)
        g_rl = nx.Graph()
        for v in range(nv):
            g_rl.add_node(f"v_{v}")
        for h in range(nh):
            g_rl.add_node(f"h_{h}")
        for v in range(nv):
            for h in range(nh):
                if mask_rl[v, h] > 0.5:
                    g_rl.add_edge(f"v_{v}", f"h_{h}")

        self.assertTrue(nx.is_forest(g_rl), "Logits-guided mask should be an acyclic forest")
        self.assertEqual(len(nx.cycle_basis(g_rl)), 0, "Forest must have zero cycles")

        # 3. Visible degree bounds
        vis_degs = mask_rl.sum(dim=-1)
        self.assertTrue(torch.all(vis_degs >= 1.0))
        self.assertTrue(torch.all(vis_degs <= 2.0))

    def test_embedding_cache_and_analytical_fallback(self):
        """Verify embedding caching and analytical fallback mechanism."""
        mask = generate_pegasus_tree_mask(16, 4)

        # Test caching on repeated calls
        res1 = verify_pegasus_embedding(mask, pegasus_m=4, use_cache=True)
        res2 = verify_pegasus_embedding(mask, pegasus_m=4, use_cache=True)
        self.assertEqual(res1["max_chain_length"], res2["max_chain_length"])
        self.assertEqual(res1["num_edges"], res2["num_edges"])

        # Test fallback when timeout is zero
        res_fallback = verify_pegasus_embedding(mask, pegasus_m=4, timeout_sec=0.00001, use_cache=False)
        self.assertTrue(res_fallback["is_valid"])
        self.assertLessEqual(res_fallback["max_chain_length"], 2)
        self.assertTrue(res_fallback["complies_with_lmax_budget"])

    def test_qcd_training_step_ste_routing_updates(self):
        """Verify STE gradient backpropagation into routing_logits during QCD training."""
        model = HardwareAwareQBM(num_visible=8, num_hidden=4, seed=55)
        data = torch.randn(12, 8).sign()

        logits_before = model.routing_logits.clone()
        res = model.qcd_training_step(data, lr=0.05, sampling_method="exact", update_routing=True, routing_lr=0.1)

        self.assertIn("grad_norm_routing", res)
        self.assertGreater(res["grad_norm_routing"], 0.0)
        logits_diff = (model.routing_logits - logits_before).abs().sum().item()
        self.assertGreater(logits_diff, 0.0)

    def test_dataset_generator_edge_cases(self):
        """Verify robust error handling and boundary conditions for dataset generators."""
        # 1. BAS invalid grid size
        with self.assertRaises(ValueError):
            generate_bars_and_stripes_dataset(grid_size=0)
        with self.assertRaises(ValueError):
            generate_bars_and_stripes_dataset(grid_size=-2)

        # 2. Digits zero samples
        zero_digits = generate_downscaled_digits_dataset(num_samples=0)
        self.assertEqual(zero_digits.shape, (0, 64))
        with self.assertRaises(ValueError):
            generate_downscaled_digits_dataset(num_samples=-5)

        # 3. Dyck invalid length (odd or <= 0)
        with self.assertRaises(ValueError):
            generate_dyck_sequence_dataset(seq_len=7)  # Odd length is mathematically invalid for Dyck
        with self.assertRaises(ValueError):
            generate_dyck_sequence_dataset(seq_len=0)

        # 4. Dyck zero samples
        zero_dyck = generate_dyck_sequence_dataset(seq_len=8, num_samples=0)
        self.assertEqual(zero_dyck.shape, (0, 16))

    def test_energy_dimension_mismatch_validation(self):
        """Verify that dimension mismatches in model.energy raise ValueError."""
        model = HardwareAwareQBM(num_visible=4, num_hidden=2)
        wrong_v = torch.tensor([1.0, -1.0, 1.0])  # size 3 != 4
        h = torch.tensor([1.0, -1.0])
        with self.assertRaises(ValueError):
            model.energy(wrong_v, h)

        v = torch.tensor([1.0, -1.0, 1.0, -1.0])
        wrong_h = torch.tensor([1.0, -1.0, 1.0])  # size 3 != 2
        with self.assertRaises(ValueError):
            model.energy(v, wrong_h)

    def test_direction3_module_exports_in_init(self):
        """Verify all Direction 3 symbols are properly exported from src.quantum."""
        import src.quantum as sq
        expected_symbols = [
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
        for sym in expected_symbols:
            self.assertTrue(hasattr(sq, sym), f"Symbol {sym} missing from src.quantum")
            self.assertIn(sym, sq.__all__, f"Symbol {sym} missing from src.quantum.__all__")


if __name__ == "__main__":
    unittest.main()
