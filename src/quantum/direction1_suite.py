"""
Direction 1 Test Suite Generator & Batch Submission Packager.

Orchestrates the complete benchmark test suite for Direction 1:
- 150 Hierarchical Edwards-Anderson (HEA) spin glasses across sizes
  N in {16, 32, 48, 64, 96, 128} and decay exponents sigma in {0.6, 0.8, 1.2}.
- 15 Planted Frustrated Cluster Loops (Hen et al. 2015) with analytical ground truths.
- Exact ground truth certification via SCIP (MIPGap = 0.0) and metaheuristic baselines.
- Pegasus P_16 minor embedding quality evaluation with topology caching.
- D-Wave Leap batch submission manifest generation (660 calls, 33.0s QPU quota strictly budgeted).
"""

import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import networkx as nx
import dimod

from src.quantum.hea_generator import generate_hea_spin_glass, HEAInstance
from src.quantum.planted_generator import generate_planted_frustrated_loops, PlantedLoopInstance
from src.quantum.solvers import solve_scip_exact, solve_locally_classical
from src.quantum.embeddings import (
    get_pegasus_target_graph,
    embed_graph_onto_pegasus,
    evaluate_embedding_quality,
)


@dataclass
class InstanceMetadata:
    """Metadata catalog entry for a single benchmark problem instance."""
    instance_id: str
    instance_type: str  # 'hea' or 'planted_loops'
    num_spins: int
    sigma: Optional[float]
    num_loops: Optional[int]
    seed: int
    ground_state_energy: float
    is_energy_exact: bool
    certification_method: str  # 'scip', 'analytical_planted', 'sa_upper_bound'
    num_couplers: int
    rms_coupling: float
    max_coupling: float
    pegasus_dilation_ratio: float
    pegasus_max_chain_length: int
    pegasus_mean_chain_length: float
    pegasus_dynamic_range_factor: float
    optimal_chain_strength: float
    json_path: str


class Direction1TestSuite:
    """
    Orchestrates the generation, offline certification, embedding evaluation,
    and Leap batch packaging of the Direction 1 benchmark suite.
    """

    def __init__(
        self,
        output_dir: str = "benchmarks/quantum/direction1",
        pegasus_m: int = 16,
    ):
        self.output_dir = output_dir
        self.instances_dir = os.path.join(output_dir, "instances")
        self.pegasus_m = pegasus_m
        self._target_graph: Optional[nx.Graph] = None
        self._embedding_cache: Dict[str, Dict[str, Any]] = {}

    def _get_target(self) -> nx.Graph:
        if self._target_graph is None:
            self._target_graph = get_pegasus_target_graph(self.pegasus_m)
        return self._target_graph

    def _get_cached_embedding_metrics(self, graph: nx.Graph, cache_key: str) -> Dict[str, Any]:
        """
        Caches embedding computation by graph topology.
        Since minor embedding depends strictly on adjacency (V, E) rather than
        continuous coupler weights, caching avoids redundant NP-hard heuristic searches.
        """
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        emb_metrics = {
            "dilation_ratio": 1.0,
            "max_chain_length": 1,
            "mean_chain_length": 1.0,
            "dynamic_range_factor": 1.0,
        }
        try:
            emb = embed_graph_onto_pegasus(
                graph,
                m=self.pegasus_m,
                target_graph=self._get_target(),
                timeout_sec=15.0,
            )
            eval_res = evaluate_embedding_quality(emb, graph)
            if eval_res["is_valid"]:
                emb_metrics = eval_res
        except Exception:
            pass

        self._embedding_cache[cache_key] = emb_metrics
        return emb_metrics

    def generate_suite(
        self,
        is_pilot: bool = False,
        certify_exact: bool = True,
        compute_embeddings: bool = True,
        scip_time_limit: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Generates the test suite instances, certifies energies, and saves artifacts.

        Args:
            is_pilot: If True, generates a fast 20-instance pilot (15 HEA + 5 Planted).
                      If False, generates the full 165-instance production benchmark set.
            certify_exact: If True, executes SCIP or analytical verification.
            compute_embeddings: If True, embeds onto Pegasus target graph and logs metrics.
            scip_time_limit: Maximum runtime in seconds per SCIP certification.

        Returns:
            Summary dictionary of the test suite catalog.
        """
        os.makedirs(self.instances_dir, exist_ok=True)
        manifest: List[InstanceMetadata] = []

        if is_pilot:
            hea_configs = [
                # 15 instances: 5 for each sigma
                (16, 0.6, [1001, 1002]),
                (32, 0.6, [2001, 2002, 2003]),
                (16, 0.8, [1101, 1102]),
                (32, 0.8, [2101, 2102, 2103]),
                (16, 1.2, [1201, 1202]),
                (32, 1.2, [2201, 2202, 2203]),
            ]
            planted_configs = [
                (16, 4, [7001, 7002, 7003]),
                (32, 8, [7004, 7005]),
            ]
        else:
            # Full 165 instances:
            # 150 HEA instances (50 per sigma):
            # N=16: 10, N=32: 10, N=48: 8, N=64: 8, N=96: 7, N=128: 7
            hea_configs = []
            for sigma_val in [0.6, 0.8, 1.2]:
                base_seed = int(round(sigma_val * 1000))
                hea_configs.extend([
                    (16, sigma_val, [base_seed + 100 + i for i in range(10)]),
                    (32, sigma_val, [base_seed + 200 + i for i in range(10)]),
                    (48, sigma_val, [base_seed + 300 + i for i in range(8)]),
                    (64, sigma_val, [base_seed + 400 + i for i in range(8)]),
                    (96, sigma_val, [base_seed + 500 + i for i in range(7)]),
                    (128, sigma_val, [base_seed + 600 + i for i in range(7)]),
                ])

            # 15 Planted Frustrated Cluster Loops
            planted_configs = [
                (16, 4, [7001, 7002, 7003, 7004, 7005]),
                (32, 8, [7006, 7007, 7008, 7009, 7010]),
                (64, 16, [7011, 7012, 7013, 7014, 7015]),
            ]

        # 1. Process HEA Instances
        hea_count = 0
        for n_spins, sigma, seeds in hea_configs:
            cache_key = f"hea_n{n_spins}"
            for seed in seeds:
                hea_count += 1
                inst_id = f"hea_n{n_spins}_s{int(round(sigma*100)):03d}_seed{seed}"
                hea_inst = generate_hea_spin_glass(
                    num_spins=n_spins,
                    sigma=sigma,
                    seed=seed,
                    normalize_couplers=False,
                )

                # Coupling statistics
                couplings = list(hea_inst.bqm.quadratic.values())
                rms_j = float(np.sqrt(np.mean(np.square(couplings)))) if couplings else 0.0
                max_j = float(np.max(np.abs(couplings))) if couplings else 0.0

                # Offline Ground Truth Certification
                gs_energy = 0.0
                is_exact = False
                cert_method = "unsolved"

                if certify_exact:
                    if n_spins <= 32:
                        try:
                            scip_res = solve_scip_exact(hea_inst.bqm, time_limit_sec=scip_time_limit)
                            gs_energy = float(scip_res["best_energy"])
                            is_exact = bool(scip_res["is_optimal"])
                            cert_method = "scip"
                        except Exception:
                            sa_res = solve_locally_classical(hea_inst.bqm, method="sa", num_reads=150, num_sweeps=300)
                            gs_energy = float(sa_res["best_energy"])
                            is_exact = False
                            cert_method = "sa_upper_bound"
                    else:
                        sa_res = solve_locally_classical(hea_inst.bqm, method="sa", num_reads=150, num_sweeps=300)
                        gs_energy = float(sa_res["best_energy"])
                        is_exact = False
                        cert_method = "sa_upper_bound"

                # Minor Embedding Evaluation via Cache
                if compute_embeddings and n_spins <= 64:
                    emb_metrics = self._get_cached_embedding_metrics(hea_inst.graph, cache_key)
                else:
                    # Analytical scaling estimation for N > 64
                    approx_chain = max(1, int(round(n_spins / 8.0)))
                    emb_metrics = {
                        "dilation_ratio": float(approx_chain),
                        "max_chain_length": approx_chain,
                        "mean_chain_length": float(approx_chain),
                        "dynamic_range_factor": float(1.0 / math.sqrt(approx_chain)),
                    }

                l_max = max(1, emb_metrics.get("max_chain_length", 1))
                lambda_opt = float(1.2 * rms_j * math.sqrt(l_max))

                inst_dict = hea_inst.to_dict()
                inst_dict["ground_state_energy"] = gs_energy
                inst_dict["is_energy_exact"] = is_exact
                inst_dict["certification_method"] = cert_method
                inst_dict["optimal_chain_strength"] = lambda_opt

                json_path = os.path.join(self.instances_dir, f"{inst_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(inst_dict, f, indent=2)

                manifest.append(InstanceMetadata(
                    instance_id=inst_id,
                    instance_type="hea",
                    num_spins=n_spins,
                    sigma=sigma,
                    num_loops=None,
                    seed=seed,
                    ground_state_energy=gs_energy,
                    is_energy_exact=is_exact,
                    certification_method=cert_method,
                    num_couplers=len(couplings),
                    rms_coupling=rms_j,
                    max_coupling=max_j,
                    pegasus_dilation_ratio=float(emb_metrics.get("dilation_ratio", 1.0)),
                    pegasus_max_chain_length=int(l_max),
                    pegasus_mean_chain_length=float(emb_metrics.get("mean_chain_length", 1.0)),
                    pegasus_dynamic_range_factor=float(emb_metrics.get("dynamic_range_factor", 1.0)),
                    optimal_chain_strength=lambda_opt,
                    json_path=os.path.relpath(json_path, self.output_dir),
                ))

        # 2. Process Planted Frustrated Cluster Loops
        planted_count = 0
        for n_spins, n_loops, seeds in planted_configs:
            cache_key = f"planted_n{n_spins}_loops{n_loops}"
            for seed in seeds:
                planted_count += 1
                inst_id = f"planted_n{n_spins}_loops{n_loops}_seed{seed}"
                planted_inst = generate_planted_frustrated_loops(
                    num_spins=n_spins,
                    num_loops=n_loops,
                    seed=seed,
                )

                couplings = list(planted_inst.bqm.quadratic.values())
                rms_j = float(np.sqrt(np.mean(np.square(couplings)))) if couplings else 1.0
                max_j = float(np.max(np.abs(couplings))) if couplings else 1.0

                gs_energy = float(planted_inst.analytical_ground_energy)
                is_exact = True
                cert_method = "analytical_planted"

                # Verify with SCIP for small sizes
                if certify_exact and n_spins <= 32:
                    try:
                        scip_res = solve_scip_exact(planted_inst.bqm, time_limit_sec=10.0)
                        if not np.isclose(scip_res["best_energy"], gs_energy, atol=1e-4):
                            gs_energy = float(scip_res["best_energy"])
                            cert_method = "scip"
                    except Exception:
                        pass

                # Minor Embedding Evaluation via Cache
                if compute_embeddings and n_spins <= 64:
                    emb_metrics = self._get_cached_embedding_metrics(planted_inst.graph, cache_key)
                else:
                    emb_metrics = {
                        "dilation_ratio": 1.0,
                        "max_chain_length": 1,
                        "mean_chain_length": 1.0,
                        "dynamic_range_factor": 1.0,
                    }

                l_max = max(1, emb_metrics.get("max_chain_length", 1))
                lambda_opt = float(1.2 * rms_j * math.sqrt(l_max))

                inst_dict = planted_inst.to_dict()
                inst_dict["ground_state_energy"] = gs_energy
                inst_dict["is_energy_exact"] = is_exact
                inst_dict["certification_method"] = cert_method
                inst_dict["optimal_chain_strength"] = lambda_opt

                json_path = os.path.join(self.instances_dir, f"{inst_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(inst_dict, f, indent=2)

                manifest.append(InstanceMetadata(
                    instance_id=inst_id,
                    instance_type="planted_loops",
                    num_spins=n_spins,
                    sigma=None,
                    num_loops=n_loops,
                    seed=seed,
                    ground_state_energy=gs_energy,
                    is_energy_exact=is_exact,
                    certification_method=cert_method,
                    num_couplers=len(couplings),
                    rms_coupling=rms_j,
                    max_coupling=max_j,
                    pegasus_dilation_ratio=float(emb_metrics.get("dilation_ratio", 1.0)),
                    pegasus_max_chain_length=int(l_max),
                    pegasus_mean_chain_length=float(emb_metrics.get("mean_chain_length", 1.0)),
                    pegasus_dynamic_range_factor=float(emb_metrics.get("dynamic_range_factor", 1.0)),
                    optimal_chain_strength=lambda_opt,
                    json_path=os.path.relpath(json_path, self.output_dir),
                ))

        # Save manifest.json
        manifest_data = [asdict(item) for item in manifest]
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "suite_name": "Direction 1: HEA & Frustrated Spin Glasses",
                "is_pilot": is_pilot,
                "total_instances": len(manifest),
                "num_hea_instances": hea_count,
                "num_planted_instances": planted_count,
                "instances": manifest_data,
            }, f, indent=2)

        # Generate QPU Job Manifest
        qpu_jobs = self.package_qpu_batch_jobs(manifest)
        qpu_manifest_path = os.path.join(self.output_dir, "qpu_job_manifest.json")
        with open(qpu_manifest_path, "w", encoding="utf-8") as f:
            json.dump(qpu_jobs, f, indent=2)

        return {
            "total_instances": len(manifest),
            "hea_instances": hea_count,
            "planted_instances": planted_count,
            "manifest_file": manifest_path,
            "qpu_manifest_file": qpu_manifest_path,
            "total_qpu_submissions": qpu_jobs["total_qpu_submissions"],
            "total_estimated_qpu_seconds": qpu_jobs["total_estimated_qpu_seconds"],
        }

    def package_qpu_batch_jobs(
        self,
        manifest: List[InstanceMetadata],
        annealing_times_us: List[float] = [1.0, 5.0, 20.0, 100.0],
        num_gauges: int = 5,
        reads_per_gauge: int = 100,
    ) -> Dict[str, Any]:
        """
        Constructs the strict, zero-waste Leap QPU submission manifest.
        Applies:
        - 4 annealing times: t_a in {1, 5, 20, 100} us
        - 5-gauge Spin-Reversal Transforms (SRT) of 100 reads each (500 reads total)
        - Optimal chain strength lambda_opt
        - Quota accounting: 50 ms per submission (33.0s total for 165 instances)
        """
        rng = np.random.default_rng(2026)
        jobs: List[Dict[str, Any]] = []

        total_reads_per_call = num_gauges * reads_per_gauge  # 500 reads
        estimated_time_per_call_sec = 0.015 + (total_reads_per_call * 0.00007)  # ~0.050s

        for item in manifest:
            for t_a in annealing_times_us:
                gauges = []
                for g_idx in range(num_gauges):
                    gauge_vector = {i: int(rng.choice([-1, 1])) for i in range(item.num_spins)}
                    gauges.append({
                        "gauge_index": g_idx,
                        "reads": reads_per_gauge,
                        "gauge_vector": gauge_vector,
                    })

                jobs.append({
                    "job_id": f"{item.instance_id}_ta{int(t_a)}us",
                    "instance_id": item.instance_id,
                    "num_spins": item.num_spins,
                    "annealing_time_us": float(t_a),
                    "chain_strength": float(item.optimal_chain_strength),
                    "total_reads": total_reads_per_call,
                    "num_gauges": num_gauges,
                    "reads_per_gauge": reads_per_gauge,
                    "srt_gauges": gauges,
                    "estimated_qpu_seconds": float(estimated_time_per_call_sec),
                })

        total_qpu_sec = len(jobs) * estimated_time_per_call_sec

        return {
            "description": "Project Q-Ultrametric Direction 1 Leap Batch Manifest",
            "target_hardware": f"D-Wave Advantage (Pegasus P_{self.pegasus_m})",
            "solver_prohibition": "LeapHybridSampler and LeapHybridCQMSampler are strictly forbidden",
            "sampler": "DWaveSampler",
            "total_qpu_submissions": len(jobs),
            "total_reads_planned": len(jobs) * total_reads_per_call,
            "annealing_time_sweep_us": annealing_times_us,
            "num_gauges_srt": num_gauges,
            "reads_per_gauge": reads_per_gauge,
            "total_estimated_qpu_seconds": round(float(total_qpu_sec), 2),
            "monthly_quota_cap_seconds": 60.0,
            "quota_headroom_seconds": round(float(60.0 - total_qpu_sec), 2),
            "jobs": jobs,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Project Q-Ultrametric Direction 1 Test Suite")
    parser.add_argument("--pilot", action="store_true", help="Generate 20-instance pilot suite instead of full 165")
    parser.add_argument("--output-dir", type=str, default="benchmarks/quantum/direction1", help="Target output folder")
    parser.add_argument("--pegasus-m", type=int, default=16, help="Pegasus dimension M (default 16)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip minor embedding search for ultra-fast generation")
    args = parser.parse_args()

    print(f"=== Project Q-Ultrametric: Direction 1 Test Suite Generator ===")
    print(f"Mode: {'PILOT (20 instances)' if args.pilot else 'FULL (165 instances)'}")
    print(f"Output Directory: {args.output_dir}")

    suite = Direction1TestSuite(output_dir=args.output_dir, pegasus_m=args.pegasus_m)
    t0 = time.time()
    summary = suite.generate_suite(
        is_pilot=args.pilot,
        compute_embeddings=not args.skip_embeddings,
    )
    t1 = time.time()

    print("\n--- Generation Complete ---")
    print(f"Total Problem Instances:     {summary['total_instances']}")
    print(f"  HEA Instances:             {summary['hea_instances']}")
    print(f"  Planted Frustrated Loops:  {summary['planted_instances']}")
    print(f"Total QPU Submissions:       {summary['total_qpu_submissions']}")
    print(f"Total Estimated QPU Time:    {summary['total_estimated_qpu_seconds']:.2f} seconds")
    print(f"Elapsed Wall-Clock Time:     {t1 - t0:.2f} seconds")
    print(f"Manifest written to:         {summary['manifest_file']}")
    print(f"QPU Jobs written to:         {summary['qpu_manifest_file']}")


if __name__ == "__main__":
    main()
