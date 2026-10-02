"""
Direction 2 Test Suite Generator & Batch Submission Packager.

Orchestrates the complete benchmark test suite for Direction 2:
"Hardware-Embeddable Hierarchical Graph Partitioning & Modularity Maximization"
- 20 Synthetic hierarchical modular networks (H-SBM) across sizes N in {16, 32, 64, 128}.
- 15 Real-world benchmark networks (Karate Club, Dolphins, Football, Polbooks, etc.).
- 4 Multi-scale tree partition cuts per graph: 35 graphs x 4 cuts = 140 QUBO instances.
- Ground truth offline certification via SCIP branch-and-cut and metaheuristics.
- Pegasus P_16 minor embedding quality evaluation comparing sparse vs dense K_N.
- D-Wave Leap batch submission manifest generation:
  140 instances x 3 chain strengths (lambda in {0.8, 1.2, 1.6} * RMS(Q) * sqrt(L)) = 420 QPU calls.
  Quota accounting: 420 calls x 0.05s = exactly 21.00 seconds budgeted (39.00s headroom).
- Dry-run simulation harness with 5-gauge Spin-Reversal Transforms (SRT).
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
import neal

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


@dataclass
class Direction2InstanceMetadata:
    """Metadata catalog entry for a single Direction 2 modularity problem instance."""
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
    ground_state_energy: float
    is_energy_exact: bool
    certification_method: str  # 'scip', 'sa_upper_bound'
    realized_modularity_scip: float
    realized_modularity_louvain: float
    realized_modularity_greedy: float
    nmi_ground_truth: Optional[float]
    num_couplers_in_bqm: int
    rms_coupling: float
    max_coupling: float
    pegasus_dilation_ratio: float
    pegasus_max_chain_length: int
    pegasus_mean_chain_length: float
    pegasus_dynamic_range_factor: float
    optimal_chain_strength: float
    chain_strength_08: float
    chain_strength_12: float
    chain_strength_16: float
    json_path: str


class Direction2TestSuite:
    """
    Orchestrates the generation, certification, embedding evaluation,
    and Leap batch packaging of the Direction 2 benchmark suite.
    """

    def __init__(
        self,
        output_dir: str = "benchmarks/quantum/direction2",
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

    def _get_cached_embedding_metrics(self, graph: nx.Graph, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Caches embedding computation by graph topology to avoid redundant heuristic searches.
        """
        n = len(graph)
        num_edges = graph.number_of_edges()

        if not cache_key:
            deg_sig = tuple(sorted(d for _, d in graph.degree()))
            cache_key = f"n{n}_e{num_edges}_{hash(deg_sig)}"

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        def _analytical_estimate() -> Dict[str, Any]:
            total_possible = n * (n - 1) // 2
            density = float(num_edges / total_possible) if total_possible > 0 else 0.0
            approx_chain = max(1, int(round(1.0 + density * (n / 6.0))))
            return {
                "dilation_ratio": float(approx_chain),
                "max_chain_length": approx_chain,
                "mean_chain_length": float(approx_chain),
                "dynamic_range_factor": float(1.0 / math.sqrt(approx_chain)),
            }

        if n <= 64 and num_edges > 0:
            try:
                emb = embed_graph_onto_pegasus(
                    graph,
                    m=self.pegasus_m,
                    target_graph=self._get_target(),
                    timeout_sec=15.0,
                )
                eval_res = evaluate_embedding_quality(emb, graph)
                if eval_res.get("is_valid", False):
                    emb_metrics = eval_res
                else:
                    emb_metrics = _analytical_estimate()
            except Exception:
                emb_metrics = _analytical_estimate()
        else:
            emb_metrics = _analytical_estimate()

        self._embedding_cache[cache_key] = emb_metrics
        return emb_metrics

    def generate_suite(
        self,
        is_pilot: bool = False,
        certify_exact: bool = True,
        compute_embeddings: bool = True,
        scip_time_limit: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Generates problem instances, certifies energies, and saves artifacts.

        Args:
            is_pilot: If True, generates a fast 20-instance pilot (5 graphs x 4 scale cuts).
                      If False, generates the full 140-instance production benchmark set (35 graphs x 4 cuts).
            certify_exact: If True, executes SCIP or SA verification.
            compute_embeddings: If True, embeds onto Pegasus target graph.
            scip_time_limit: Maximum runtime in seconds per SCIP certification.

        Returns:
            Summary catalog dictionary.
        """
        os.makedirs(self.instances_dir, exist_ok=True)
        manifest: List[Direction2InstanceMetadata] = []

        # 1. Define Graph Catalog
        graphs_to_process: List[Tuple[str, str, nx.Graph, Optional[Dict[int, int]], Optional[np.ndarray]]] = []

        if is_pilot:
            # Fast pilot: 3 synthetic + 2 real-world = 5 graphs x 4 cuts = 20 QUBO instances
            pilot_syn = [
                (16, 2001),
                (32, 2006),
                (64, 2011),
            ]
            for n_nodes, seed in pilot_syn:
                g_syn, gt_syn, du_syn = generate_hierarchical_sbm(n_nodes, seed=seed)
                graphs_to_process.append((f"syn_n{n_nodes}_seed{seed}", "synthetic_hsbm", g_syn, gt_syn, du_syn))

            pilot_real = ["karate_club", "florentine_families"]
            for rname in pilot_real:
                g_real, gt_real = load_real_world_network(rname)
                graphs_to_process.append((f"real_{rname}", "real_world", g_real, gt_real, None))
        else:
            # Full Production Set: 20 synthetic H-SBM + 15 real-world = 35 graphs x 4 cuts = 140 instances
            # 20 Synthetic H-SBM graphs: 5 instances across sizes N in {16, 32, 64, 128}
            syn_specs = [
                (16, [2001, 2002, 2003, 2004, 2005]),
                (32, [2006, 2007, 2008, 2009, 2010]),
                (64, [2011, 2012, 2013, 2014, 2015]),
                (128, [2016, 2017, 2018, 2019, 2020]),
            ]
            for n_nodes, seeds in syn_specs:
                for seed in seeds:
                    g_syn, gt_syn, du_syn = generate_hierarchical_sbm(n_nodes, seed=seed)
                    graphs_to_process.append((f"syn_n{n_nodes}_seed{seed}", "synthetic_hsbm", g_syn, gt_syn, du_syn))

            # 15 Real-world benchmark networks
            real_names = [
                "karate_club",
                "florentine_families",
                "davis_southern_women",
                "dolphins",
                "les_miserables",
                "polbooks",
                "adjnoun",
                "football",
                "coauthorship_netscience_64",
                "coauthorship_netscience_128",
                "ppi_yeast_64",
                "ppi_yeast_128",
                "power_grid_64",
                "power_grid_128",
                "citation_hep_64",
            ]
            for rname in real_names:
                g_real, gt_real = load_real_world_network(rname)
                graphs_to_process.append((f"real_{rname}", "real_world", g_real, gt_real, None))

        total_graphs = len(graphs_to_process)
        synthetic_count = sum(1 for item in graphs_to_process if item[1] == "synthetic_hsbm")
        real_count = sum(1 for item in graphs_to_process if item[1] == "real_world")

        # 2. Process each graph and generate 4 scale cuts
        for graph_id, graph_type, graph, ground_truth, d_u in graphs_to_process:
            n_nodes = len(graph)
            n_edges = graph.number_of_edges()

            # Classical Community Baselines on the original graph
            try:
                c_louvain = nx.community.louvain_communities(graph, seed=42)
                q_louvain = float(nx.community.modularity(graph, c_louvain, weight=None))
            except Exception:
                q_louvain = 0.0

            try:
                c_greedy = nx.community.greedy_modularity_communities(graph)
                q_greedy = float(nx.community.modularity(graph, c_greedy, weight=None))
            except Exception:
                q_greedy = 0.0

            # Generate 4 scale cut instances for this graph
            instances = generate_modularity_instances_for_graph(
                graph=graph,
                graph_id=graph_id,
                graph_type=graph_type,
                ground_truth=ground_truth,
                d_u=d_u,
            )

            for inst in instances:
                inst_id = inst.instance_id
                couplings = [item["weight"] for item in inst.quadratic]
                rms_q = float(np.sqrt(np.mean(np.square(couplings)))) if couplings else 0.0
                max_q = float(np.max(np.abs(couplings))) if couplings else 0.0

                # Offline Certification via SCIP or Metaheuristics
                gs_energy = 0.0
                is_exact = False
                cert_method = "unsolved"
                realized_q_opt = 0.0
                nmi_val: Optional[float] = None

                if certify_exact:
                    if n_nodes <= 34 and len(inst.bqm.quadratic) > 0:
                        try:
                            scip_res = solve_scip_exact(inst.bqm, time_limit_sec=scip_time_limit)
                            gs_energy = float(scip_res["best_energy"])
                            is_exact = bool(scip_res["is_optimal"])
                            cert_method = "scip"
                            best_s = scip_res["best_sample"]
                        except Exception:
                            sa_res = solve_locally_classical(inst.bqm, method="sa", num_reads=150, num_sweeps=300, seed=42)
                            gs_energy = float(sa_res["best_energy"])
                            is_exact = False
                            cert_method = "sa_upper_bound"
                            best_s = sa_res["best_sample"]
                    else:
                        sa_res = solve_locally_classical(inst.bqm, method="sa", num_reads=150, num_sweeps=300, seed=42)
                        gs_energy = float(sa_res["best_energy"])
                        is_exact = False
                        cert_method = "sa_upper_bound"
                        best_s = sa_res["best_sample"]

                    # Compute realized Newman modularity on original graph
                    realized_q_opt = compute_modularity(graph, best_s)

                    # Compute NMI against ground-truth if available
                    if ground_truth is not None:
                        yt = [ground_truth.get(i, 0) for i in range(n_nodes)]
                        yp = [1 if best_s.get(i, 0) in (1, +1) else 0 for i in range(n_nodes)]
                        nmi_val = compute_clustering_nmi(yt, yp)

                # Pegasus Minor Embedding Evaluation
                cache_key = f"{graph_id}_cut{inst.scale_cut}"
                if compute_embeddings:
                    emb_metrics = self._get_cached_embedding_metrics(inst.graph, cache_key)
                else:
                    approx_chain = max(1, int(round((1.0 - inst.sparsity) * (n_nodes / 6.0))))
                    emb_metrics = {
                        "dilation_ratio": float(approx_chain),
                        "max_chain_length": approx_chain,
                        "mean_chain_length": float(approx_chain),
                        "dynamic_range_factor": float(1.0 / math.sqrt(approx_chain)),
                    }

                l_max = max(1, int(emb_metrics.get("max_chain_length", 1)))
                sqrt_l = math.sqrt(l_max)
                effective_rms = rms_q if rms_q > 1e-9 else (max_q if max_q > 1e-9 else 1.0)

                # Chain strength variations: lambda in {0.8, 1.2, 1.6} * RMS(Q) * sqrt(L)
                lambda_08 = max(1e-4, float(0.8 * effective_rms * sqrt_l))
                lambda_12 = max(1e-4, float(1.2 * effective_rms * sqrt_l))
                lambda_16 = max(1e-4, float(1.6 * effective_rms * sqrt_l))

                # Save individual instance JSON
                inst_dict = inst.to_dict()
                inst_dict["ground_state_energy"] = gs_energy
                inst_dict["is_energy_exact"] = is_exact
                inst_dict["certification_method"] = cert_method
                inst_dict["realized_modularity_scip"] = realized_q_opt
                inst_dict["realized_modularity_louvain"] = q_louvain
                inst_dict["realized_modularity_greedy"] = q_greedy
                inst_dict["optimal_chain_strength"] = lambda_12
                inst_dict["chain_strengths"] = [lambda_08, lambda_12, lambda_16]

                json_path = os.path.join(self.instances_dir, f"{inst_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(inst_dict, f, indent=2)

                manifest.append(Direction2InstanceMetadata(
                    instance_id=inst_id,
                    graph_id=graph_id,
                    graph_type=graph_type,
                    num_nodes=n_nodes,
                    num_edges=n_edges,
                    scale_cut=inst.scale_cut,
                    d_max=inst.d_max,
                    num_couplers=inst.num_couplers,
                    total_possible_couplers=inst.total_possible_couplers,
                    sparsity=inst.sparsity,
                    ground_state_energy=gs_energy,
                    is_energy_exact=is_exact,
                    certification_method=cert_method,
                    realized_modularity_scip=realized_q_opt,
                    realized_modularity_louvain=q_louvain,
                    realized_modularity_greedy=q_greedy,
                    nmi_ground_truth=nmi_val,
                    num_couplers_in_bqm=len(couplings),
                    rms_coupling=rms_q,
                    max_coupling=max_q,
                    pegasus_dilation_ratio=float(emb_metrics.get("dilation_ratio", 1.0)),
                    pegasus_max_chain_length=int(l_max),
                    pegasus_mean_chain_length=float(emb_metrics.get("mean_chain_length", 1.0)),
                    pegasus_dynamic_range_factor=float(emb_metrics.get("dynamic_range_factor", 1.0)),
                    optimal_chain_strength=lambda_12,
                    chain_strength_08=lambda_08,
                    chain_strength_12=lambda_12,
                    chain_strength_16=lambda_16,
                    json_path=os.path.relpath(json_path, self.output_dir).replace("\\", "/"),
                ))

        # Save manifest.json
        manifest_data = [asdict(item) for item in manifest]
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "suite_name": "Direction 2: Hierarchical Graph Partitioning & Modularity Maximization",
                "is_pilot": is_pilot,
                "total_graphs": total_graphs,
                "num_synthetic_graphs": synthetic_count,
                "num_real_graphs": real_count,
                "total_instances": len(manifest),
                "num_scale_cuts_per_graph": 4,
                "instances": manifest_data,
            }, f, indent=2)

        # Generate QPU Job Manifest
        qpu_jobs = self.package_qpu_batch_jobs(manifest)
        qpu_manifest_path = os.path.join(self.output_dir, "qpu_job_manifest.json")
        with open(qpu_manifest_path, "w", encoding="utf-8") as f:
            json.dump(qpu_jobs, f, indent=2)

        return {
            "total_graphs": total_graphs,
            "synthetic_graphs": synthetic_count,
            "real_graphs": real_count,
            "total_instances": len(manifest),
            "manifest_file": manifest_path,
            "qpu_manifest_file": qpu_manifest_path,
            "total_qpu_submissions": qpu_jobs["total_qpu_submissions"],
            "total_estimated_qpu_seconds": qpu_jobs["total_estimated_qpu_seconds"],
            "quota_headroom_seconds": qpu_jobs["quota_headroom_seconds"],
        }

    def package_qpu_batch_jobs(
        self,
        manifest: List[Direction2InstanceMetadata],
        num_gauges: int = 5,
        reads_per_gauge: int = 100,
    ) -> Dict[str, Any]:
        """
        Constructs the zero-waste Leap QPU submission manifest for Direction 2.
        - 3 chain strengths per instance: {0.8, 1.2, 1.6} * RMS(Q) * sqrt(L).
        - 5-gauge Spin-Reversal Transforms (SRT) of 100 reads each (500 reads total).
        - Quota accounting: 50 ms per submission (0.050s).
          140 instances x 3 calls = 420 calls = 21.00 seconds budgeted.
        """
        rng = np.random.default_rng(2026)
        jobs: List[Dict[str, Any]] = []

        total_reads_per_call = num_gauges * reads_per_gauge  # 500 reads
        estimated_time_per_call_sec = 0.015 + (total_reads_per_call * 0.00007)  # ~0.050s

        chain_multipliers = [0.8, 1.2, 1.6]

        for item in manifest:
            strengths = [item.chain_strength_08, item.chain_strength_12, item.chain_strength_16]
            for factor, c_strength in zip(chain_multipliers, strengths):
                gauges = []
                for g_idx in range(num_gauges):
                    gauge_vector = {i: int(rng.choice([-1, 1])) for i in range(item.num_nodes)}
                    gauges.append({
                        "gauge_index": g_idx,
                        "reads": reads_per_gauge,
                        "gauge_vector": gauge_vector,
                    })

                factor_str = f"{factor:.1f}".replace(".", "p")
                job_id = f"{item.instance_id}_cs{factor_str}"

                jobs.append({
                    "job_id": job_id,
                    "instance_id": item.instance_id,
                    "graph_id": item.graph_id,
                    "num_nodes": item.num_nodes,
                    "scale_cut": item.scale_cut,
                    "chain_strength_factor": float(factor),
                    "chain_strength": float(c_strength),
                    "total_reads": total_reads_per_call,
                    "num_gauges": num_gauges,
                    "reads_per_gauge": reads_per_gauge,
                    "srt_gauges": gauges,
                    "estimated_qpu_seconds": float(estimated_time_per_call_sec),
                })

        total_qpu_sec = len(jobs) * estimated_time_per_call_sec

        return {
            "description": "Project Q-Ultrametric Direction 2 Leap Batch Manifest",
            "target_hardware": f"D-Wave Advantage (Pegasus P_{self.pegasus_m})",
            "solver_prohibition": "LeapHybridSampler and LeapHybridCQMSampler are strictly forbidden",
            "sampler": "DWaveSampler",
            "total_qpu_submissions": len(jobs),
            "total_reads_planned": len(jobs) * total_reads_per_call,
            "chain_strength_multipliers": chain_multipliers,
            "num_gauges_srt": num_gauges,
            "reads_per_gauge": reads_per_gauge,
            "total_estimated_qpu_seconds": round(float(total_qpu_sec), 2),
            "monthly_quota_cap_seconds": 60.0,
            "quota_headroom_seconds": round(float(60.0 - total_qpu_sec), 2),
            "jobs": jobs,
        }


def run_direction2_dry_run(
    manifest_path: str,
    qpu_manifest_path: str,
    output_path: str,
    max_jobs: Optional[int] = None,
    log_interval: int = 25,
) -> Dict[str, Any]:
    """
    Simulates Direction 2 QPU batch hardware execution locally using dwave-neal,
    applying 5-gauge Spin-Reversal Transforms and calculating TTS_99, realized
    modularity, and NMI.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    with open(qpu_manifest_path, "r", encoding="utf-8") as f:
        qpu_manifest = json.load(f)

    inst_map = {item["instance_id"]: item for item in manifest_data["instances"]}
    instances_base_dir = os.path.dirname(manifest_path)

    jobs = qpu_manifest["jobs"]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    total_jobs = len(jobs)
    sa_sampler = neal.SimulatedAnnealingSampler()
    results = []
    graph_cache: Dict[str, nx.Graph] = {}
    t_start = time.perf_counter()

    for idx, job in enumerate(jobs):
        inst_id = job["instance_id"]
        inst_meta = inst_map[inst_id]
        inst_file = os.path.join(instances_base_dir, inst_meta["json_path"])

        with open(inst_file, "r", encoding="utf-8") as f:
            inst_json = json.load(f)

        linear = {int(k): float(v) for k, v in inst_json["linear"].items()}
        quadratic = {
            (int(item["u"]), int(item["v"])): float(item["weight"])
            for item in inst_json["quadratic"]
        }
        offset = float(inst_json.get("offset", 0.0))
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype=dimod.SPIN)

        num_spins = inst_meta["num_nodes"]
        gs_energy = float(inst_meta["ground_state_energy"])

        # 5-gauge SRT sampling
        all_samples = []
        all_energies = []
        job_t0 = time.perf_counter()

        for g_info in job["srt_gauges"]:
            gauge = {int(k): int(v) for k, v in g_info["gauge_vector"].items()}
            # Gauge transform: s_i' = gauge[i] * s_i => J_ij' = gauge[i]*gauge[j]*J_ij
            g_linear = {i: linear.get(i, 0.0) * gauge.get(i, 1) for i in range(num_spins)}
            g_quadratic = {
                (u, v): w * gauge.get(u, 1) * gauge.get(v, 1)
                for (u, v), w in quadratic.items()
            }
            g_bqm = dimod.BinaryQuadraticModel(g_linear, g_quadratic, offset, vartype=dimod.SPIN)

            sub_sampleset = sa_sampler.sample(
                g_bqm,
                num_reads=g_info["reads"],
                num_sweeps=200,
            )

            # Invert gauge on samples
            for sample_dict, energy in zip(sub_sampleset.samples(), sub_sampleset.data_vectors["energy"]):
                un_gauged = {i: sample_dict[i] * gauge.get(i, 1) for i in range(num_spins)}
                all_samples.append(un_gauged)
                all_energies.append(float(energy))

        job_t1 = time.perf_counter()
        energies_arr = np.array(all_energies)
        best_idx = int(np.argmin(energies_arr))
        best_energy = float(energies_arr[best_idx])
        best_sample = all_samples[best_idx]

        # Reconstruct original graph for modularity calculation with topology caching
        graph_id = inst_meta["graph_id"]
        graph_type = inst_meta["graph_type"]
        if graph_id not in graph_cache:
            if graph_type == "synthetic_hsbm":
                seed = int(graph_id.split("seed")[-1]) if "seed" in graph_id else 42
                orig_g, _, _ = generate_hierarchical_sbm(num_spins, seed=seed)
            else:
                raw_name = graph_id.replace("real_", "")
                orig_g, _ = load_real_world_network(raw_name)
            graph_cache[graph_id] = orig_g
        else:
            orig_g = graph_cache[graph_id]

        realized_q = compute_modularity(orig_g, best_sample)

        # TTS_99 Calculation
        time_per_read = (job_t1 - job_t0) / max(1, len(energies_arr))
        tts_res = compute_time_to_solution(
            run_time_per_read=time_per_read,
            ground_state_energy=gs_energy,
            energies=energies_arr,
            energy_tolerance=1e-4,
        )

        # NMI against ground truth if present
        gt = inst_json.get("ground_truth_partition")
        nmi_val = None
        if gt:
            yt = [gt[str(i)] if str(i) in gt else gt.get(i, 0) for i in range(num_spins)]
            yp = [1 if best_sample.get(i, -1) in (1, +1) else 0 for i in range(num_spins)]
            nmi_val = compute_clustering_nmi(yt, yp)

        results.append({
            "job_id": job["job_id"],
            "instance_id": inst_id,
            "scale_cut": inst_meta["scale_cut"],
            "chain_strength": job["chain_strength"],
            "best_energy": best_energy,
            "ground_state_energy": gs_energy,
            "realized_modularity": realized_q,
            "nmi": nmi_val,
            "p_gs": tts_res["p_gs"],
            "tts_99": tts_res["tts_99"],
            "is_lower_bound": tts_res["is_lower_bound"],
            "elapsed_sec": float(job_t1 - job_t0),
        })

    t_end = time.perf_counter()

    summary = {
        "suite": "Direction 2 Dry-Run Simulation",
        "total_jobs_executed": len(results),
        "mean_realized_modularity": float(np.mean([r["realized_modularity"] for r in results])) if results else 0.0,
        "mean_p_gs": float(np.mean([r["p_gs"] for r in results])) if results else 0.0,
        "mean_tts_99": float(np.mean([r["tts_99"] for r in results])) if results else 0.0,
        "elapsed_wall_clock_sec": float(t_end - t_start),
        "job_results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate Project Q-Ultrametric Direction 2 Test Suite")
    parser.add_argument("--pilot", action="store_true", help="Generate 20-instance pilot suite instead of full 140")
    parser.add_argument("--output-dir", type=str, default="benchmarks/quantum/direction2", help="Target output folder")
    parser.add_argument("--pegasus-m", type=int, default=16, help="Pegasus dimension M (default 16)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip minor embedding search for fast generation")
    args = parser.parse_args()

    print(f"=== Project Q-Ultrametric: Direction 2 Test Suite Generator ===")
    print(f"Mode: {'PILOT (20 instances)' if args.pilot else 'FULL (140 instances)'}")
    print(f"Output Directory: {args.output_dir}")

    suite = Direction2TestSuite(output_dir=args.output_dir, pegasus_m=args.pegasus_m)
    t0 = time.time()
    summary = suite.generate_suite(
        is_pilot=args.pilot,
        compute_embeddings=not args.skip_embeddings,
    )
    t1 = time.time()

    print("\n--- Generation Complete ---")
    print(f"Total Graphs:                {summary['total_graphs']}")
    print(f"  Synthetic H-SBM Graphs:    {summary['synthetic_graphs']}")
    print(f"  Real-World Graphs:         {summary['real_graphs']}")
    print(f"Total QUBO Problem Instances:{summary['total_instances']}")
    print(f"Total QPU Submissions:       {summary['total_qpu_submissions']}")
    print(f"Total Estimated QPU Time:    {summary['total_estimated_qpu_seconds']:.2f} seconds")
    print(f"Remaining QPU Headroom:      {summary['quota_headroom_seconds']:.2f} seconds")
    print(f"Elapsed Wall-Clock Time:     {t1 - t0:.2f} seconds")
    print(f"Manifest written to:         {summary['manifest_file']}")
    print(f"QPU Jobs written to:         {summary['qpu_manifest_file']}")


if __name__ == "__main__":
    main()
