"""
D-Wave Leap Hardware Submission & Local Dry-Run Simulator.

Executes the Direction 1 QPU job manifest:
- --dry-run: Simulates D-Wave QPU hardware execution 100% locally via
             classical simulated annealing with Spin-Reversal Transforms (SRT),
             computing TTS_99 and P_gs metrics against certified ground truths ($0 cost).
- --live: Submits batch queries strictly to pure DWaveSampler on Pegasus P_16,
          enforcing the architectural ban on LeapHybridSampler to protect monthly quota.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import dimod
import neal

from src.quantum.solvers import compute_time_to_solution


def test_qpu_connection() -> Dict[str, Any]:
    """
    Validates Leap credentials and checks availability of Pegasus Advantage solvers
    without consuming any QPU annealing time.
    """
    try:
        from dwave.cloud import Client
    except ImportError:
        return {"connected": False, "error": "dwave-cloud-client not installed"}

    try:
        with Client.from_config() as client:
            solvers = client.get_solvers(topology__type="pegasus")
            solver_info = [
                {
                    "id": s.id,
                    "num_qubits": s.properties.get("num_qubits", 0),
                    "is_online": s.online,
                    "avg_anneal_time_range": s.properties.get("annealing_time_range", [1.0, 2000.0]),
                }
                for s in solvers
            ]
            return {
                "connected": True,
                "token_present": True,
                "pegasus_solvers": solver_info,
            }
    except Exception as e:
        return {
            "connected": False,
            "token_present": bool(os.environ.get("DWAVE_API_TOKEN")),
            "error": str(e),
        }


def run_dry_run_simulation(
    manifest_path: str,
    qpu_manifest_path: str,
    output_path: str,
    max_jobs: Optional[int] = None,
    log_interval: int = 25,
) -> Dict[str, Any]:
    """
    Simulates hardware batch execution locally using dwave-neal, applying
    5-gauge Spin-Reversal Transforms and calculating TTS_99.
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
    t_start = time.perf_counter()

    print(f"Starting Dry-Run Simulation: {total_jobs} total QPU jobs with 5-gauge SRT (500 reads/job)")

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
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)

        # Execute 5 SRT gauges
        all_energies = []
        for gauge_info in job["srt_gauges"]:
            gauge_vec = {int(k): int(v) for k, v in gauge_info["gauge_vector"].items()}
            num_reads = gauge_info["reads"]

            # Gauge transform: h'_i = h_i * g_i, J'_{ij} = J_{ij} * g_i * g_j
            gauged_linear = {i: linear[i] * gauge_vec[i] for i in linear}
            gauged_quadratic = {
                (u, v): quadratic[(u, v)] * gauge_vec[u] * gauge_vec[v]
                for (u, v) in quadratic
            }
            gauged_bqm = dimod.BinaryQuadraticModel(gauged_linear, gauged_quadratic, 0.0, vartype=dimod.SPIN)

            # Sample on simulated annealer (simulating QPU readouts)
            sim_sweeps = min(250, max(50, int(job["annealing_time_us"] * 5)))
            sampleset = sa_sampler.sample(
                gauged_bqm,
                num_reads=num_reads,
                num_sweeps=sim_sweeps,
                seed=job["num_spins"] + gauge_info["gauge_index"],
            )

            # Gauge untransform: s_i = s'_i * g_i
            for sample, energy in zip(sampleset.record.sample, sampleset.record.energy):
                all_energies.append(float(energy))

        energies_arr = np.array(all_energies)
        target_gs = float(inst_meta["ground_state_energy"])

        # Compute TTS_99 using 0.07 ms read time (D-Wave standard read time)
        tts_res = compute_time_to_solution(
            run_time_per_read=0.00007,
            ground_state_energy=target_gs,
            energies=energies_arr,
            energy_tolerance=0.05,
        )

        results.append({
            "job_id": job["job_id"],
            "instance_id": inst_id,
            "num_spins": job["num_spins"],
            "annealing_time_us": job["annealing_time_us"],
            "chain_strength": job["chain_strength"],
            "target_gs_energy": target_gs,
            "min_observed_energy": float(np.min(energies_arr)),
            "p_gs": float(tts_res["p_gs"]),
            "success_count": int(tts_res["success_count"]),
            "tts_99_sec": float(tts_res["tts_99"]),
            "is_tts_lower_bound": bool(tts_res["is_lower_bound"]),
            "total_reads": len(all_energies),
        })

        if (idx + 1) % log_interval == 0 or (idx + 1) == total_jobs:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / max(0.01, elapsed)
            remaining_sec = (total_jobs - (idx + 1)) / max(0.01, rate)
            print(f"[{idx+1:03d}/{total_jobs:03d}] {job['job_id']:<35} | P_gs: {tts_res['p_gs']*100:5.1f}% | TTS: {tts_res['tts_99']*1e3:6.2f} ms | Elapsed: {elapsed:5.1f}s | ETA: {remaining_sec:5.1f}s")

    t_end = time.perf_counter()

    summary = {
        "execution_mode": "dry_run_simulation",
        "total_jobs_executed": len(results),
        "total_reads_simulated": sum(r["total_reads"] for r in results),
        "wall_time_sec": float(t_end - t_start),
        "mean_p_gs": float(np.mean([r["p_gs"] for r in results])) if results else 0.0,
        "median_tts_99_sec": float(np.median([r["tts_99_sec"] for r in results])) if results else 0.0,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_live_qpu_submission(
    manifest_path: str,
    qpu_manifest_path: str,
    output_path: str,
    confirm_live: bool = False,
    max_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Submits batch jobs to physical D-Wave Advantage QPU using pure DWaveSampler.
    Strictly safeguards monthly quota and forbids LeapHybridSampler.
    """
    if not confirm_live:
        print("\n" + "="*70)
        print("SAFETY GUARD ACTIVE: Live QPU Submission Not Confirmed.")
        print("Running live submissions consumes physical D-Wave QPU quota (60s/mo limit).")
        print("To execute against physical D-Wave Advantage hardware, provide:")
        print("    --live --confirm-live-submission")
        print("="*70 + "\n")
        return {"status": "unconfirmed", "message": "Specify --confirm-live-submission to execute on QPU."}

    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
    except ImportError:
        raise RuntimeError("dwave-system is required for live hardware submissions.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    with open(qpu_manifest_path, "r", encoding="utf-8") as f:
        qpu_manifest = json.load(f)

    inst_map = {item["instance_id"]: item for item in manifest_data["instances"]}
    instances_base_dir = os.path.dirname(manifest_path)

    print("Connecting to D-Wave Advantage QPU via Leap API...")
    raw_sampler = DWaveSampler(solver={"topology__type": "pegasus"})
    print(f"Connected to solver: {raw_sampler.solver.id} ({raw_sampler.properties.get('num_qubits', 'unknown')} fabric qubits)")
    sampler = EmbeddingComposite(raw_sampler)

    jobs = qpu_manifest["jobs"]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    results = []
    t_start = time.perf_counter()
    total_qpu_access_time_us = 0.0

    print(f"Beginning physical submission of {len(jobs)} jobs...")

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
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.SPIN)

        all_energies = []
        job_qpu_time_us = 0.0

        for gauge_info in job["srt_gauges"]:
            gauge_vec = {int(k): int(v) for k, v in gauge_info["gauge_vector"].items()}
            num_reads = gauge_info["reads"]

            gauged_linear = {i: linear[i] * gauge_vec[i] for i in linear}
            gauged_quadratic = {
                (u, v): quadratic[(u, v)] * gauge_vec[u] * gauge_vec[v]
                for (u, v) in quadratic
            }
            gauged_bqm = dimod.BinaryQuadraticModel(gauged_linear, gauged_quadratic, 0.0, vartype=dimod.SPIN)

            sampleset = sampler.sample(
                gauged_bqm,
                num_reads=num_reads,
                annealing_time=job["annealing_time_us"],
                chain_strength=job["chain_strength"],
                auto_scale=True,
            )

            if "timing" in sampleset.info:
                job_qpu_time_us += sampleset.info["timing"].get("qpu_access_time", 0.0)

            for energy in sampleset.record.energy:
                all_energies.append(float(energy))

        total_qpu_access_time_us += job_qpu_time_us
        energies_arr = np.array(all_energies)
        target_gs = float(inst_meta["ground_state_energy"])

        tts_res = compute_time_to_solution(
            run_time_per_read=float(job["annealing_time_us"]) * 1e-6,
            ground_state_energy=target_gs,
            energies=energies_arr,
            energy_tolerance=0.05,
        )

        results.append({
            "job_id": job["job_id"],
            "instance_id": inst_id,
            "num_spins": job["num_spins"],
            "annealing_time_us": job["annealing_time_us"],
            "chain_strength": job["chain_strength"],
            "target_gs_energy": target_gs,
            "min_observed_energy": float(np.min(energies_arr)),
            "p_gs": float(tts_res["p_gs"]),
            "success_count": int(tts_res["success_count"]),
            "tts_99_sec": float(tts_res["tts_99"]),
            "is_tts_lower_bound": bool(tts_res["is_lower_bound"]),
            "qpu_access_time_ms": float(job_qpu_time_us / 1000.0),
            "total_reads": len(all_energies),
        })

        print(f"[{idx+1:03d}/{len(jobs):03d}] QPU: {job['job_id']:<35} | P_gs: {tts_res['p_gs']*100:5.1f}% | QPU Time: {job_qpu_time_us/1000.0:6.1f} ms")

    t_end = time.perf_counter()

    summary = {
        "execution_mode": "physical_dwave_qpu",
        "solver_id": raw_sampler.solver.id,
        "total_jobs_executed": len(results),
        "total_reads_harvested": sum(r["total_reads"] for r in results),
        "wall_time_sec": float(t_end - t_start),
        "total_qpu_access_seconds": float(total_qpu_access_time_us / 1e6),
        "mean_p_gs": float(np.mean([r["p_gs"] for r in results])) if results else 0.0,
        "median_tts_99_sec": float(np.median([r["tts_99_sec"] for r in results])) if results else 0.0,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="D-Wave Leap Runner & Local Simulator")
    parser.add_argument("--manifest", type=str, default="benchmarks/quantum/direction1/manifest.json")
    parser.add_argument("--qpu-manifest", type=str, default="benchmarks/quantum/direction1/qpu_job_manifest.json")
    parser.add_argument("--output", type=str, default="benchmarks/quantum/direction1/benchmark_results.json")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate execution locally ($0 budget)")
    parser.add_argument("--live", action="store_true", help="Submit to physical D-Wave Advantage QPU")
    parser.add_argument("--confirm-live-submission", action="store_true", help="Confirm execution on physical QPU")
    parser.add_argument("--test-connection", action="store_true", help="Test Leap credentials and query solvers without spending quota")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit number of jobs to run")
    parser.add_argument("--log-interval", type=int, default=25, help="Logging cadence")
    args = parser.parse_args()

    if args.test_connection:
        print("Testing D-Wave Leap Connection...")
        conn = test_qpu_connection()
        print(json.dumps(conn, indent=2))
        return

    if args.live:
        summary = run_live_qpu_submission(
            args.manifest,
            args.qpu_manifest,
            args.output,
            confirm_live=args.confirm_live_submission,
            max_jobs=args.max_jobs,
        )
        if summary.get("status") == "unconfirmed":
            return
        print("\n--- Physical QPU Execution Summary ---")
        print(f"Total Jobs Dispatched:       {summary['total_jobs_executed']}")
        print(f"Total QPU Access Time:       {summary['total_qpu_access_seconds']:.3f} seconds")
        print(f"Mean Ground State Prob:      {summary['mean_p_gs']*100:.2f}%")
        print(f"Results Catalog:             {args.output}")
    else:
        print(f"=== Project Q-Ultrametric: Direction 1 Dry-Run Simulation ===")
        summary = run_dry_run_simulation(
            args.manifest,
            args.qpu_manifest,
            args.output,
            max_jobs=args.max_jobs,
            log_interval=args.log_interval,
        )
        print("\n--- Simulation Summary ---")
        print(f"Total Jobs Executed:         {summary['total_jobs_executed']}")
        print(f"Total Reads Simulated:       {summary['total_reads_simulated']:,}")
        print(f"Wall-Clock Time:             {summary['wall_time_sec']:.2f} seconds")
        print(f"Mean Ground State Prob:      {summary['mean_p_gs']*100:.2f}%")
        print(f"Median TTS_99:               {summary['median_tts_99_sec']*1e3:.2f} ms")
        print(f"Results Catalog:             {args.output}")


if __name__ == "__main__":
    main()
