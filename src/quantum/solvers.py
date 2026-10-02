"""
Local Classical Solvers & Benchmark Execution Harness.

Provides zero-cost local baseline solvers:
- Exact branch-and-bound via SCIP (OR-Tools) with McCormick envelopes (MIPGap = 0.0)
- dimod.ExactSolver (for N <= 20)
- Simulated Annealing (dwave-neal)
- Tabu search (dwave-tabu)
- Time-to-Solution (TTS_99) statistical calculations with Clopper-Pearson bounds.
"""

from typing import Dict, Any, List, Optional
import time
import math
import numpy as np
import dimod
import neal
import tabu
from ortools.linear_solver import pywraplp


def solve_scip_exact(
    bqm: dimod.BinaryQuadraticModel,
    time_limit_sec: float = 60.0,
) -> Dict[str, Any]:
    """
    Solves an Ising or QUBO model to certified global optimality using SCIP via OR-Tools.
    Applies standard McCormick linearization envelopes:
        w_ij = x_i * x_j,  w_ij <= x_i, w_ij <= x_j, w_ij >= x_i + x_j - 1.

    Args:
        bqm: dimod.BinaryQuadraticModel (vartype SPIN or BINARY).
        time_limit_sec: Time limit in seconds for the branch-and-cut search.

    Returns:
        Dictionary with:
        - best_energy: Certified minimum energy.
        - best_sample: Optimal spin or binary configuration.
        - is_optimal: True if proven optimal within time limit (MIPGap = 0.0).
        - elapsed_sec: Wall-clock runtime in seconds.
    """
    t0 = time.perf_counter()

    # Convert to SPIN model for uniform formulation
    spin_bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    variables = sorted(list(spin_bqm.variables))
    var_to_idx = {v: i for i, v in enumerate(variables)}
    num_vars = len(variables)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver could not be initialized via OR-Tools.")

    solver.SetTimeLimit(int(time_limit_sec * 1000))

    # Binary variables x_i in {0, 1} where s_i = 2 * x_i - 1
    x = [solver.BoolVar(f"x_{i}") for i in range(num_vars)]
    obj = solver.Objective()

    # Linear and constant contributions
    offset = float(spin_bqm.offset)
    for v, h_val in spin_bqm.linear.items():
        i = var_to_idx[v]
        # h_val * s_i = h_val * (2 * x_i - 1) = 2 * h_val * x_i - h_val
        obj.SetCoefficient(x[i], obj.GetCoefficient(x[i]) + 2.0 * float(h_val))
        offset -= float(h_val)

    # Quadratic contributions via McCormick envelopes
    for (u, v), j_val in spin_bqm.quadratic.items():
        i = var_to_idx[u]
        j = var_to_idx[v]
        # j_val * s_i * s_j = j_val * (4 * w_ij - 2 * x_i - 2 * x_j + 1)
        wij = solver.NumVar(0.0, 1.0, f"w_{i}_{j}")
        solver.Add(wij <= x[i])
        solver.Add(wij <= x[j])
        solver.Add(wij >= x[i] + x[j] - 1.0)

        obj.SetCoefficient(wij, obj.GetCoefficient(wij) + 4.0 * float(j_val))
        obj.SetCoefficient(x[i], obj.GetCoefficient(x[i]) - 2.0 * float(j_val))
        obj.SetCoefficient(x[j], obj.GetCoefficient(x[j]) - 2.0 * float(j_val))
        offset += float(j_val)

    obj.SetOffset(offset)
    obj.SetMinimization()

    status = solver.Solve()
    t1 = time.perf_counter()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise TimeoutError(f"SCIP did not find a feasible integer solution within {time_limit_sec}s (status={status}).")

    is_optimal = (status == pywraplp.Solver.OPTIMAL)
    best_energy = float(solver.Objective().Value())

    # Map back to original vartype and variable labels
    best_sample = {}
    for v in variables:
        idx = var_to_idx[v]
        val_01 = int(round(x[idx].solution_value()))
        if bqm.vartype == dimod.SPIN:
            best_sample[v] = 2 * val_01 - 1
        else:
            best_sample[v] = val_01

    return {
        "method": "scip",
        "best_energy": best_energy,
        "best_sample": best_sample,
        "is_optimal": is_optimal,
        "elapsed_sec": float(t1 - t0),
    }


def solve_locally_classical(
    bqm: dimod.BinaryQuadraticModel,
    method: str = "sa",
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: Optional[int] = None,
    time_limit_sec: float = 60.0,
) -> Dict[str, Any]:
    """
    Executes a classical solver locally at zero cost.

    Args:
        bqm: The Binary Quadratic Model (SPIN or BINARY).
        method: "scip" (Exact branch-and-cut via OR-Tools SCIP),
                "sa" (Simulated Annealing via neal),
                "tabu" (Tabu search via dwave-tabu),
                "exact" (Exact solver for N <= 20 via dimod).
        num_reads: Number of independent heuristic samples (for SA / Tabu).
        num_sweeps: Number of Monte Carlo sweeps per read (for SA).
        seed: Random seed.
        time_limit_sec: Time limit for exact solvers.

    Returns:
        Dictionary with:
        - best_energy: Lowest energy found.
        - best_sample: Best spin/binary configuration.
        - elapsed_sec: Execution runtime in seconds.
    """
    if method == "scip":
        return solve_scip_exact(bqm, time_limit_sec=time_limit_sec)

    t0 = time.perf_counter()

    if method == "sa":
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
        )
    elif method == "tabu":
        sampler = tabu.TabuSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            seed=seed,
        )
    elif method == "exact":
        if len(bqm.variables) > 20:
            raise ValueError(f"dimod.ExactSolver restricted to N <= 20 variables, got {len(bqm.variables)}. Use method='scip' instead.")
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
    else:
        raise ValueError(f"Unknown classical solver method: {method}")

    t1 = time.perf_counter()
    best_record = sampleset.first

    return {
        "method": method,
        "best_energy": float(best_record.energy),
        "best_sample": best_record.sample,
        "sampleset": sampleset,
        "elapsed_sec": float(t1 - t0),
    }


def compute_time_to_solution(
    run_time_per_read: float,
    ground_state_energy: float,
    energies: np.ndarray,
    target_confidence: float = 0.99,
    energy_tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Computes Time-to-Solution (TTS_99):
        TTS(p) = t_read * ln(1 - p) / ln(1 - P_gs)

    Handles P_gs = 0 via Clopper-Pearson / Wilson upper bound to yield rigorous lower bound on TTS.
    """
    num_reads = len(energies)
    if num_reads == 0:
        return {
            "tts_99": 0.0,
            "p_gs": 0.0,
            "success_count": 0,
            "num_reads": 0,
            "is_lower_bound": True,
        }

    successes = int(np.sum(np.isclose(energies, ground_state_energy, atol=energy_tolerance)))
    p_gs = successes / num_reads

    if successes == 0:
        # Exact Clopper-Pearson 95% upper bound on success probability:
        # (1 - p_upper)^N = 1 - 0.95 = 0.05 => 1 - p_upper = 0.05^(1/N)
        # TTS lower bound = run_time * ln(1 - target_confidence) / ln(1 - p_upper)
        # ln(1 - p_upper) = (1/N) * ln(0.05)
        alpha = 0.05  # 95% confidence upper bound on P_gs
        log_one_minus_p = math.log(alpha) / num_reads
        tts = run_time_per_read * math.log(1.0 - target_confidence) / log_one_minus_p
        is_lower_bound = True
    elif p_gs >= 1.0:
        tts = run_time_per_read
        is_lower_bound = False
    else:
        tts = run_time_per_read * math.log(1.0 - target_confidence) / math.log(1.0 - p_gs)
        is_lower_bound = False

    return {
        "tts_99": float(tts),
        "p_gs": float(p_gs),
        "success_count": successes,
        "num_reads": num_reads,
        "is_lower_bound": is_lower_bound,
    }
