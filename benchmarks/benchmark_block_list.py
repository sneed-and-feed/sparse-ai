"""
Block-list kernel benchmark: does cost scale with the number of ACTIVE blocks?
==============================================================================

Compares, at matched routing / sparsity:

  sdpa       F.scaled_dot_product_attention (FlashAttention-2 backend on Ampere+),
             dense; the baseline that matters.
  scan       benchmarks/benchmark_triton.py kernel (README numbers): visits every
             key block and tests its route. Non-causal only.
  lib        ultrametric.kernel.ultrametric_attention_triton (README quickstart
             ``block_sparse_attention``). Its wrapper rebuilds the block lists and
             syncs (.item()) on every call, so its time INCLUDES list construction;
             compare it to ``list+build``.
  list       ultrametric.block_list.block_list_attention, kernel only (lists
             prebuilt; in a model they are built once per forward and shared by
             every layer that uses the same routing).
  list+build list kernel + build_block_lists.
  build      build_block_lists alone.

Routing patterns (per-block routing vectors, p-ary, real_depth = ceil(log_p NB)):
  tree    block b's natural path (MSB first). At full depth this is block-diagonal:
          the best case (contiguous keys, identical counts per row).
  random  independent random branch per level, per (batch, head, block). Rows get
          different counts and non-contiguous keys: closer to a learned router.

depth 0 = every block active (dense through the same kernel) -> "own dense".
density = active block pairs / possible pairs (NB^2, or NB(NB+1)/2 when causal);
ideal   = 1 / density = best possible speedup over the kernel's own dense path.

Correctness: every timed sparse kernel is checked against an fp32 block-masked
reference on batch 0 and the first --check_heads heads (query-chunked, so it fits
at 16K+).

Do NOT run this while another job is using the same GPU: timings become
meaningless. The script warns if it sees memory in use by other processes.

Usage (Colab / any CUDA box, from the repo root):
  python benchmarks/benchmark_block_list.py
  python benchmarks/benchmark_block_list.py --seq_lens 2048 8192 --patterns random --causal on
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import triton  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import benchmark_triton as bt  # noqa: E402  (scan kernel + helpers)
from ultrametric.block_list import (  # noqa: E402
    block_list_attention,
    block_mask_reference,
    build_block_lists,
)

try:
    from ultrametric.kernel import ultrametric_attention_triton as lib_attention  # noqa: E402
    LIB_IMPORT_ERROR = None
except Exception as ex:  # pragma: no cover - depends on environment
    lib_attention = None
    LIB_IMPORT_ERROR = f"{type(ex).__name__}: {ex}"


# ----------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------

def make_router(pattern: str, Z: int, H: int, NB: int, real_depth: int, td: int, p: int,
                device: str, seed: int) -> torch.Tensor:
    """(Z, H, NB, td) int32 routing vectors; levels >= real_depth are 0."""
    r = torch.zeros(Z, H, NB, td, dtype=torch.int32, device=device)
    if pattern == "tree":
        b = torch.arange(NB, device=device)
        for d in range(real_depth):
            r[..., d] = ((b // (p ** (real_depth - 1 - d))) % p).to(torch.int32)
    elif pattern == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        r[..., :real_depth] = torch.randint(0, p, (Z, H, NB, real_depth), generator=g,
                                            dtype=torch.int32).to(device)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    return r


# ----------------------------------------------------------------------------
# Reference (query-chunked fp32, batch 0, first few heads)
# ----------------------------------------------------------------------------

@torch.no_grad()
def chunked_reference(q, k, v, block_mask, route_block, is_causal, chunk=2048):
    """q, k, v: (h, N, D); block_mask: (h, NB, NB) bool. Returns fp32 (h, N, D)."""
    h, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    kf, vf = k.float(), v.float()
    col_blk = torch.arange(N, device=q.device) // route_block
    out = torch.empty(h, N, D, dtype=torch.float32, device=q.device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        rows = torch.arange(s, e, device=q.device)
        sc = torch.matmul(q[:, s:e].float(), kf.transpose(-2, -1)) * scale  # (h, c, N)
        m = block_mask[:, rows // route_block][:, :, col_blk]               # (h, c, N)
        if is_causal:
            m = m & (rows[:, None] >= torch.arange(N, device=q.device)[None, :])
        sc = sc.masked_fill(~m, float("-inf"))
        out[:, s:e] = torch.matmul(torch.softmax(sc, dim=-1), vf)
        del sc, m
    return out


def max_err(out, ref, heads):
    return (out[0, :heads].float() - ref).abs().max().item()


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

def timeit(fn, warmup, rep):
    """Median / p20 / p80 ms via triton.testing.do_bench (flushes L2 between reps)."""
    with torch.no_grad():
        med, lo, hi = do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return float(med), float(lo), float(hi)


def try_time(name, fn, warmup, rep, errors):
    try:
        return timeit(fn, warmup, rep)
    except Exception as ex:
        msg = f"{type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}"
        errors[name] = msg[:300]
        torch.cuda.synchronize()
        return None


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_branch():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def warn_if_gpu_busy():
    free, total = torch.cuda.mem_get_info()
    ours = torch.cuda.memory_reserved()
    other_gb = (total - free - ours) / 1024 ** 3
    if other_gb > 1.0:
        print(f"WARNING: ~{other_gb:.1f} GB of GPU memory is used by other processes. "
              f"If another job is running, these timings are not valid.")
    return round(other_gb, 2)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def fmt(x, w=7, nd=2):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--route_block", type=int, default=128)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--patterns", nargs="+", default=["tree", "random"], choices=["tree", "random"])
    ap.add_argument("--causal", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--depths", type=str, default="auto",
                    help="'auto' = {0, 1, mid, full}, or comma list like 0,2,4")
    ap.add_argument("--warmup", type=int, default=25, help="do_bench warmup (ms)")
    ap.add_argument("--rep", type=int, default=100, help="do_bench measurement time (ms)")
    ap.add_argument("--check_heads", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--skip_lib", action="store_true")
    ap.add_argument("--skip_scan", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    dtype = torch.float16
    Z, H, D, RB, p = args.batch, args.heads, args.head_dim, args.route_block, args.p
    ch = min(args.check_heads, H)

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} | torch {torch.__version__} | triton {triton.__version__} | "
          f"branch {git_branch()} @ {(git_sha() or '?')[:8]}")
    other_gb = warn_if_gpu_busy()
    if LIB_IMPORT_ERROR:
        print(f"lib kernel unavailable: {LIB_IMPORT_ERROR}")
    print(f"B={Z} H={H} D={D} route_block={RB} p={p} fp16 | do_bench median, L2 flushed\n")

    results = []
    hdr = (f"{'N':>6} {'pat':>6} {'c':>1} {'dep':>5} {'dens':>6} {'ideal':>6} | "
           f"{'sdpa':>7} {'scan':>7} {'lib':>7} {'list':>7} {'l+bld':>7} {'build':>6} | "
           f"{'vsSDPA':>6} {'vsOwn':>6} {'eff':>5} | {'err':>7}")
    print(hdr)
    print("-" * len(hdr))

    for N in args.seq_lens:
        NB = triton.cdiv(N, RB)
        real_depth = max(1, math.ceil(math.log(max(NB, 2), p) - 1e-9))
        td = bt.next_pow2(real_depth)  # scan kernel needs a power-of-two level count
        if args.depths == "auto":
            depths = sorted({0, 1, max(1, real_depth // 2), real_depth})
        else:
            depths = sorted({int(x) for x in args.depths.split(",") if int(x) <= real_depth})

        torch.manual_seed(args.seed)
        q = torch.randn(Z, H, N, D, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        for causal_flag in args.causal:
            causal = causal_flag == "on"
            sdpa_t = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                            args.warmup, args.rep)

            for pattern in args.patterns:
                router = make_router(pattern, Z, H, NB, real_depth, td, p, device, args.seed)
                own_dense = None
                for depth in depths:
                    errors, errs = {}, {}
                    bl = build_block_lists(router, depth, is_causal=causal)
                    possible = NB * (NB + 1) // 2 if causal else NB * NB
                    density = bl.counts.sum().item() / (Z * H * possible)
                    ideal = 1.0 / density

                    list_t = try_time("list", lambda: block_list_attention(
                        q, k, v, bl, route_block=RB, is_causal=causal), args.warmup, args.rep, errors)
                    build_t = try_time("build", lambda: build_block_lists(
                        router, depth, is_causal=causal, arity=p), args.warmup, args.rep, errors)
                    build_sync_t = try_time("build_sync", lambda: build_block_lists(
                        router, depth, is_causal=causal), args.warmup, args.rep, errors)
                    lb_t = try_time("list+build", lambda: block_list_attention(
                        q, k, v, build_block_lists(router, depth, is_causal=causal, arity=p),
                        route_block=RB, is_causal=causal), args.warmup, args.rep, errors)

                    scan_ok = (not args.skip_scan and not causal and RB == 128 and N % 128 == 0)
                    scan_t = try_time("scan", lambda: bt.triton_attention(
                        q, k, v, router, depth, td, p), args.warmup, args.rep, errors) if scan_ok else None

                    lib_ok = (not args.skip_lib and lib_attention is not None and RB == 128)
                    lib_t = try_time("lib", lambda: lib_attention(
                        q, k, v, router, req_depth=depth, p=p, is_causal=causal),
                        args.warmup, args.rep, errors) if lib_ok else None

                    # Correctness vs fp32 block-masked reference (batch 0, first ch heads).
                    with torch.no_grad():
                        bm = block_mask_reference(router[:1, :ch], depth, is_causal=causal)[0]
                        ref = chunked_reference(q[0, :ch], k[0, :ch], v[0, :ch], bm, RB, causal)
                        sl = (slice(0, 1), slice(0, ch))
                        q1, k1, v1 = q[sl].contiguous(), k[sl].contiguous(), v[sl].contiguous()
                        r1 = router[sl].contiguous()
                        if list_t is not None:
                            bl1 = build_block_lists(r1, depth, is_causal=causal)
                            errs["list"] = max_err(block_list_attention(
                                q1, k1, v1, bl1, route_block=RB, is_causal=causal), ref, ch)
                        if scan_t is not None:
                            errs["scan"] = max_err(bt.triton_attention(q1, k1, v1, r1, depth, td, p), ref, ch)
                        if lib_t is not None:
                            errs["lib"] = max_err(lib_attention(
                                q1, k1, v1, r1, req_depth=depth, p=p, is_causal=causal)[0], ref, ch)
                        if depth == 0:
                            sd = F.scaled_dot_product_attention(q1, k1, v1, is_causal=causal)
                            errs["sdpa"] = max_err(sd, ref, ch)
                        del bm, ref, q1, k1, v1, r1

                    lm = list_t[0] if list_t else None
                    if depth == 0:
                        own_dense = lm
                    vs_sdpa = sdpa_t[0] / lm if lm else None
                    vs_own = own_dense / lm if (lm and own_dense) else None
                    eff = vs_own / ideal if vs_own else None
                    worst = max(errs.values()) if errs else float("nan")
                    passed = all(e < args.tol for e in errs.values())

                    row = {
                        "seq_len": N, "num_blocks": NB, "pattern": pattern, "causal": causal,
                        "depth": depth, "real_depth": real_depth,
                        "density": round(density, 5), "ideal_speedup_vs_own_dense": round(ideal, 2),
                        "max_count": bl.max_count,
                        "mean_count": round(bl.counts.float().mean().item(), 2),
                        "ms": {"sdpa": sdpa_t, "scan": scan_t, "lib_incl_build": lib_t,
                               "list": list_t, "list_plus_build": lb_t, "build": build_t, "build_sync": build_sync_t},
                        "speedup_list_vs_sdpa": round(vs_sdpa, 3) if vs_sdpa else None,
                        "speedup_list_vs_own_dense": round(vs_own, 3) if vs_own else None,
                        "efficiency_vs_ideal": round(eff, 3) if eff else None,
                        "max_abs_err": errs, "errors": errors, "pass": passed,
                    }
                    results.append(row)

                    med = lambda t: t[0] if t else None  # noqa: E731
                    print(f"{N:>6} {pattern:>6} {'y' if causal else 'n':>1} {depth:>2}/{real_depth:<2} "
                          f"{density:>6.3f} {ideal:>6.1f} | {fmt(med(sdpa_t))} {fmt(med(scan_t))} "
                          f"{fmt(med(lib_t))} {fmt(med(list_t))} {fmt(med(lb_t))} {fmt(med(build_t), 6)} | "
                          f"{fmt(vs_sdpa, 6)} {fmt(vs_own, 6)} {fmt(eff, 5)} | "
                          f"{worst:>7.1e} {'ok' if passed else 'FAIL'}"
                          + (f"  [{'; '.join(f'{k_}: {v_}' for k_, v_ in errors.items())}]" if errors else ""))
                    del bl
                del router
                torch.cuda.empty_cache()
        del q, k, v
        torch.cuda.empty_cache()

    n_fail = sum(not r["pass"] for r in results)
    print(f"\n{len(results)} configs, {n_fail} correctness failures (tol {args.tol}).")
    print("vsSDPA > 1 means the list kernel beats FlashAttention-2 (dense) at that sparsity.")
    print("eff = (speedup over own dense) / (ideal = 1/density); 1.0 = cost tracks active blocks exactly.")

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(), "git_branch": git_branch(),
        "gpu": gpu,
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
        "other_process_gpu_mem_gb_at_start": other_gb,
        "lib_import_error": LIB_IMPORT_ERROR,
        "config": {"batch": Z, "heads": H, "head_dim": D, "route_block": RB, "p": p,
                   "dtype": "float16", "timer": "triton.testing.do_bench",
                   "warmup_ms": args.warmup, "rep_ms": args.rep,
                   "stat": "[median, p20, p80] ms", "check_heads": ch, "tol": args.tol,
                   "seed": args.seed},
        "notes": "lib_incl_build includes the lib wrapper's own list construction and .item() sync; "
                 "compare it with list_plus_build. list is kernel-only with prebuilt lists. build/list_plus_build use the sync-free builder (arity=p); build_sync is the original builder with host syncs.",
    }
    out = args.output or os.path.join(
        REPO, "benchmarks", "results",
        f"benchmark_block_list_{gpu.split()[-1].replace('-', '_').lower()}_{time.strftime('%Y-%m-%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
