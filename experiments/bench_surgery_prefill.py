"""
End-to-end prefill benchmark: Llama-3.1-8B + learned router, block-list kernel vs FlashAttention
==============================================================================================

Same model/router recipe as experiments/eval_qasper_router.py (inject_surgery, WikiText-2
warmup, collapse init). For each sequence length N and condition, measures on one
WikiText-2 *test* sequence of N tokens (batch 1, causal prefill, no KV cache):

  dense_sdpa      backend=block_list, r=0: router skipped, attention = SDPA/FlashAttention-2.
                  This is the reference ("unmodified model").
  block_list@r    backend=block_list, r>0: router + block-granular routing (per-block majority
                  vote) + Triton block-list kernel. Includes router + list-building cost.
  eager@r         backend=eager, r>0: the token-level routed mask used for the QASPER / NIAH
                  results (materializes N x N in fp32). Quality reference; slow by design.
                  Only for N <= --eager_max_len.

Metrics per condition:
  total_ms   median wall time of model.model(input_ids) (all layers, no lm_head).
  attn_ms    median summed time inside the 32 attention modules (router, QKV/O projections,
             attention), via CUDA events on module hooks.
  ppl        perplexity of the same N tokens (lm_head applied in chunks after the timed runs).
  measured_budget  mean fraction of causal keys each query may attend to (averaged over layers).
  block_density    active / possible causal blocks (block_list only).

Do not run while another job uses the GPU.

Usage (Colab, from repo root):
  python experiments/bench_surgery_prefill.py
  python experiments/bench_surgery_prefill.py --seq_lens 4096,8192 --depths 2,3 --skip_warmup
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_surgery import inject_surgery
from sweep_topological_niah import warmup_router_wikitext


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_test_tokens(tok, n_max):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    if ids.numel() < n_max:
        reps = math.ceil(n_max / ids.numel())
        print(f"[Data] WikiText-2 test has {ids.numel()} tokens < {n_max}; repeating x{reps}")
        ids = ids.repeat(reps)
    return ids


class AttnTimer:
    """Sum of CUDA-event times spent inside the attention modules."""

    def __init__(self, modules):
        self.pairs, self.handles, self.enabled = [], [], False
        for m in modules:
            self.handles.append(m.register_forward_pre_hook(self._pre))
            self.handles.append(m.register_forward_hook(self._post))

    def _pre(self, mod, args):
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.pairs.append([e, None])

    def _post(self, mod, args, out):
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.pairs[-1][1] = e

    def start(self):
        self.pairs, self.enabled = [], True

    def stop(self):
        self.enabled = False
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.pairs)


def set_condition(model, backend, r, collect):
    cfg = model.config
    cfg.surgical_attention_backend = backend
    cfg.surgical_req_depth = r
    cfg.surgical_mask_override = None
    cfg.surgical_collect_stats = collect


def layer_stats(model):
    allowed, causal, dens = [], [], []
    for layer in model.model.layers:
        m = layer.self_attn
        if getattr(m, "last_mean_allowed_keys", None) is not None:
            allowed.append(m.last_mean_allowed_keys)
            causal.append(m.last_mean_causal_keys)
            m.last_mean_allowed_keys = None
        if getattr(m, "last_block_density", None) is not None:
            dens.append(m.last_block_density)
            m.last_block_density = None
    budget = (sum(allowed) / sum(causal)) if allowed else None
    return budget, (sum(dens) / len(dens) if dens else None)


@torch.no_grad()
def perplexity(model, ids, chunk=1024):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state  # (1, N, d)
    nll, count = 0.0, 0
    for s in range(0, ids.shape[1] - 1, chunk):
        e = min(ids.shape[1] - 1, s + chunk)
        logits = model.lm_head(hidden[:, s:e]).float()
        nll += F.cross_entropy(logits.view(-1, logits.shape[-1]), ids[0, s + 1:e + 1], reduction="sum").item()
        count += e - s
    return math.exp(nll / count)


@torch.no_grad()
def run_condition(model, timer, ids, backend, r, warmup, reps):
    # 1) stats + perplexity pass (not timed; collecting stats syncs the GPU)
    set_condition(model, backend, r, collect=True)
    ppl = perplexity(model, ids)
    budget, density = layer_stats(model)
    if r == 0:
        budget = 1.0

    # 2) timed passes
    set_condition(model, backend, r, collect=False)
    for _ in range(warmup):  # also triggers Triton autotuning for this shape
        model.model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    totals, attns = [], []
    for _ in range(reps):
        timer.start()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        model.model(input_ids=ids, use_cache=False)
        e.record()
        attns.append(timer.stop())
        totals.append(s.elapsed_time(e))
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {
        "total_ms": round(med(totals), 2), "total_ms_all": [round(x, 2) for x in totals],
        "attn_ms": round(med(attns), 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "ppl": round(ppl, 4),
        "measured_budget": round(budget, 4) if budget is not None else None,
        "block_density": round(density, 4) if density is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--seq_lens", default="4096,8192,16384")
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--eager_max_len", type=int, default=8192,
                    help="Run the token-level eager@r reference only up to this N (it is O(N^2) memory)")
    ap.add_argument("--train_steps", type=int, default=80)
    ap.add_argument("--skip_warmup", action="store_true")
    ap.add_argument("--init_mode", default="collapse", choices=["collapse", "random"])
    ap.add_argument("--route_block", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    gpu = torch.cuda.get_device_name(0)

    free, total = torch.cuda.mem_get_info()
    if (total - free) / 1024 ** 3 > 1.0:
        print(f"WARNING: {(total - free) / 1024 ** 3:.1f} GB already in use on the GPU; timings may be invalid.")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[1/4] Loading {args.model_id} on {gpu}")
    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map="cuda", token=hf_token)

    print("[2/4] Injecting router (same recipe as QASPER eval)")
    model = inject_surgery(model, tree_depth=5, arity=2, preserve_sinks=True, init_mode=args.init_mode)
    model.config.surgical_route_block = args.route_block
    if not args.skip_warmup:
        print(f"[3/4] WikiText-2 warmup ({args.train_steps} steps)")
        warmup_router_wikitext(model, tok, steps=args.train_steps, device="cuda")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    tokens = load_test_tokens(tok, max(seq_lens))
    timer = AttnTimer([l.self_attn for l in model.model.layers])

    print(f"[4/4] Prefill benchmark | N={seq_lens} | depths={depths} | route_block={args.route_block}")
    hdr = (f"{'N':>6} {'condition':>14} | {'total ms':>9} {'attn ms':>9} {'vs dense':>8} {'attn vs':>8} | "
           f"{'ppl':>8} {'budget':>7} {'blkdens':>7} {'mem GB':>7}")
    print(hdr)
    print("-" * len(hdr))
    results = []
    for N in seq_lens:
        ids = tokens[:N].unsqueeze(0).cuda()
        conds = [("dense_sdpa", "block_list", 0)]
        conds += [(f"block_list@{r}", "block_list", r) for r in depths]
        if N <= args.eager_max_len:
            conds += [(f"eager@{r}", "eager", r) for r in depths]
        ref = None
        for name, backend, r in conds:
            try:
                row = run_condition(model, timer, ids, backend, r, args.warmup, args.reps)
                err = None
            except torch.cuda.OutOfMemoryError as ex:
                row, err = {}, f"OOM: {str(ex).splitlines()[0][:200]}"
                torch.cuda.empty_cache()
            except Exception as ex:
                row, err = {}, f"{type(ex).__name__}: {str(ex).splitlines()[0][:300] if str(ex) else ''}"
                torch.cuda.empty_cache()
            if name == "dense_sdpa" and row:
                ref = row
            if row and ref:
                row["speedup_total_vs_dense"] = round(ref["total_ms"] / row["total_ms"], 3)
                row["speedup_attn_vs_dense"] = round(ref["attn_ms"] / row["attn_ms"], 3)
                row["ppl_ratio_vs_dense"] = round(row["ppl"] / ref["ppl"], 4)
            results.append({"seq_len": N, "condition": name, "backend": backend, "req_depth": r,
                            **row, "error": err})
            if err:
                print(f"{N:>6} {name:>14} | {err}")
                continue
            f = lambda x, w, nd=2: (f"{x:>{w}.{nd}f}" if x is not None else f"{'-':>{w}}")  # noqa: E731
            print(f"{N:>6} {name:>14} | {f(row['total_ms'], 9)} {f(row['attn_ms'], 9)} "
                  f"{f(row.get('speedup_total_vs_dense'), 8)} {f(row.get('speedup_attn_vs_dense'), 8)} | "
                  f"{f(row['ppl'], 8, 3)} {f(row['measured_budget'], 7, 3)} {f(row['block_density'], 7, 3)} "
                  f"{f(row['peak_mem_gb'], 7, 1)}")
        del ids
        torch.cuda.empty_cache()

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(), "gpu": gpu, "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "triton": __import__("triton").__version__,
        "dtype": str(dtype), "args": vars(args),
        "router_levels": int(model.model.layers[0].self_attn.router.levels),
        "local_window_tokens": int(getattr(model.config, "surgical_local_window", 16)),
        "notes": ("dense_sdpa skips the router (unmodified-model reference). block_list@r includes router, "
                  "block routing and list building. eager@r is the token-level mask used in the QASPER/NIAH "
                  "results; block_list@r uses a coarser block-granular mask, so its ppl/quality is not "
                  "identical by construction. Batch 1, no KV cache, no lm_head in timings."),
    }
    out = args.output or os.path.join("experiments", "results",
                                      f"bench_surgery_prefill_{gpu.split()[-1].replace('-', '_').lower()}_"
                                      f"{time.strftime('%Y-%m-%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"meta": meta, "results": results}, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
