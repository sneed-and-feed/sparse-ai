#!/usr/bin/env python3
"""
Sparse AI: QASPER (LongBench) evaluation of the learned router (Step 3F config)
===============================================================================
Answers the open question: does the WikiText-warmed Dynamic Topology Router (3F),
which retained NIAH retrieval at high sparsity, also preserve *natural-document QA*?

Conditions, all on the same surgically-patched model and the same samples:
  * dense      : r = 0 (unmodified attention; quality ceiling)
  * routed@r   : learned router, required matching depth r
  * window@r   : matched-budget baseline -- causal sliding window + sink with
                 W = mean number of keys the router actually allowed on THAT
                 sample (measured, not assumed)

The router's sparsity is *measured* per sample (fraction of causal keys each
query may attend to, averaged over layers/heads/queries), because the
theoretical 2^-r budget assumes perfectly balanced routing.

Metric: LongBench qa_f1 (SQuAD-style token F1, max over references), with
bootstrap 95% CIs and paired routed-minus-window differences.

Outputs (in --output_dir):
  qasper_router_predictions.jsonl   one line per (sample, condition)
  qasper_router_summary.json        aggregate metrics + full config/provenance

Note: the eval path materializes the full N x N score matrix in fp32 and masks
it, so this measures QUALITY ONLY, not speed. Peak memory per layer is roughly
heads * N^2 * 4 bytes (~4.3 GB at N = 6000 for Llama-3.1-8B).
"""

import os
import sys
import json
import math
import random
import string
import argparse
import subprocess
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_surgery import inject_surgery
from sweep_topological_niah import warmup_router_wikitext


# ----------------------------------------------------------------------------
# LongBench QASPER data + prompt (official LongBench template)
# ----------------------------------------------------------------------------

QASPER_PROMPT = (
    "You are given a scientific article and a question. Answer the question as concisely "
    "as you can, using a single phrase or sentence if possible. If the question cannot be "
    "answered based on the information in the article, write \"unanswerable\". If the "
    "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
    "provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the "
    "above article as concisely as you can, using a single phrase or sentence if possible. "
    "If the question cannot be answered based on the information in the article, write "
    "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
    "\"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
)


def load_qasper():
    """Load LongBench QASPER test split (200 samples), with a zip fallback for
    `datasets` versions that no longer run dataset scripts."""
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "qasper", split="test", trust_remote_code=True)
        return [dict(x) for x in ds]
    except Exception as e:
        print(f"[Data] load_dataset failed ({e}); falling back to data.zip from the Hub.")
        import zipfile
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("qasper.jsonl"))
            with z.open(name) as f:
                return [json.loads(line) for line in f]


def build_prompt_ids(tokenizer, sample, max_ctx):
    """LongBench convention: if too long, truncate the MIDDLE of the prompt."""
    prompt = QASPER_PROMPT.format(context=sample["context"], input=sample["input"])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    truncated = len(ids) > max_ctx
    if truncated:
        half = max_ctx // 2
        prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + \
                 tokenizer.decode(ids[-half:], skip_special_tokens=True)
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(ids, torch.Tensor):  # newer transformers return a BatchEncoding
            ids = ids["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids, truncated


# ----------------------------------------------------------------------------
# LongBench qa_f1
# ----------------------------------------------------------------------------

def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(w for w in s.split() if w not in {"a", "an", "the"})
    return " ".join(s.split())


def qa_f1(pred, ref):
    p, g = normalize_answer(pred).split(), normalize_answer(ref).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred, answers):
    # LongBench strips to first line for QA tasks
    pred = pred.lstrip("\n").split("\n")[0]
    return max(qa_f1(pred, a) for a in answers)


def is_degenerate(text, min_words=20, max_unique_ratio=0.2):
    w = text.split()
    return len(w) >= min_words and len(set(w)) / len(w) < max_unique_ratio


def bootstrap_ci(xs, n=2000, seed=0):
    if not xs:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------

def attn_modules(model):
    return [l.self_attn for l in model.model.layers]


def set_condition(model, req_depth, override=None, window=None, collect=False):
    cfg = model.config
    cfg.surgical_req_depth = req_depth
    cfg.surgical_mask_override = override
    cfg.surgical_window_size = window
    cfg.surgical_collect_stats = collect


def measured_budget(model):
    """Mean over layers of (allowed causal keys per query) and causal keys per query."""
    allowed, causal = [], []
    for m in attn_modules(model):
        if getattr(m, "last_mean_allowed_keys", None) is not None:
            allowed.append(m.last_mean_allowed_keys)
            causal.append(m.last_mean_causal_keys)
            m.last_mean_allowed_keys = None
    if not allowed:
        return None, None
    return sum(allowed) / len(allowed), sum(causal) / len(causal)


@torch.no_grad()
def generate(model, tokenizer, input_ids, max_new_tokens):
    input_ids = input_ids.to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="QASPER eval of the learned router vs matched-budget window")
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--depths", default="2,3,4", help="Comma-separated r > 0 to evaluate")
    ap.add_argument("--max_samples", type=int, default=200, help="QASPER test has 200")
    ap.add_argument("--max_ctx", type=int, default=6000, help="Prompt tokens before middle-truncation")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--train_steps", type=int, default=80, help="WikiText warmup steps (3F used 80)")
    ap.add_argument("--skip_warmup", action="store_true", help="Untrained router (3E-style control)")
    ap.add_argument("--init_mode", default="collapse", choices=["collapse", "random"])
    ap.add_argument("--tree_mode", action="store_true", help="Use Option B: true tree router")
    ap.add_argument("--no_window_baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="experiments/results")
    ap.add_argument("--attention_backend", default="eager", choices=["eager", "block_list"],
                    help="'block_list' = block-granular routing + Triton kernel for prefill (routed@r and dense); "
                         "window@r baselines always use the eager path")
    ap.add_argument("--oracle_mode", action="store_true", help="Evaluate HySparse2-style oracle at matched budget")
    ap.add_argument("--kolibri_rope", action="store_true", help="Strip RoPE from every 5th layer (kolibri pattern)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    assert all(d > 0 for d in depths), "r=0 (dense) is always run; pass only r > 0"

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "" if args.attention_backend == "eager" else f"_{args.attention_backend}"
    pred_path = os.path.join(args.output_dir, f"qasper_router_predictions{suffix}.jsonl")
    sum_path = os.path.join(args.output_dir, f"qasper_router_summary{suffix}.json")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[1/4] Loading {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype,
                                                 device_map="cuda", token=hf_token)

    print("[2/4] Injecting router (tree_depth=5, arity=2, sinks)")
    model = inject_surgery(model, tree_depth=5, arity=2, preserve_sinks=True, init_mode=args.init_mode, tree_mode=args.tree_mode)
    if not args.skip_warmup:
        print(f"[3/4] WikiText warmup ({args.train_steps} steps; same recipe as 3F)")
        warmup_router_wikitext(model, tok, steps=args.train_steps, device="cuda")
    model.eval()
    model.config.use_cache = True  # HF disables this during gradient checkpointing but doesn't restore it!
    model.config.surgical_attention_backend = args.attention_backend
    if args.kolibri_rope:
        model.config.surgical_kolibri_rope = True

    data = load_qasper()[: args.max_samples]
    print(f"[4/4] Evaluating {len(data)} QASPER samples | depths {depths} | "
          f"window baseline {'off' if args.no_window_baseline else 'on'}")

    conds = ["dense"] + [f"routed@{r}" for r in depths]
    if not args.no_window_baseline:
        conds += [f"window@{r}" for r in depths]
    if args.oracle_mode:
        conds += [f"oracle@{r}" for r in depths]
    per = {c: [] for c in conds}           # F1 per sample
    degen = {c: 0 for c in conds}
    frac = {f"routed@{r}": [] for r in depths}  # measured allowed / causal
    n_trunc = 0

    with open(pred_path, "w", encoding="utf-8") as fout:
        for i, s in enumerate(tqdm(data, desc="QASPER")):
            ids, trunc = build_prompt_ids(tok, s, args.max_ctx)
            n_trunc += int(trunc)

            def run(cond, **kw):
                set_condition(model, **kw)
                pred = generate(model, tok, ids, args.max_new_tokens)
                f1 = score(pred, s["answers"])
                per[cond].append(f1)
                degen[cond] += int(is_degenerate(pred))
                return pred, f1

            rows = []
            pred, f1 = run("dense", req_depth=0)
            rows.append({"cond": "dense", "pred": pred, "f1": f1})

            for r in depths:
                pred, f1 = run(f"routed@{r}", req_depth=r, collect=True)
                allowed, causal = measured_budget(model)
                fr = allowed / causal if allowed else None
                if fr is not None:
                    frac[f"routed@{r}"].append(fr)
                rows.append({"cond": f"routed@{r}", "pred": pred, "f1": f1,
                             "mean_allowed_keys": allowed, "mean_causal_keys": causal,
                             "allowed_fraction": fr})

                if not args.no_window_baseline and allowed:
                    # window override lives in the r>0 inference path; req_depth only
                    # needs to be >0 here, the routed mask is fully replaced.
                    W = max(1, int(round(allowed)))
                    pred, f1 = run(f"window@{r}", req_depth=r, override="window", window=W)
                    rows.append({"cond": f"window@{r}", "pred": pred, "f1": f1, "window": W})

                if args.oracle_mode and allowed:
                    W = max(1, int(round(allowed)))
                    pred, f1 = run(f"oracle@{r}", req_depth=r, override="oracle", window=W)
                    rows.append({"cond": f"oracle@{r}", "pred": pred, "f1": f1, "window": W})

            for row in rows:
                fout.write(json.dumps({"idx": i, "_id": s.get("_id"), "length": s.get("length"),
                                       "prompt_tokens": ids.shape[1], "truncated": trunc,
                                       "answers": s["answers"], **row}) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                msg = " | ".join(f"{c} {100*sum(v)/len(v):.1f}" for c, v in per.items() if v)
                tqdm.write(f"  [{i+1}] F1: {msg}")

    set_condition(model, req_depth=0)

    summary = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model_id": args.model_id,
            "args": vars(args),
            "n_samples": len(data),
            "n_truncated": n_trunc,
            "router": {"tree_depth": 5, "arity": 2, "preserve_sinks": True,
                       "warmup": None if args.skip_warmup else f"wikitext-2 x {args.train_steps} steps"},
            "metric": "LongBench qa_f1 (first line of prediction, max over references)",
        },
        "results": {},
    }
    for c, v in per.items():
        summary["results"][c] = {
            "f1": 100 * sum(v) / len(v) if v else None,
            "f1_ci95": [100 * x for x in bootstrap_ci(v, seed=args.seed)],
            "n": len(v),
            "degenerate_outputs": degen[c],
        }
    for r in depths:
        rc = f"routed@{r}"
        fr = frac[rc]
        summary["results"][rc]["measured_allowed_fraction"] = sum(fr) / len(fr) if fr else None
        summary["results"][rc]["theoretical_fraction"] = 2.0 ** (-r)
        wc = f"window@{r}"
        if wc in per and len(per[wc]) == len(per[rc]):
            diffs = [a - b for a, b in zip(per[rc], per[wc])]
            summary["results"][rc]["minus_window_f1"] = 100 * sum(diffs) / len(diffs)
            summary["results"][rc]["minus_window_ci95"] = [100 * x for x in bootstrap_ci(diffs, seed=args.seed)]

    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 92)
    print(f"QASPER (n={len(data)}, truncated={n_trunc})  F1 [95% CI]  | measured budget | vs window")
    print("=" * 92)
    for c, d in summary["results"].items():
        extra = ""
        if "measured_allowed_fraction" in d and d["measured_allowed_fraction"] is not None:
            extra += f" | budget {100*d['measured_allowed_fraction']:.1f}% (theory {100*d['theoretical_fraction']:.1f}%)"
        if "minus_window_f1" in d:
            lo, hi = d["minus_window_ci95"]
            extra += f" | Δ {d['minus_window_f1']:+.1f} [{lo:+.1f}, {hi:+.1f}]"
        print(f"{c:<12} {d['f1']:5.1f} [{d['f1_ci95'][0]:.1f}, {d['f1_ci95'][1]:.1f}]"
              f"  degen {d['degenerate_outputs']}{extra}")
    print(f"\n[Saved] {pred_path}\n[Saved] {sum_path}")


if __name__ == "__main__":
    main()
