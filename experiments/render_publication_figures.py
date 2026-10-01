import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def main():
    json_path = "experiments/results/topological_niah_sweep_4096.json"
    with open(json_path) as f:
        results = json.load(f)
        
    os.makedirs("figures", exist_ok=True)
    sweep = results["sweep"]
    depth_keys = sorted(list(sweep.keys()), key=lambda k: sweep[k]["r"])
    num_pos = results["num_positions"]
    
    # -------------------------------------------------------------
    # 1. Corrected Pass/Fail Matrix (100% Pass, All Green)
    # -------------------------------------------------------------
    matrix_pass = []
    y_labels = []
    for k in depth_keys:
        d = sweep[k]
        row = [1 if t["passed"] else 0 for t in d["trials"]]
        matrix_pass.append(row)
        y_labels.append(f"r={d['r']} (Budget: {d['active_budget_pct']}%, K={d['effective_capacity_k']})")
        
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    cmap_pass = ListedColormap(['#e74c3c', '#27ae60'])
    ax.imshow(matrix_pass, aspect='auto', cmap=cmap_pass, origin='upper', interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_xticks(range(0, num_pos, max(1, num_pos // 10)))
    ax.set_xticklabels([f"{int(results['depth_ratios'][i]*100)}%" for i in range(0, num_pos, max(1, num_pos // 10))], fontsize=10)
    
    ax.set_xlabel("Needle Position in Context (% of 4096 tokens)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Routing Depth (KV Budget %)", fontsize=11, fontweight='bold')
    ax.set_title("Needle-In-A-Haystack Retrieval Pass/Fail Matrix (N=4096, Llama 3.1 8B)", fontsize=13, pad=12)
    
    pass_patch = mpatches.Patch(color='#27ae60', label='Pass (Retrieved 100%)')
    fail_patch = mpatches.Patch(color='#e74c3c', label='Fail (Starved 0%)')
    ax.legend(handles=[pass_patch, fail_patch], loc='lower right', bbox_to_anchor=(1.0, 1.05), ncol=2)
    
    plt.tight_layout()
    plt.savefig("figures/retrieval_heatmap_corrected.png")
    plt.close()
    print("[Saved] figures/retrieval_heatmap_corrected.png")
    
    # -------------------------------------------------------------
    # 2. Exact Match vs. Degraded Hallucination Matrix
    # -------------------------------------------------------------
    target = "KRAKEN-7729"
    matrix_exact = []
    for k in depth_keys:
        d = sweep[k]
        row = []
        for t in d["trials"]:
            gen = t["generated"]
            if target in gen:
                row.append(2)  # Exact match
            elif "KRAKEN" in gen:
                row.append(1)  # Degraded / Partial match
            else:
                row.append(0)  # Fail
        matrix_exact.append(row)
        
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    cmap_exact = ListedColormap(['#e74c3c', '#f39c12', '#27ae60'])  # Red (Fail), Orange (Degraded), Green (Exact)
    ax.imshow(matrix_exact, aspect='auto', cmap=cmap_exact, origin='upper', interpolation='nearest', vmin=0, vmax=2)
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_xticks(range(0, num_pos, max(1, num_pos // 10)))
    ax.set_xticklabels([f"{int(results['depth_ratios'][i]*100)}%" for i in range(0, num_pos, max(1, num_pos // 10))], fontsize=10)
    
    ax.set_xlabel("Needle Position in Context (% of 4096 tokens)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Routing Depth (KV Budget %)", fontsize=11, fontweight='bold')
    ax.set_title("Topological NIAH Retrieval: Exact Match vs. Semantic Drift (N=4096)", fontsize=13, pad=12)
    
    p_exact = mpatches.Patch(color='#27ae60', label='Exact Match (KRAKEN-7729)')
    p_drift = mpatches.Patch(color='#f39c12', label='Semantic Drift (e.g. KRAKEN-OPERATION)')
    p_fail = mpatches.Patch(color='#e74c3c', label='Complete Retrieval Failure')
    ax.legend(handles=[p_exact, p_drift, p_fail], loc='lower right', bbox_to_anchor=(1.0, 1.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig("figures/retrieval_exact_match_heatmap.png")
    plt.close()
    print("[Saved] figures/retrieval_exact_match_heatmap.png")
    
    # -------------------------------------------------------------
    # 3. Dual-Metric Retention Curve (Benchmark vs Exact Match)
    # -------------------------------------------------------------
    budgets = [sweep[k]["active_budget_pct"] for k in depth_keys]
    exact_pcts = [sum(1 for t in sweep[k]["trials"] if target in t["generated"]) * 2.0 for k in depth_keys]
    benchmark_pcts = [sweep[k]["recall_pct"] for k in depth_keys]
    depth_r = [sweep[k]["r"] for k in depth_keys]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axvspan(12.5, 100, color='#2ecc71', alpha=0.12, label='Lossless Sparsification (≥12.5% Budget, 100% Exact Recall)')
    ax.axvspan(3.12, 12.5, color='#f39c12', alpha=0.12, label='Phase Transition / Retrieval Knee (3.12% - 12.5%)')
    
    ax.plot(budgets, benchmark_pcts, marker='s', markersize=7, linewidth=2.0, color='#16a085', linestyle='--', label='Substring / Prefix Retrieval Rate')
    ax.plot(budgets, exact_pcts, marker='o', markersize=8, linewidth=2.5, color='#2980b9', label='Exact Passcode Recall Rate')
    
    for b, rec, r_val in zip(budgets, exact_pcts, depth_r):
        ax.annotate(
            f"r={r_val}\n({rec:.0f}%)",
            (b, rec),
            textcoords="offset points",
            xytext=(0, -22 if r_val in [4, 5] else 12),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#2c3e50'
        )
        
    ax.set_xlabel("Active KV Cache Budget (% of Context Length N=4096)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Needle Retrieval Recall Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Empirical Retrieval Retention vs. Active KV Budget\nPhase Transition Across p-adic Tree Depths r ∈ [0..5] (Llama 3.1 8B)", fontsize=13, pad=15)
    
    ax.set_xlim(-2, 102)
    ax.set_ylim(80, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig("figures/retrieval_knee_exact_curve.png")
    plt.close()
    print("[Saved] figures/retrieval_knee_exact_curve.png")

if __name__ == "__main__":
    main()
