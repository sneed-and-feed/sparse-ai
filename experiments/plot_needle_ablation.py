import os
import json
import matplotlib.pyplot as plt

def main():
    os.makedirs("figures", exist_ok=True)
    
    with open("experiments/results/topological_niah_sweep_4096.json") as f:
        data_3a = json.load(f)["sweep"]
        
    with open("experiments/results/topological_niah_sweep_3b_random.json") as f:
        data_3b = json.load(f)["sweep"]
        
    with open("experiments/results/niah_sweep_banal_random.json") as f:
        data_3e = json.load(f)["sweep"]
        
    depth_keys = sorted(list(data_3a.keys()), key=lambda k: data_3a[k]["r"])
    
    budgets = [data_3a[k]["active_budget_pct"] for k in depth_keys]
    recalls_3a = [data_3a[k]["recall_pct"] for k in depth_keys]
    recalls_3b = [data_3b[k]["recall_pct"] for k in depth_keys]
    recalls_3e = [data_3e[k]["recall_pct"] for k in depth_keys]
    depth_r = [data_3a[k]["r"] for k in depth_keys]
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    # Highlight zones
    ax.axvspan(20, 100, color='#2ecc71', alpha=0.10, label='Safe Sparsification (>20% Budget)')
    ax.axvspan(10, 20, color='#f39c12', alpha=0.10, label='Phase Transition / Retrieval Knee (10-20%)')
    ax.axvspan(0, 10, color='#e74c3c', alpha=0.10, label='Information Starvation Collapse (<10%)')
    
    # Plot Step 3A (Trained, Outlier)
    ax.plot(budgets, recalls_3a, marker='o', markersize=8, linewidth=2.5, color='#27ae60', 
            label='Step 3A: Trained Dynamic Router (Outlier Needle)')
    
    # Plot Step 3B (Untrained Random, Outlier)
    ax.plot(budgets, recalls_3b, marker='s', markersize=8, linewidth=2.2, color='#2980b9', linestyle='--', 
            label='Step 3B: Untrained Random Router (Outlier Needle: KRAKEN-7729)')
    
    # Plot Step 3E (Untrained Random, Banal)
    ax.plot(budgets, recalls_3e, marker='^', markersize=8, linewidth=2.2, color='#e67e22', linestyle='-.', 
            label='Step 3E: Untrained Random Router (Banal Needle: spherical coordinates)')
    
    # Annotate differences
    for b, r3b, r3e, r_val in zip(budgets, recalls_3b, recalls_3e, depth_r):
        if r_val in [1, 2, 3]:
            delta = r3b - r3e
            if delta > 0:
                ax.annotate(f"Δ={delta:.0f}%\n(Outlier effect)", (b, (r3b + r3e)/2), 
                            textcoords="offset points", xytext=(12, -5), ha='left', 
                            fontsize=8, fontweight='bold', color='#8e44ad')
        ax.annotate(f"r={r_val}\n({r3e:.0f}%)", (b, r3e), textcoords="offset points", 
                    xytext=(0, -22 if r_val in [2,3] else 10), ha='center', 
                    fontsize=8, fontweight='bold', color='#d35400')
                    
    ax.set_xlabel("Active KV Cache Budget (% of Context Length N=4096)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Needle Retrieval Recall Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Peer Review Ablation: Outlier vs. Banal Needle under Random Hyperplanes\nIsolating Pretrained Representation Geometry from High-Surprisal Artifacts (Llama 3.1 8B)", 
                 fontsize=12.5, pad=15)
    
    ax.set_xlim(-2, 102)
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9.5)
    
    plt.tight_layout()
    out_path = "figures/comparison_outlier_vs_banal_needle.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()
