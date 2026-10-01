import os
import json
import matplotlib.pyplot as plt

def main():
    os.makedirs("figures", exist_ok=True)
    
    with open("experiments/results/topological_niah_sweep_4096.json") as f:
        data_3a = json.load(f)["sweep"]
        
    with open("experiments/results/topological_niah_sweep_3b_random.json") as f:
        data_3b = json.load(f)["sweep"]
        
    depth_keys = sorted(list(data_3a.keys()), key=lambda k: data_3a[k]["r"])
    
    budgets = [data_3a[k]["active_budget_pct"] for k in depth_keys]
    recalls_3a = [data_3a[k]["recall_pct"] for k in depth_keys]
    recalls_3b = [data_3b[k]["recall_pct"] for k in depth_keys]
    depth_r = [data_3a[k]["r"] for k in depth_keys]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Highlight zones
    ax.axvspan(20, 100, color='#2ecc71', alpha=0.12, label='Safe Sparsification (>20% Budget, >90% Recall)')
    ax.axvspan(10, 20, color='#f39c12', alpha=0.12, label='Phase Transition / Retrieval Knee (10-20%)')
    ax.axvspan(0, 10, color='#e74c3c', alpha=0.12, label='Information Starvation Collapse (<10%)')
    
    # Plot Step 3A (Trained)
    ax.plot(budgets, recalls_3a, marker='o', markersize=8, linewidth=2.5, color='#27ae60', label='Step 3A: Trained Dynamic Router (100% Retained)')
    
    # Plot Step 3B (Untrained Random)
    ax.plot(budgets, recalls_3b, marker='^', markersize=8, linewidth=2.5, color='#e74c3c', linestyle='--', label='Step 3B: Untrained Random Router (Zero Warmup)')
    
    # Annotate points
    for b, r3a, r3b, r_val in zip(budgets, recalls_3a, recalls_3b, depth_r):
        ax.annotate(f"r={r_val}\n({r3b:.0f}%)", (b, r3b), textcoords="offset points", xytext=(0, -22 if r_val in [2,3,4,5] else 12), ha='center', fontsize=8, fontweight='bold', color='#c0392b')
        if r_val in [3, 4, 5]:
            ax.annotate(f"+{r3a - r3b:.0f}%", (b, (r3a + r3b)/2), textcoords="offset points", xytext=(15, 0), ha='left', fontsize=8, fontweight='bold', color='#27ae60')
            
    ax.set_xlabel("Active KV Cache Budget (% of Context Length N=4096)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Needle Retrieval Recall Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Peer Review Ablation: Trained vs. Untrained Random Router\nEmpirical Phase Transition Across Tree Depths r ∈ [0..5] (Llama 3.1 8B)", fontsize=13, pad=15)
    
    ax.set_xlim(-2, 102)
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='center right', frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    out_path = "figures/comparison_3a_vs_3b_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()
