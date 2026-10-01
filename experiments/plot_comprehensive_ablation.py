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
        
    with open("experiments/results/niah_sweep_banal_wikitext.json") as f:
        data_3f = json.load(f)["sweep"]
        
    depth_keys = sorted(list(data_3a.keys()), key=lambda k: data_3a[k]["r"])
    
    budgets = [data_3a[k]["active_budget_pct"] for k in depth_keys]
    recalls_3a = [data_3a[k]["recall_pct"] for k in depth_keys]
    recalls_3b = [data_3b[k]["recall_pct"] for k in depth_keys]
    recalls_3e = [data_3e[k]["recall_pct"] for k in depth_keys]
    recalls_3f = [data_3f[k]["recall_pct"] for k in depth_keys]
    depth_r = [data_3a[k]["r"] for k in depth_keys]
    
    fig, ax = plt.subplots(figsize=(11.5, 7), dpi=300)
    
    # Highlight zones
    ax.axvspan(20, 100, color='#2ecc71', alpha=0.10, label='Safe Sparsification (>20% Budget, >95% Recall)')
    ax.axvspan(10, 20, color='#f39c12', alpha=0.10, label='Phase Transition / Retrieval Knee (10-20%)')
    ax.axvspan(0, 10, color='#e74c3c', alpha=0.10, label='Information Starvation Collapse (<10%)')
    
    # Plot Step 3A (Task-Trained, Outlier)
    ax.plot(budgets, recalls_3a, marker='o', markersize=8, linewidth=2.2, color='#27ae60', 
            label='Step 3A: Task-Trained Router (Outlier Needle)')
            
    # Plot Step 3F (Unsupervised WikiText-2, Banal Needle)
    ax.plot(budgets, recalls_3f, marker='D', markersize=8, linewidth=2.6, color='#8e44ad', 
            label='Step 3F: Unsupervised WikiText-2 Router (Banal Needle: spherical coordinates)')
            
    # Plot Step 3B (Untrained Random, Outlier)
    ax.plot(budgets, recalls_3b, marker='s', markersize=8, linewidth=2.0, color='#2980b9', linestyle='--', 
            label='Step 3B: Untrained Random Router (Outlier Needle: KRAKEN-7729)')
            
    # Plot Step 3E (Untrained Random, Banal)
    ax.plot(budgets, recalls_3e, marker='^', markersize=8, linewidth=2.0, color='#e67e22', linestyle='-.', 
            label='Step 3E: Untrained Random Router (Banal Needle: spherical coordinates)')
    
    # Annotate key recovery jumps from 3E to 3F
    for b, r3e, r3f, r_val in zip(budgets, recalls_3e, recalls_3f, depth_r):
        if r_val in [2, 3, 4]:
            delta = r3f - r3e
            ax.annotate(f"+{delta:.0f}%\n(WikiText)", (b, (r3e + r3f)/2), 
                        textcoords="offset points", xytext=(14, -6), ha='left', 
                        fontsize=8.5, fontweight='bold', color='#8e44ad')
        ax.annotate(f"r={r_val}\n({r3f:.0f}%)", (b, r3f), textcoords="offset points", 
                    xytext=(0, 10), ha='center', 
                    fontsize=8, fontweight='bold', color='#4a235a')
                    
    ax.set_xlabel("Active KV Cache Budget (% of Context Length N=4096)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Needle Retrieval Recall Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Definitive Peer Review Ablation: Unsupervised General-Domain Pretraining\nEliminates Outlier Surprisal Artifacts on In-Distribution Needles (Llama 3.1 8B)", 
                 fontsize=13, pad=15)
    
    ax.set_xlim(-2, 102)
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    out_path = "figures/comparison_comprehensive_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()
