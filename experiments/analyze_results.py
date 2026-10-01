import json

with open('experiments/results/topological_niah_sweep_4096.json') as f:
    data = json.load(f)

target = 'KRAKEN-7729'
for r_key, r_data in data['sweep'].items():
    r = r_data['r']
    budget = r_data['active_budget_pct']
    exact = sum(1 for t in r_data['trials'] if target in t['generated'])
    partial = sum(1 for t in r_data['trials'] if 'KRAKEN' in t['generated'] and target not in t['generated'])
    fail = sum(1 for t in r_data['trials'] if 'KRAKEN' not in t['generated'])
    print(f"r={r} (budget={budget:5.2f}% | K={r_data['effective_capacity_k']:4d}): exact={exact:2d}/50 ({exact*2:5.1f}%), partial={partial:2d}/50 ({partial*2:5.1f}%), total_pass={r_data['successes']:2d}/50")
    if partial > 0:
        degraded = [f"pos {t['pos_idx']} ({t['needle_ratio']:.2f}): '{t['generated']}'" for t in r_data['trials'] if 'KRAKEN' in t['generated'] and target not in t['generated']]
        for d in degraded:
            print(f"   -> {d}")
