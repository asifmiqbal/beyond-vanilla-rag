import json
import numpy as np
from collections import defaultdict

with open('expanded_multidomain_results.json') as f:
    raw = json.load(f)

stream_keys = [
    ("S1_1.7B_Ctrl", "Small Local (Qwen 1.7B)", "Control"),
    ("S2_1.7B_Regex", "Small Local (Qwen 1.7B)", "Regex Trie"),
    ("S3_1.7B_Jev", "Small Local (Qwen 1.7B)", "TypeSafe Jev"),
    ("S4_7B_Ctrl", "Medium Local (Qwen 7B 4-bit)", "Control"),
    ("S5_7B_Regex", "Medium Local (Qwen 7B 4-bit)", "Regex Trie"),
    ("S6_7B_Jev", "Medium Local (Qwen 7B 4-bit)", "TypeSafe Jev"),
    ("S7_Gemini_Ctrl", "Frontier Cloud (Gemini 3.8 Flash)", "Control"),
    ("S8_Gemini_Regex", "Frontier Cloud (Gemini 3.8 Flash)", "Regex Trie"),
    ("S9_Gemini_Jev", "Frontier Cloud (Gemini 3.8 Flash)", "TypeSafe Jev"),
]

stats = defaultdict(lambda: {'lat': [], 'recall': [], 'tok': [], 'eff': []})
domain_stats = defaultdict(lambda: defaultdict(lambda: {'lat': [], 'recall': [], 'tok': [], 'eff': []}))
complexity_stats = defaultdict(lambda: defaultdict(lambda: {'lat': [], 'recall': [], 'tok': [], 'eff': []}))
density_stats = defaultdict(lambda: defaultdict(lambda: {'lat': [], 'recall': [], 'tok': [], 'eff': []}))

for scen_id, sdata in raw.items():
    dom = sdata['domain']
    comp = sdata['complexity']
    dens = sdata['kb_density']
    
    for skey, model, regime in stream_keys:
        st = sdata['streams'][skey]
        lat = st['median_ms']
        recall = st['grounding_pct']
        tok = st['out_tokens']
        eff = st['token_efficiency']
        
        pair = (model, regime)
        stats[pair]['lat'].append(lat)
        stats[pair]['recall'].append(recall)
        stats[pair]['tok'].append(tok)
        stats[pair]['eff'].append(eff)
        
        domain_stats[dom][pair]['lat'].append(lat)
        domain_stats[dom][pair]['recall'].append(recall)
        domain_stats[dom][pair]['tok'].append(tok)
        domain_stats[dom][pair]['eff'].append(eff)
        
        complexity_stats[comp][pair]['lat'].append(lat)
        complexity_stats[comp][pair]['recall'].append(recall)
        complexity_stats[comp][pair]['tok'].append(tok)
        complexity_stats[comp][pair]['eff'].append(eff)

        density_stats[dens][pair]['lat'].append(lat)
        density_stats[dens][pair]['recall'].append(recall)
        density_stats[dens][pair]['tok'].append(tok)
        density_stats[dens][pair]['eff'].append(eff)

md = []
md.append("### Table: Aggregate Empirical Metrics Across 324 Experimental Conditions\n")
md.append("| Model Scale | Classification Regime | Median Latency ($p50$) | Grounding Recall (%) | Output Tokens | Token Efficiency ($\eta$) |")
md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
for _, model, regime in stream_keys:
    v = stats[(model, regime)]
    md.append(f"| **{model}** | {regime} | {np.mean(v['lat']):.1f} ms | {np.mean(v['recall']):.1f}% | {np.mean(v['tok']):.1f} tok | **{np.mean(v['eff']):.3f}** |")

md.append("\n### Table: Domain Breakdown Performance (4 Enterprise Telco Domains)\n")
md.append("| Domain | Model Scale | Control Latency | Regex Latency | Jev Latency | Jev Latency Delta | Control Recall | Jev Recall | Jev Efficiency |")
md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for dom in ['Fiber', 'Billing', 'Network', 'Contract']:
    for model in ["Small Local (Qwen 1.7B)", "Medium Local (Qwen 7B 4-bit)", "Frontier Cloud (Gemini 3.8 Flash)"]:
        c_lat = np.mean(domain_stats[dom][(model, "Control")]['lat'])
        r_lat = np.mean(domain_stats[dom][(model, "Regex Trie")]['lat'])
        j_lat = np.mean(domain_stats[dom][(model, "TypeSafe Jev")]['lat'])
        c_rec = np.mean(domain_stats[dom][(model, "Control")]['recall'])
        j_rec = np.mean(domain_stats[dom][(model, "TypeSafe Jev")]['recall'])
        j_eff = np.mean(domain_stats[dom][(model, "TypeSafe Jev")]['eff'])
        delta_lat = j_lat - c_lat
        pct_speedup = (c_lat - j_lat) / c_lat * 100.0
        sign = "-" if delta_lat < 0 else "+"
        md.append(f"| {dom} | {model.split()[0]} {model.split()[1]} | {c_lat:.1f} ms | {r_lat:.1f} ms | **{j_lat:.1f} ms** | {sign}{abs(delta_lat):.1f} ms ({pct_speedup:+.1f}%) | {c_rec:.1f}% | {j_rec:.1f}% | **{j_eff:.3f}** |")

with open('Docs/summary_tables.md', 'w') as f:
    f.write('\n'.join(md))

print("Summary tables written to Docs/summary_tables.md successfully!")
