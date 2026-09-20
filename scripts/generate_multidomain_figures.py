"""Generate publication figures for the Expanded Multi-Domain Research Paper.
Data source: expanded_multidomain_results.json
"""

import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

DATA_FILE = "expanded_multidomain_results.json"
FIG_DIR = "Docs/figures"
ARTIFACT_DIR = r"C:\Users\asifm\.gemini\antigravity\brain\c9ee5ec3-b7ae-4154-9727-b20a5cfb709c"

os.makedirs(FIG_DIR, exist_ok=True)
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# -----------------------------------------------------------------------------
# FIGURE 1: 9-Stream Pareto Frontier (Latency vs. Grounding Recall)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6))

stream_meta = {
    "S1_1.7B_Ctrl": {"label": "Qwen 1.7B (Control)", "col": "#aec7e8", "m": "o"},
    "S2_1.7B_Regex": {"label": "Qwen 1.7B (+ Regex)", "col": "#3182bd", "m": "s"},
    "S3_1.7B_Jev": {"label": "Qwen 1.7B (+ TypeSafe Jev)", "col": "#08519c", "m": "D"},
    "S4_7B_Ctrl": {"label": "Qwen 7B (Control)", "col": "#c7e9c0", "m": "o"},
    "S5_7B_Regex": {"label": "Qwen 7B (+ Regex)", "col": "#74c476", "m": "s"},
    "S6_7B_Jev": {"label": "Qwen 7B (+ TypeSafe Jev)", "col": "#006d2c", "m": "D"},
    "S7_Gemini_Ctrl": {"label": "Gemini Flash (Control)", "col": "#fcae91", "m": "o"},
    "S8_Gemini_Regex": {"label": "Gemini Flash (+ Regex)", "col": "#fb6a4a", "m": "s"},
    "S9_Gemini_Jev": {"label": "Gemini Flash (+ TypeSafe Jev)", "col": "#cb181d", "m": "D"},
}

pts = []
for sid, meta in stream_meta.items():
    lats = [v["streams"][sid]["median_ms"] for v in raw_data.values()]
    grds = [v["streams"][sid]["grounding_pct"] for v in raw_data.values()]
    m_lat = np.mean(lats)
    m_grd = np.mean(grds)
    pts.append((m_lat, m_grd, meta["label"], meta["col"], meta["m"]))
    ax.scatter(m_lat, m_grd, color=meta["col"], s=140, marker=meta["m"],
               edgecolors="black", linewidths=1.2, label=meta["label"], zorder=5)

for m_lat, m_grd, lbl, col, _ in pts:
    offset_y = 1.5 if "Jev" in lbl else -2.5
    offset_x = 15 if "Gemini" not in lbl else -65
    ax.annotate(f"{lbl}\n({m_lat:.0f}ms, {m_grd:.1f}%)", (m_lat, m_grd),
                xytext=(m_lat + offset_x, m_grd + offset_y),
                fontsize=8, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=col, linewidth=0.8))

ax.set_title("Figure 1: Multi-Domain Latency vs. Grounding Recall Pareto Frontier (324 Cells)")
ax.set_xlabel("Hardware Median Compute Latency (ms) [Lower is Better]")
ax.set_ylabel("Grounding Recall (%) [Higher is Better]")
ax.set_xlim(400, 2600)
ax.set_ylim(45, 95)
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()
fig1_p = os.path.join(FIG_DIR, "fig1_multidomain_pareto.png")
plt.savefig(fig1_p)
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 2: Token Efficiency Comparison (Control vs. Regex vs. Jev)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

model_groups = ["Qwen 1.7B (Local)", "Qwen 7B (Local 4-bit)", "Gemini Flash (Cloud)"]
ctrl_eff = [
    np.mean([v["streams"]["S1_1.7B_Ctrl"]["token_efficiency"] for v in raw_data.values()]),
    np.mean([v["streams"]["S4_7B_Ctrl"]["token_efficiency"] for v in raw_data.values()]),
    np.mean([v["streams"]["S7_Gemini_Ctrl"]["token_efficiency"] for v in raw_data.values()]),
]
regex_eff = [
    np.mean([v["streams"]["S2_1.7B_Regex"]["token_efficiency"] for v in raw_data.values()]),
    np.mean([v["streams"]["S5_7B_Regex"]["token_efficiency"] for v in raw_data.values()]),
    np.mean([v["streams"]["S8_Gemini_Regex"]["token_efficiency"] for v in raw_data.values()]),
]
jev_eff = [
    np.mean([v["streams"]["S3_1.7B_Jev"]["token_efficiency"] for v in raw_data.values()]),
    np.mean([v["streams"]["S6_7B_Jev"]["token_efficiency"] for v in raw_data.values()]),
    np.mean([v["streams"]["S9_Gemini_Jev"]["token_efficiency"] for v in raw_data.values()]),
]

x = np.arange(len(model_groups))
w = 0.25

b1 = ax.bar(x - w, ctrl_eff, w, label="Control (Raw RAG)", color="#bdbdbd", edgecolor="black")
b2 = ax.bar(x, regex_eff, w, label="+ Regex Trie", color="#74a9cf", edgecolor="black")
b3 = ax.bar(x + w, jev_eff, w, label="+ TypeSafe AI Jev", color="#02818a", edgecolor="black")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_title("Figure 2: Token Efficiency (Grounding Recall % per Output Token) Across Regimes")
ax.set_ylabel("Efficiency Ratio (η = % Recall / Token)")
ax.set_xticks(x)
ax.set_xticklabels(model_groups)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper left", frameon=True, facecolor="white")
plt.tight_layout()
fig2_p = os.path.join(FIG_DIR, "fig2_multidomain_token_efficiency.png")
plt.savefig(fig2_p)
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 3: Latency Speedup by Domain (Control vs. Jev)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

domain_names = ["Fiber Broadband", "Billing Disputes", "5G Network Ops", "Contract Upgrades"]
dom_keys = ["Fiber", "Billing", "Network", "Contract"]

ctrl_lats_17b = [np.mean([v["streams"]["S1_1.7B_Ctrl"]["median_ms"] for v in raw_data.values() if v["domain"] == dk]) for dk in dom_keys]
jev_lats_17b = [np.mean([v["streams"]["S3_1.7B_Jev"]["median_ms"] for v in raw_data.values() if v["domain"] == dk]) for dk in dom_keys]

x = np.arange(len(domain_names))
w = 0.32

b1 = ax.bar(x - w/2, ctrl_lats_17b, w, label="Qwen 1.7B Control", color="#9ecae1", edgecolor="black")
b2 = ax.bar(x + w/2, jev_lats_17b, w, label="Qwen 1.7B + TypeSafe Jev", color="#08519c", edgecolor="black")

for i, dk in enumerate(dom_keys):
    diff = ctrl_lats_17b[i] - jev_lats_17b[i]
    pct = (diff / ctrl_lats_17b[i]) * 100
    ax.annotate(f"-{diff:.0f}ms\n(-{pct:.1f}%)", (x[i] + w/2, jev_lats_17b[i] + 40),
                ha="center", fontsize=8.5, fontweight="bold", color="#08519c")

ax.set_title("Figure 3: Qwen 1.7B Latency Acceleration via TypeSafe Jev Across 4 Operational Domains")
ax.set_ylabel("Median Latency (ms) [Lower is Better]")
ax.set_xticks(x)
ax.set_xticklabels(domain_names)
ax.set_ylim(0, 1300)
ax.legend(loc="upper right", frameon=True, facecolor="white")
plt.tight_layout()
fig3_p = os.path.join(FIG_DIR, "fig3_multidomain_speedup.png")
plt.savefig(fig3_p)
plt.close()

# Copy to artifacts directory
for fn in ["fig1_multidomain_pareto.png", "fig2_multidomain_token_efficiency.png", "fig3_multidomain_speedup.png"]:
    src = os.path.join(FIG_DIR, fn)
    dst = os.path.join(ARTIFACT_DIR, fn)
    shutil.copy2(src, dst)

print("Generated and copied all expanded multi-domain publication figures!")
