"""Generate publication-quality figures for the scientific research paper.
Data source: scientific_matrix_results.json
Output directory: Docs/figures/ and artifact directory.
"""

from __future__ import annotations

import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration for academic paper look
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

DATA_FILE = "scientific_matrix_results.json"
FIG_DIR = "Docs/figures"
ARTIFACT_DIR = r"C:\Users\asifm\.gemini\antigravity\brain\c9ee5ec3-b7ae-4154-9727-b20a5cfb709c"

os.makedirs(FIG_DIR, exist_ok=True)
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Models and metadata
MODELS = {
    "S1_Jev_Qwen1.7B": {"name": "Jev + Qwen 1.7B", "color": "#1f77b4", "marker": "o", "type": "SLM (Jev)"},
    "S2_Ctrl_Qwen1.7B": {"name": "Control Qwen 1.7B", "color": "#aec7e8", "marker": "x", "type": "SLM (Ctrl)"},
    "S3_Jev_Qwen7B": {"name": "Jev + Qwen 2.5 7B", "color": "#2ca02c", "marker": "s", "type": "MLM (Jev)"},
    "S4_Ctrl_Qwen7B": {"name": "Control Qwen 2.5 7B", "color": "#98df8a", "marker": "+", "type": "MLM (Ctrl)"},
    "S5_Jev_Gemini": {"name": "Jev + Gemini 3.8 Flash", "color": "#d62728", "marker": "^", "type": "Frontier (Jev)"},
    "S6_Ctrl_Gemini": {"name": "Control Gemini 3.8 Flash", "color": "#ff9896", "marker": "v", "type": "Frontier (Ctrl)"},
}

# -----------------------------------------------------------------------------
# FIGURE 1: Pareto Frontier (Latency vs. Grounding Recall)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

avg_points = []
for mid, mcfg in MODELS.items():
    lats = [sc["streams"][mid]["median_ms"] for sc in raw_data.values() if mid in sc["streams"]]
    grounds = [sc["streams"][mid]["grounding_pct"] for sc in raw_data.values() if mid in sc["streams"]]
    mean_lat = np.mean(lats)
    mean_ground = np.mean(grounds)
    avg_points.append((mean_lat, mean_ground, mcfg["name"], mcfg["color"], mcfg["marker"]))

    # Plot individual scenario scatter points faintly
    ax.scatter(lats, grounds, color=mcfg["color"], alpha=0.25, s=30, marker=mcfg["marker"])
    # Plot mean aggregate point prominently
    ax.scatter(mean_lat, mean_ground, color=mcfg["color"], s=130, marker=mcfg["marker"],
               edgecolors="black", linewidths=1.2, label=mcfg["name"], zorder=5)

# Annotate key sweet spots
for plat, pgrd, name, col, _ in avg_points:
    offset_y = 1.2 if "Jev" in name else -2.2
    offset_x = 20 if "Gemini" not in name else -60
    ax.annotate(f"{name}\n({plat:.0f}ms, {pgrd:.1f}%)", (plat, pgrd),
                xytext=(plat + offset_x, pgrd + offset_y),
                fontsize=8.5, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=col, linewidth=0.7))

# Draw Pareto frontier line
pareto_x = [avg_points[0][0], avg_points[2][0], avg_points[5][0], avg_points[4][0]]
pareto_y = [avg_points[0][1], avg_points[2][1], avg_points[5][1], avg_points[4][1]]
# Sort by x
pts_sorted = sorted(zip(pareto_x, pareto_y), key=lambda x: x[0])
ax.plot([p[0] for p in pts_sorted], [p[1] for p in pts_sorted], "k--", alpha=0.5, label="Empirical Pareto Frontier")

ax.set_title("Figure 1: Latency vs. Grounding Recall Pareto Frontier Across 54 Benchmark Cells")
ax.set_xlabel("Median Hardware Compute Latency (ms) [Lower is Better]")
ax.set_ylabel("Grounding Recall (%) [Higher is Better]")
ax.set_xlim(0, 3600)
ax.set_ylim(0, 60)
ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig1_pareto_frontier.png")
plt.savefig(fig1_path)
plt.close()
print(f"Generated {fig1_path}")

# -----------------------------------------------------------------------------
# FIGURE 2: Autoregressive Law Validation (Tokens vs. Latency in 7B Model)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))

tokens_7b = []
lats_7b = []
labels_7b = []

for sc_name, sc in raw_data.items():
    # S3 Jev
    tok_jev = sc["streams"]["S3_Jev_Qwen7B"]["out_tokens"]
    lat_jev = sc["streams"]["S3_Jev_Qwen7B"]["median_ms"]
    tokens_7b.append(tok_jev)
    lats_7b.append(lat_jev)
    labels_7b.append("Jev-Assisted")

    # S4 Ctrl
    tok_ctrl = sc["streams"]["S4_Ctrl_Qwen7B"]["out_tokens"]
    lat_ctrl = sc["streams"]["S4_Ctrl_Qwen7B"]["median_ms"]
    tokens_7b.append(tok_ctrl)
    lats_7b.append(lat_ctrl)
    labels_7b.append("Control")

tok_arr = np.array(tokens_7b)
lat_arr = np.array(lats_7b)

# Linear regression
slope, intercept = np.polyfit(tok_arr, lat_arr, 1)
r_squared = np.corrcoef(tok_arr, lat_arr)[0, 1] ** 2

# Plot points
jev_mask = [l == "Jev-Assisted" for l in labels_7b]
ctrl_mask = [l == "Control" for l in labels_7b]

ax.scatter(tok_arr[jev_mask], lat_arr[jev_mask], color="#2ca02c", s=90, marker="s",
           edgecolors="black", linewidths=1.0, label="Jev + Qwen 7B (Budget Capped)", zorder=4)
ax.scatter(tok_arr[ctrl_mask], lat_arr[ctrl_mask], color="#d62728", s=90, marker="^",
           edgecolors="black", linewidths=1.0, label="Control Qwen 7B (Uncapped)", zorder=4)

x_line = np.linspace(70, 190, 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, "k-", linewidth=1.5,
        label=f"Fit: Latency = {slope:.1f} × Tokens + {intercept:.0f} (R² = {r_squared:.2f})")

ax.annotate(f"Decode Rate: {1000/slope:.1f} tokens/sec\n(Empirical Hardware Limit)",
            xy=(130, slope * 130 + intercept),
            xytext=(100, 3100),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffcc", edgecolor="gray"))

ax.set_title("Figure 2: Empirical Verification of the Autoregressive Law on Qwen 7B 4-bit (RTX 4070)")
ax.set_xlabel("Generated Output Tokens (Tokens)")
ax.set_ylabel("Measured Median Latency (ms)")
ax.set_xlim(60, 200)
ax.set_ylim(1200, 3800)
ax.legend(loc="upper left", frameon=True, facecolor="white")
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "fig2_autoregressive_law.png")
plt.savefig(fig2_path)
plt.close()
print(f"Generated {fig2_path}")

# -----------------------------------------------------------------------------
# FIGURE 3: Attention Dilution Matrix (Query Length vs. Grounding Recall)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

query_levels = ["Short", "Mid", "Long"]
x_pos = np.arange(len(query_levels))
width = 0.22

models_to_plot = [
    ("S1_Jev_Qwen1.7B", "Qwen 1.7B (Local SLM)", "#1f77b4"),
    ("S3_Jev_Qwen7B", "Qwen 2.5 7B (Local 4-bit)", "#2ca02c"),
    ("S5_Jev_Gemini", "Gemini 3.8 Flash (Frontier Cloud)", "#d62728"),
]

for i, (m_id, m_label, m_col) in enumerate(models_to_plot):
    means_by_q = []
    for q in query_levels:
        q_grounds = [
            sc["streams"][m_id]["grounding_pct"]
            for sc in raw_data.values()
            if sc["query_level"] == q and m_id in sc["streams"]
        ]
        means_by_q.append(np.mean(q_grounds))

    bars = ax.bar(x_pos + (i - 1) * width, means_by_q, width, label=m_label, color=m_col, edgecolor="black", linewidth=0.8)
    # Add data labels
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Figure 3: Attention Dilution Effect: Grounding Recall Collapse on Long Queries (~80 Words)")
ax.set_xlabel("User Query Length / Narrative Complexity")
ax.set_ylabel("Grounding Entity Recall (%)")
ax.set_xticks(x_pos)
ax.set_xticklabels(["Short (~8 words)\n'Fiber cut, LOS red'",
                    "Mid (~26 words)\n'Drop cable severed...'",
                    "Long (~80 words)\n'Excavation crew narrative...'"])
ax.set_ylim(0, 65)
ax.legend(loc="upper right", frameon=True, facecolor="white")
plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "fig3_attention_dilution_heatmap.png")
plt.savefig(fig3_path)
plt.close()
print(f"Generated {fig3_path}")

# -----------------------------------------------------------------------------
# FIGURE 4: Speedup Breakdown: Jev vs. Control Across Architectures
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

arch_names = ["Qwen 1.7B (Local)", "Qwen 2.5 7B (Local 4-bit)", "Gemini 3.8 Flash (Cloud)"]
ctrl_lats = [
    np.mean([sc["streams"]["S2_Ctrl_Qwen1.7B"]["median_ms"] for sc in raw_data.values()]),
    np.mean([sc["streams"]["S4_Ctrl_Qwen7B"]["median_ms"] for sc in raw_data.values()]),
    np.mean([sc["streams"]["S6_Ctrl_Gemini"]["median_ms"] for sc in raw_data.values()]),
]
jev_lats = [
    np.mean([sc["streams"]["S1_Jev_Qwen1.7B"]["median_ms"] for sc in raw_data.values()]),
    np.mean([sc["streams"]["S3_Jev_Qwen7B"]["median_ms"] for sc in raw_data.values()]),
    np.mean([sc["streams"]["S5_Jev_Gemini"]["median_ms"] for sc in raw_data.values()]),
]

x_idx = np.arange(len(arch_names))
b_width = 0.32

rects1 = ax.bar(x_idx - b_width/2, ctrl_lats, b_width, label="Control Baseline (Unconstrained)", color="#999999", edgecolor="black")
rects2 = ax.bar(x_idx + b_width/2, jev_lats, b_width, label="Jev-Assisted (Adaptive Budgeting)", color="#1b9e77", edgecolor="black")

# Add percentage annotations
diff_7b = ctrl_lats[1] - jev_lats[1]
pct_7b = (diff_7b / ctrl_lats[1]) * 100
ax.annotate(f"-{diff_7b:.0f} ms\n(-{pct_7b:.1f}%)",
            xy=(x_idx[1] + b_width/2, jev_lats[1]),
            xytext=(x_idx[1], 2400),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=10, fontweight="bold", color="#1b9e77",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e6f5d0", edgecolor="#1b9e77"))

# Add data labels
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.0f} ms", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.0f} ms", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Figure 4: Latency Comparison: Jev-Assisted vs. Control Baseline Across Model Scales")
ax.set_ylabel("Average Median Latency (ms) [Lower is Better]")
ax.set_xticks(x_idx)
ax.set_xticklabels(arch_names)
ax.set_ylim(0, 3500)
ax.legend(loc="upper left", frameon=True, facecolor="white")
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "fig4_jev_speedup_breakdown.png")
plt.savefig(fig4_path)
plt.close()
print(f"Generated {fig4_path}")

# Copy to artifacts directory so they can be viewed in artifact viewer
for fn in ["fig1_pareto_frontier.png", "fig2_autoregressive_law.png", "fig3_attention_dilution_heatmap.png", "fig4_jev_speedup_breakdown.png"]:
    src = os.path.join(FIG_DIR, fn)
    dst = os.path.join(ARTIFACT_DIR, fn)
    shutil.copy2(src, dst)
    print(f"Copied to artifact: {dst}")

print("All figures successfully created and copied to artifacts!")
