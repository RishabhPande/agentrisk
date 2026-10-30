"""Figure: share of sessions declined per class, Legacy vs AgentGate.
Reads results/main.csv (semi-synthetic benchmark, mean of 5 seeds). Run from repo root:
    python docs/plot_policies.py
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = pd.read_csv("results/main.csv")
m = d.groupby(["policy", "cls"])[["decline", "stepup", "hard_decline"]].mean()
legit = [("H", "Human"), ("AU", "Agent,\nunverified"), ("AV", "Agent,\nverified")]
fraud = [("FH", "Manual\nfraud"), ("FB", "Bot\nfraud"), ("FAU", "Fraud via\nunverified agent"),
         ("FAV", "Laundering via\nagent platform"), ("FHJ", "Hijacked\nagent"), ("FSP", "Spoofed\nsignature")]
BLUE, RED, GREEN, GRID = "#2e86ab", "#e4572e", "#59a14f", "#e6e6e6"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 6]})
for ax, group, title in zip(axes, [legit, fraud],
        ["Legitimate sessions (lower is better)", "Fraud sessions (higher is better)"]):
    x = np.arange(len(group)); w = 0.38
    for i, (pol, col) in enumerate([("Legacy", RED), ("AgentGate", BLUE)]):
        hard = np.array([m.loc[(pol, c), "hard_decline"] for c, _ in group])
        tot = np.array([m.loc[(pol, c), "decline"] for c, _ in group])
        step = tot - hard
        xs = x + (i - 0.5) * w
        ax.bar(xs, hard, w, color=col, label=f"{pol}: hard decline")
        ax.bar(xs, step, w, bottom=hard, color=col, alpha=0.4, hatch="//", edgecolor="white",
               label=f"{pol}: step-up")
        for xi, t in zip(xs, tot):
            ax.text(xi, t + 0.015, f"{t:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in group], fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=10, loc="left")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Share of sessions declined or stepped up")
axes[1].legend(fontsize=7, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.suptitle("AgentRisk benchmark: Legacy vs AgentGate policy, by session class", x=0.01, ha="left",
             fontweight="bold")
fig.text(0.01, 0.01, "Semi-synthetic benchmark output (results/main.csv, mean of 5 seeds). "
         "Absolute rates describe the benchmark, not any production system.", fontsize=7, color="#666")
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig("docs/policies.png", dpi=150)
