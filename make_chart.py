"""
Render the results bar chart used in the README (results.png).

Speedup = CPU time / Apple-GPU time, so >1 means the GPU is faster. Log scale,
with a break-even line at 1.0. The story is the left-to-right shape: the
memory-bound elementwise steps clear the line easily; the compute-bound steps
(PCA, KNN) -- the ones that need GPU linear algebra -- do not.

Run:  python make_chart.py   ->  results.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Measured on Apple M5 Pro (see RESULTS.md / bench.log)
steps = ["normalize\n+ log1p", "scale", "PCA\n(50 comps)", "exact KNN\n(k=15)"]
kind = ["memory-bound", "memory-bound", "compute-bound", "compute-bound"]
compute_only = [10.7, 5.7, 1.4, 0.08]   # CPU / MPS, data resident on GPU
with_transfer = [5.5, 4.6, 1.3, 0.26]    # CPU / MPS, incl. host->device copy

x = np.arange(len(steps))
w = 0.38

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

WIN, LOSE = "#2a9d54", "#c0392b"


def colors(vals):
    return [WIN if v >= 1 else LOSE for v in vals]


ax.set_yscale("log")
ax.set_ylim(0.05, 22)

# shaded regions behind the two camps (drawn first, sit underneath the bars)
ax.axvspan(-0.5, 1.5, color="#2a9d54", alpha=0.05, zorder=0)
ax.axvspan(1.5, 3.5, color="#c0392b", alpha=0.05, zorder=0)
ax.text(0.5, 17, "memory-bound\n(GPU wins, but cheap)", ha="center", va="top",
        fontsize=9, color="#2a9d54")
ax.text(2.5, 17, "compute-bound\n(needs GPU linalg — unavailable)", ha="center", va="top",
        fontsize=9, color="#c0392b")

b1 = ax.bar(x - w / 2, compute_only, w, color=colors(compute_only),
            label="GPU compute-only", edgecolor="white", linewidth=0.6, zorder=3)
b2 = ax.bar(x + w / 2, with_transfer, w, color=colors(with_transfer), alpha=0.55,
            label="GPU incl. host->device transfer", edgecolor="white", linewidth=0.6, zorder=3)

ax.axhline(1.0, color="#333", lw=1.3, ls="--", zorder=2)
ax.text(-0.45, 1.07, "break-even (CPU)", ha="left", va="bottom",
        fontsize=9, color="#333", style="italic")

ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 5, 10])
ax.set_yticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "5x", "10x"])
ax.set_ylabel("Speedup vs CPU  (higher = GPU faster)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=10)

# all value labels sit just above their bar top, colored by win/loss
for bars, vals in ((b1, compute_only), (b2, with_transfer)):
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:g}x", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=WIN if v >= 1 else LOSE)

ax.set_title("Single-cell preprocessing: Apple GPU (PyTorch-MPS) vs CPU\nApple M5 Pro · 50k cells × 20k genes",
             fontsize=12.5, fontweight="bold", pad=14)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#eee", lw=0.8)

fig.tight_layout()
fig.savefig("results.png", facecolor="white", bbox_inches="tight")
print("wrote results.png")
