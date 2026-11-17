"""The headline figure: win rate against the share of turns sent to the model (full 1,000-prompt benchmark).

    python bench/moneyshot.py      # writes bench/results/moneyshot.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BENCH=all python bench/chooser.py bench  (chooser + deferral, model - best)
SHARE = [0, 5, 10, 20, 30, 40, 50, 60, 75, 100]
HYBRID = [30.5, 32.7, 34.4, 37.4, 39.8, 41.8, 43.0, 44.1, 44.4, 38.8]
MODEL, PYBOT = 38.8, 23.0

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

# where the hybrid beats the model on every turn
ax.fill_between(SHARE, MODEL, HYBRID, where=[h >= MODEL for h in HYBRID], interpolate=True, color=BLUE, alpha=0.10, lw=0)
ax.axhline(MODEL, color=INK2, lw=1.5, ls=(0, (5, 4)))
ax.plot(SHARE, HYBRID, color=BLUE, lw=2.5, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
ax.plot([0], [PYBOT], marker="o", ms=10, color=ORANGE, mec=SURF, mew=2, zorder=4, clip_on=False)

ax.text(100, MODEL - 1.3, "SmolLM2-360M on every turn  38.8%", ha="right", va="top", color=INK2, fontsize=11)
ax.text(2.5, PYBOT, "PyBot alone  23.0%", va="center", color=INK, fontsize=11)
ax.text(2.5, 30.5, "Python only, no model  30.5%", va="center", color=INK, fontsize=11)
ax.annotate("Ties the model\nwith 70% of turns in Python", xy=(30, 39.8), xytext=(14, 45.2),
            color=INK, fontsize=11, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.annotate("Beats it by 3.1 pts\nwith 60% in Python", xy=(40, 41.8), xytext=(45, 34.2),
            color=INK, fontsize=11, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.text(60, 45.3, "PyBot + router + SmolLM2-360M", color=BLUE, fontsize=11.5, fontweight="bold", ha="center")

ax.set_xlim(0, 100); ax.set_ylim(20, 48)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 75, 100])
ax.set_xticklabels([f"{x}%" for x in [0, 10, 20, 30, 40, 50, 60, 75, 100]])
ax.set_yticks(range(20, 50, 5)); ax.set_yticklabels([f"{y}%" for y in range(20, 50, 5)])
ax.set_xlabel("Share of turns sent to the small model", color=INK2, labelpad=8)
ax.set_ylabel("Win rate vs reference answers", color=INK2, labelpad=8)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(colors=INK2, length=0, pad=6)

fig.text(0.07, 0.955, "A rule-based chatbot that only calls a model when it should", fontsize=17, fontweight="bold", color=INK)
fig.text(0.07, 0.912, "SmolTalkBench, all 1,000 prompts. Router fitted on held-out prompts only.", fontsize=11.5, color=INK2)
fig.text(0.07, 0.02, "Judge: Qwen3.6-35B-A3B (local), AlpacaEval-style win rate with length penalty. github.com/ozkilim/pybot",
         fontsize=9, color=INK2)
fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.14)
out = os.path.join(ROOT, "bench", "results", "moneyshot.png")
fig.savefig(out, facecolor=SURF)
print(out)
