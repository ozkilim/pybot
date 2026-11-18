"""Headline figures for both experiments: how close to the parent LLM, as Python takes more of the traffic.
Reads the 1%-step curves dumped by chooser.py (DUMP=...)."""
import json
import matplotlib.pyplot as plt
import numpy as np

BLUE, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def draw(curve, out, title, sub, model_name, keys, llm_label):
    d = json.load(open(curve))
    fine, m = np.array(d["fine"]), d["model"]
    py = 100 - np.arange(101)                 # share of chats Python answers
    rel = 100 * fine / m                      # quality as % of the parent LLM
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.axhline(100, color=INK2, lw=1.2, ls=(0, (4, 3)))
    lx, ly, lha = llm_label
    ax.text(lx, ly, "%s alone (%.1f%% win rate) = 100%%" % (model_name, m), color=INK2, fontsize=9.5, ha=lha, va="center")
    ax.fill_between(py, 100, np.maximum(rel, 100), color=BLUE, alpha=.12, lw=0)
    ax.plot(py, rel, color=BLUE, lw=2.6, solid_capstyle="round")
    for x, txt, dx, dy in keys:
        y = rel[100 - x]
        ax.scatter([x], [y], s=70, color=BLUE, edgecolor=SURF, linewidth=2, zorder=5)
        ax.annotate(txt % {"q": y, "w": fine[100 - x]}, (x, y), (x + dx, y + dy), fontsize=9.5, color=INK,
                    ha="right" if dx < 0 else "left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8, shrinkA=2, shrinkB=5))
    ax.set_xlim(0, 102); ax.set_ylim(35, 122)
    ax.set_xticks(range(0, 101, 10)); ax.set_xticklabels(["%d%%" % v for v in range(0, 101, 10)])
    ax.yaxis.set_major_formatter(lambda v, _: "%d%%" % v)
    ax.set_xlabel("share of chats answered by Python (no model call)", color=INK2, fontsize=10.5)
    ax.set_ylabel("quality, as % of the parent LLM", color=INK2, fontsize=10.5)
    ax.grid(axis="y", color=GRID, lw=.8); ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    fig.text(.065, .955, title, fontsize=14.5, weight="bold", color=INK)
    fig.text(.065, .912, sub, fontsize=10, color=INK2)
    fig.text(.065, .02, "SmolTalkBench, all 1,000 prompts; win rate vs reference answers, judged by Qwen3.6-35B. Router and scorer fit on held-out prompts only.",
             fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=.09, right=.97, top=.86, bottom=.14)
    fig.savefig(out, facecolor=SURF)


R = "bench/results/"
draw(R + "curve_360m.json", R + "moneyshot_exp1_360m.png",
     "Python answers 79% of chats, and the hybrid still matches the LLM",
     "Experiment 1 · parent LLM: SmolLM2-360M · Python side: rules, retrieval, a 33k answer cache written offline",
     "SmolLM2-360M", [(100, "all Python, zero model calls: %(q).0f%%", -4, -10), (90, "PyBot-90: %(q).0f%%", -4, -11),
                      (79, "79%% Python:\nmatches the LLM", 3, 10), (50, "half Python: %(q).0f%%, beats the LLM", 3, 8)],
     (40, 96.5, "left"))
draw(R + "curve_8b.json", R + "moneyshot_exp2_8b.png",
     "Python answers half the chats at 85% of a much stronger LLM",
     "Experiment 2 · parent LLM: Llama-3.1-8B · Python side: same rules and retrieval, cache written by the 8B offline",
     "Llama-3.1-8B", [(30, "30%% Python: %(q).0f%%", 3, 7), (50, "half Python: %(q).0f%%", 3, 7), (75, "75%% Python: %(q).0f%%", 3, 7),
                      (90, "PyBot-90: %(q).0f%%", -4, -9), (100, "all Python, zero model calls: %(q).0f%%", -4, -8)],
     (100, 103, "right"))
