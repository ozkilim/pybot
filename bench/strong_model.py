"""Figure: PyBot hybrid against a strong comparator (Llama-3.1-8B), full 1,000-prompt benchmark."""
import matplotlib.pyplot as plt
S = [0, 5, 10, 20, 30, 40, 50, 60, 75, 100]
OLD = [31.5, 35.3, 38.8, 45.6, 52.6, 58.7, 65.0, 70.6, 76.3, 80.2]     # Python side with the 360M-written cache
NEW = [38.7, 42.0, 45.7, 52.1, 58.2, 63.5, 68.3, 73.6, 77.7, 80.2]     # + cache written by the 8B
ORA = [38.7, 43.7, 48.7, 58.6, 68.0, 75.7, 80.8, 83.2, 83.7, 80.2]     # perfect routing over the chooser
M, M360 = 80.2, 38.8
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax.plot(S, [NEW[0] + s / 100 * (M - NEW[0]) for s in S], ls=":", c="#999", label="random routing")
ax.plot(S, ORA, ls="--", c="#bbb", label="perfect routing (hindsight)")
ax.plot(S, OLD, c="#8aa", lw=2, marker="o", ms=4, label="hybrid, Python side as before")
ax.plot(S, NEW, c="#d1495b", lw=2.6, marker="o", ms=5, label="hybrid + cache written by the 8B")
ax.axhline(M, c="#222", lw=1); ax.text(1, M + 1, "Llama-3.1-8B alone 80.2%", fontsize=9)
ax.axhline(M360, c="#666", lw=1, ls="-."); ax.text(62, M360 - 3.2, "SmolLM2-360M alone 38.8%", fontsize=9, color="#555")
ax.annotate("Python only, zero model calls:\nties the 360M", (0, 38.7), (8, 26), fontsize=9, arrowprops=dict(arrowstyle="->", color="#d1495b"))
ax.set_xlabel("share of turns sent to the 8B (%)"); ax.set_ylabel("win rate vs reference (%)")
ax.set_title("Against a strong model, Python can't get close cheaply", fontsize=12, weight="bold")
ax.set_ylim(20, 90); ax.grid(alpha=.25); ax.legend(loc="lower right", fontsize=8.5, frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("bench/results/strong_model.png")
