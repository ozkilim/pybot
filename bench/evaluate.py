"""Score the cascade on SmolTalkBench from already-judged replies, and draw the trade-off.

The cascade's reply to a prompt is either PyBot's reply or the small model's reply, and both
were judged once (bench/judge.py). So every operating point is a selection over the same two
sets of judged replies: no extra judge calls, and the comparison is paired prompt by prompt.

    python bench/evaluate.py            # writes bench/results/results.json, RESULTS.md, tradeoff.png
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hybrid.gate import Gate  # noqa: E402

OUT = os.path.join(ROOT, "bench", "out")
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "bench", "results"))
BASELINES = [("smollm2-360m", "SmolLM2-360M"), ("smollm2-1.7b", "SmolLM2-1.7B"), ("gemma3-1b", "Gemma3-1B")]
PUBLISHED = {"PyBot": 26.7, "SmolLM-360M": 26.6, "SmolLM-1.7B": 40.6, "SmolLM-135M": 10.1}


JUDGED = os.environ.get("JUDGED_DIR", OUT)   # e.g. bench/poc for the small proof-of-concept subset


def load(name):
    path = os.path.join(JUDGED if name.endswith(".judged.jsonl") else OUT, name)
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def ci(values, n=2000, seed=0):
    """Mean and 95% bootstrap interval, in percent."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return 100 * values.mean(), 100 * np.quantile(means, 0.025), 100 * np.quantile(means, 0.975)


def main():
    os.makedirs(RES, exist_ok=True)
    weights = json.load(open(os.path.join(ROOT, "hybrid", "gate_weights.json")))
    replies, judged = load("pybot_bench.jsonl"), load("pybot_bench.judged.jsonl")
    ids = [i for i in replies if i in judged]
    code = np.array([judged[i]["score"] for i in ids])
    results = {"n": len(ids), "judge": "Qwen3.6-35B-A3B Q4 (local), AlpacaEval 2.0 weighted prompt, WildBench length penalty",
               "published_deepseek_judge": PUBLISHED, "systems": {}, "cascades": {}}
    m, lo, hi = ci(code)
    results["systems"]["PyBot (code only)"] = {"win_rate": m, "ci": [lo, hi], "llm_share": 0.0,
                                               "median_ms": 1000 * float(np.median([replies[i]["seconds"] for i in ids]))}
    for run, label in BASELINES:
        for suffix, tag in (("", ""), ("-brief", " (brief prompt)")):
            lj = load(run + suffix + "_bench.judged.jsonl")
            if lj and all(i in lj for i in ids):
                m, lo, hi = ci([lj[i]["score"] for i in ids])
                lr = load(run + suffix + "_bench.jsonl")
                results["systems"][label + tag] = {"win_rate": m, "ci": [lo, hi], "llm_share": 1.0,
                                                   "median_ms": 1000 * float(np.median([lr[i]["seconds"] for i in ids]))}

    for name, spec in weights["llm"].items():
        lj = load(spec["run"] + "_bench.judged.jsonl")
        if not lj or not all(i in lj for i in ids):
            continue
        llm = np.array([lj[i]["score"] for i in ids])
        gate = Gate(llm=name)
        margins = np.array([(lambda s: s[2] - s[1])(gate.score(replies[i]["instruction"], replies[i])) for i in ids])
        rng = np.random.default_rng(0)
        points = []
        for budget in gate.budgets():
            tau = float(spec["taus"][budget])
            defer = margins > tau
            mixed = np.where(defer, llm, code)
            m, lo, hi = ci(mixed)
            gain = mixed - code
            boot = gain[rng.integers(0, len(gain), (2000, len(gain)))].mean(1)
            # what an oracle and a coin would do with the same number of model calls
            k = int(defer.sum())
            oracle = np.where(np.isin(np.arange(len(ids)), np.argsort(code - llm)[:k]), llm, code)
            points.append({"budget": float(budget), "llm_share": float(defer.mean()), "win_rate": m, "ci": [lo, hi],
                           "gain_vs_pybot": 100 * float(gain.mean()),
                           "gain_ci": [100 * float(np.quantile(boot, 0.025)), 100 * float(np.quantile(boot, 0.975))],
                           "oracle_same_calls": 100 * float(oracle.mean()),
                           "random_same_calls": 100 * float(code.mean() + defer.mean() * (llm.mean() - code.mean())),
                           "deferred_code_win": 100 * float(code[defer].mean()) if k else None,
                           "deferred_llm_win": 100 * float(llm[defer].mean()) if k else None,
                           "kept_code_win": 100 * float(code[~defer].mean()) if k < len(ids) else None})
        results["cascades"][name] = {"run": spec["run"], "llm_alone": 100 * float(llm.mean()),
                                     "best_of_both_oracle": 100 * float(np.maximum(code, llm).mean()), "points": points}

    json.dump(results, open(os.path.join(RES, "results.json"), "w"), indent=1)
    lines = ["# SmolTalkBench results", "",
             "%d prompts. Judge: %s. Every row was scored by this same judge." % (results["n"], results["judge"]), "",
             "| System | Win rate | 95% CI | Turns sent to a model |", "|---|---:|---|---:|"]
    for label, s in results["systems"].items():
        lines.append("| %s | %.1f%% | %.1f–%.1f | %.0f%% |" % (label, s["win_rate"], *s["ci"], 100 * s["llm_share"]))
    for name, c in results["cascades"].items():
        lines += ["", "## PyBot → %s" % name, "",
                  "Model alone: %.1f%%. Picking the better of the two per prompt with hindsight: %.1f%%." % (c["llm_alone"], c["best_of_both_oracle"]), "",
                  "| Target model share | Actual | Win rate | 95% CI | Gain over PyBot | Random routing, same calls | Oracle, same calls |",
                  "|---:|---:|---:|---|---:|---:|---:|"]
        for p in c["points"]:
            lines.append("| %.0f%% | %.1f%% | %.1f%% | %.1f–%.1f | %+.1f (%+.1f to %+.1f) | %.1f%% | %.1f%% |"
                         % (100 * p["budget"], 100 * p["llm_share"], p["win_rate"], *p["ci"], p["gain_vs_pybot"],
                            *p["gain_ci"], p["random_same_calls"], p["oracle_same_calls"]))
    open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    colours = ["#1f6feb", "#d1242f", "#1a7f37"]
    base = results["systems"]["PyBot (code only)"]["win_rate"]
    for colour, (name, c) in zip(colours, results["cascades"].items()):
        xs = [0] + [100 * p["llm_share"] for p in c["points"]]
        ys = [base] + [p["win_rate"] for p in c["points"]]
        ax.plot(xs, ys, "-o", color=colour, ms=4, lw=2, label="PyBot → %s (gated)" % name)
        ax.plot([0, 100], [base, c["llm_alone"]], ":", color=colour, lw=1.2, alpha=0.8)
        ax.plot([0] + [100 * p["llm_share"] for p in c["points"]], [base] + [p["oracle_same_calls"] for p in c["points"]],
                "--", color=colour, lw=1, alpha=0.45)
    ax.plot([], [], ":", color="#555", label="random routing")
    ax.plot([], [], "--", color="#555", alpha=0.6, label="oracle routing (ceiling)")
    ax.axhline(base, color="#222", lw=1)
    ax.text(100, base, " PyBot, code only", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Share of turns answered by the model (%)")
    ax.set_ylabel("SmolTalkBench win rate (%)")
    ax.set_title("Code first, model only when the code should not answer", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "tradeoff.png"))


if __name__ == "__main__":
    main()
