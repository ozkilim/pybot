"""Fit the gate on held-out prompts and write hybrid/gate_weights.json.

Held-out = the 1,198 everyday-conversations prompts that are NOT in SmolTalkBench. Nothing from
the benchmark's 1,000 prompts is read here, so the benchmark stays a clean test of the gate.

    python bench/fit_gate.py smollm2-360m-brief smollm2-1.7b-brief gemma3-1b-brief
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hybrid.features import VECTOR, extract, reply_key  # noqa: E402

OUT = os.path.join(ROOT, "bench", "out")
BUDGETS = ("0.01", "0.02", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.40", "0.50", "0.75", "1.00")
QUESTION_ONLY = ("bias", "ask_list", "ask_fact", "ask_how", "ask_why", "ask_school", "question_len", "question_marks")
OLLAMA_NAMES = {"smollm2-360m-brief": "smollm2:360m", "smollm2-1.7b-brief": "smollm2:1.7b",
                "gemma3-1b-brief": "gemma3:1b"}


JUDGED = os.environ.get("JUDGED_DIR", OUT)   # e.g. bench/poc for the small proof-of-concept subset


def load(name):
    path = os.path.join(JUDGED if name.endswith(".judged.jsonl") else OUT, name)
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def fit(X, scores, names):
    """Logistic regression on soft targets: each row counts as a win with weight = its score."""
    Xd = np.vstack([X, X])
    yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
    wd = np.r_[scores, 1 - scores]
    keep = wd > 1e-9
    model = LogisticRegression(C=0.5, fit_intercept=False, max_iter=2000)
    model.fit(Xd[keep], yd[keep], sample_weight=wd[keep])
    return model, dict(zip(names, map(float, model.coef_[0])))


def main():
    llms = sys.argv[1:] or ["smollm2-360m-brief"]
    replies, judged = load("pybot_heldout.jsonl"), load("pybot_heldout.judged.jsonl")
    ids = [i for i in replies if i in judged]
    counts = Counter(reply_key(replies[i]["reply"]) for i in ids)
    stock = {k: n for k, n in counts.items() if n >= 2}

    def vector(i, loo):
        row = replies[i]
        table = dict(counts)
        if loo:  # a prompt must not count its own reply as evidence that the reply is a stock line
            table[reply_key(row["reply"])] -= 1
        f = extract(row["instruction"], row, table)
        return [f[name] for name in VECTOR]

    X = np.array([vector(i, True) for i in ids])
    y = np.array([judged[i]["score"] for i in ids])
    code_model, code_w = fit(X, y, VECTOR)
    hard = (y >= 0.5).astype(int)
    cv = cross_val_predict(LogisticRegression(C=0.5, fit_intercept=False, max_iter=2000), X, hard, cv=5,
                           method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    print("code tier: %d held-out prompts, win rate %.1f%%, gate AUROC (5-fold) %.3f"
          % (len(ids), 100 * y.mean(), roc_auc_score(hard, cv)))
    p_code = code_model.predict_proba(X)[:, 1]

    data = {"features": list(VECTOR), "stock": stock, "code": code_w, "llm": {}, "default_budget": "0.30",
            "fitted_on": {"prompts": len(ids), "set": "smoltalk everyday-conversations minus SmolTalkBench"}}
    q_index = [VECTOR.index(n) for n in QUESTION_ONLY]
    for name in llms:
        lj = load(name + "_heldout.judged.jsonl")
        lids = [i for i in ids if i in lj]
        if not lids:
            print("skip", name, "(no judged held-out replies)")
            continue
        rows = [ids.index(i) for i in lids]
        Xq = X[rows][:, q_index]
        yl = np.array([lj[i]["score"] for i in lids])
        llm_model, llm_w = fit(Xq, yl, QUESTION_ONLY)
        p_llm = llm_model.predict_proba(X[:, q_index])[:, 1]
        margin = p_llm - p_code
        taus = {b: float(np.quantile(margin, 1 - float(b))) - 1e-9 if float(b) < 1 else -2.0 for b in BUDGETS}
        key = OLLAMA_NAMES.get(name, name)
        data["llm"][key] = {"weights": llm_w, "taus": taus, "run": name,
                            "heldout_win_rate": float(yl.mean()), "heldout_prompts": len(lids)}
        print("%s: %d held-out prompts, win rate %.1f%%, tau@10%%=%.3f" % (name, len(lids), 100 * yl.mean(), taus["0.10"]))
    if data["llm"]:
        data["default_llm"] = next(iter(data["llm"]))
    with open(os.path.join(ROOT, "hybrid", "gate_weights.json"), "w") as handle:
        json.dump(data, handle, indent=1)
    print("code-tier weights, most against -> most for:")
    for name, w in sorted(code_w.items(), key=lambda kv: kv[1]):
        print("  %-24s %+.2f" % (name, w))


if __name__ == "__main__":
    main()
