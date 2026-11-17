"""Compare gate designs by cross-validation on held-out prompts only.

Each design ranks turns by how much it expects the model to beat the code. For a given share of
turns sent to the model, the hybrid's win rate is read off already-judged replies. Reported: the
mean win rate over a range of model shares (area under the trade-off curve), out of fold.

    JUDGED_DIR=bench/poc600 python bench/gate_search.py smollm2-360m-brief
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hybrid.features import VECTOR, extract  # noqa: E402

OUT = os.path.join(ROOT, "bench", "out")
JUDGED = os.environ.get("JUDGED_DIR", OUT)
SHARES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
GUARD = ("bias", "abstains", "hands_back", "sorry", "reply_is_question", "module_answers", "module_greeting",
         "module_the_shape_of_the_answer", "module_chat", "module_other")
SUPPORT = GUARD + ("overlap_question", "overlap_reply", "no_overlap", "reply_len", "n_knowledge_candidates")


def load(name):
    path = os.path.join(JUDGED if name.endswith(".judged.jsonl") else OUT, name)
    return {r["id"]: r for r in map(json.loads, open(path))}


def soft_fit(X, y, C):
    Xd, yd, wd = np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))], np.r_[y, 1 - y]
    keep = wd > 1e-9
    return LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(Xd[keep], yd[keep], sample_weight=wd[keep])


def curve(rank, code, llm):
    """Hybrid win rate at each share, deferring the highest-ranked turns first."""
    order = np.argsort(-rank, kind="stable")
    out = []
    for share in SHARES:
        defer = np.zeros(len(code), bool)
        defer[order[:int(round(share * len(code)))]] = True
        out.append(100 * np.where(defer, llm, code).mean())
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "smollm2-360m-brief"
    R, J, M = load("pybot_heldout.jsonl"), load("pybot_heldout.judged.jsonl"), load(run + "_heldout.judged.jsonl")
    ids = [i for i in J if i in M]
    code = np.array([J[i]["score"] for i in ids])
    llm = np.array([M[i]["score"] for i in ids])
    F = [extract(R[i]["instruction"], R[i]) for i in ids]
    X = np.array([[f[n] for n in VECTOR] for f in F])
    cols = lambda names: [VECTOR.index(n) for n in names]  # noqa: E731
    print("%d held-out prompts | code %.1f%% | %s %.1f%% | best of both %.1f%%"
          % (len(ids), 100 * code.mean(), run, 100 * llm.mean(), 100 * np.maximum(code, llm).mean()))

    designs = {}
    # zero parameters: defer when PyBot's own fallback answered, most explicit admissions first
    designs["rule: fallback module, then abstains, then no overlap"] = lambda tr, te: (
        2 * (1 - X[te][:, VECTOR.index("module_answers")]) + X[te][:, VECTOR.index("abstains")]
        + 0.5 * X[te][:, VECTOR.index("no_overlap")] - 0.25 * X[te][:, VECTOR.index("overlap_question")])
    for label, names in (("guard signals", GUARD), ("guard + support", SUPPORT), ("all 27 signals", VECTOR)):
        for C in (0.05, 0.5, 5.0):
            def design(tr, te, names=names, C=C):
                k = cols(names)
                return -soft_fit(X[tr][:, k], code[tr], C).predict_proba(X[te][:, k])[:, 1]
            designs["logistic p_code: %s, C=%g" % (label, C)] = design
    # predict the gain directly: who wins this prompt, weighted by how much
    for C in (0.05, 0.5):
        def gain(tr, te, C=C):
            d = llm[tr] - code[tr]
            m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(
                X[tr], (d > 0).astype(int), sample_weight=np.abs(d) + 1e-6)
            return m.predict_proba(X[te])[:, 1]
        designs["logistic gain (model beats code), all signals, C=%g" % C] = gain

    rng = np.random.default_rng(0)
    rows = []
    for name, design in designs.items():
        rank = np.zeros(len(ids))
        for seed in range(3):
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                rank[te] += design(tr, te) / 3
        c = curve(rank, code, llm)
        rows.append((float(np.mean(c)), name, c))
    rand = [100 * (code.mean() + s * (llm.mean() - code.mean())) for s in SHARES]
    orac = curve(llm - code, code, llm)
    print("\n%-58s %s   mean" % ("model share:", "  ".join("%4.0f%%" % (100 * s) for s in SHARES)))
    for mean, name, c in [(float(np.mean(rand)), "random routing", rand)] + sorted(rows) + [(float(np.mean(orac)), "oracle", orac)]:
        print("%-58s %s   %.1f" % (name, "  ".join("%5.1f" % v for v in c), mean))
    del rng


if __name__ == "__main__":
    main()
