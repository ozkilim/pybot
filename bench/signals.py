"""Richer gate signals, and a cross-validated comparison of gates built from them.

Signal groups, all computed without generating text:
  base      the 27 black-box signals in hybrid/features.py
  inside    PyBot's own record of how it answered: which knowledge entry, how many of the question's
            words that entry's keywords cover, whether the entry's name is about the question,
            how many entries competed, what kind of answer the question wants
  meaning   embeddings (all-MiniLM, 23M parameters): is the reply about the question; does it agree
            with logged answers to the nearest logged questions; how close is the nearest one
  history   what happened on the nearest already-judged prompts (code's score there)

    JUDGED_DIR=bench/poc600 python bench/signals.py            # cross-validation on held-out prompts
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from hybrid.features import VECTOR, content_words, extract  # noqa: E402

OUT = os.path.join(ROOT, "bench", "out")
WANTS = ("options", "yesno", "steps", "meaning", "compare", "pick", "amount", "reason")


def rows(path):
    return [json.loads(line) for line in open(path)]


def by_id(path):
    return {r["id"]: r for r in rows(path)}


def inside(row):
    t = row.get("trace") or {}
    q = content_words(row["instruction"])
    keys = {k.lower() for k in t.get("keys") or []}
    key_stems = content_words(" ".join(keys))
    name = content_words((t.get("entry_id") or "").replace("-", " "))
    hit = q & key_stems
    f = {"has_entry": 1.0 if t.get("entry_id") else 0.0,
         "key_cover": len(hit) / len(q) if q else 0.0,           # share of the question the entry's keywords cover
         "key_hits": min(len(hit), 6) / 6.0,
         "key_miss": 1.0 if t.get("entry_id") and not hit else 0.0,
         "name_hit": len(q & name) / len(name) if name else 0.0,  # is the entry's own name in the question
         "name_miss": 1.0 if name and not (q & name) else 0.0,
         "n_entries": min(t.get("n_entries") or 0, 4) / 4.0,
         "n_keys": min(len(keys), 60) / 60.0}
    for w in WANTS:
        f["wants_" + w] = 1.0 if t.get("wants") == w else 0.0
    f["wants_none"] = 1.0 if not t.get("wants") else 0.0
    return f


class Log:
    """The logged held-out prompts: question, reference answer and PyBot reply embeddings."""

    def __init__(self):
        self.rows = rows(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))
        self.pos = {r["id"]: k for k, r in enumerate(self.rows)}
        self.q = np.load(os.path.join(OUT, "emb_heldout.npy"))
        self.ref = np.load(os.path.join(OUT, "emb_heldout_ref.npy"))
        self.code = np.load(os.path.join(OUT, "emb_heldout_code.npy"))

    def meaning(self, q, code, skip=None, k=5):
        """q, code: embeddings of this question and of PyBot's reply to it."""
        sims = self.q @ q
        if skip is not None:
            sims[skip] = -1
        top = np.argsort(-sims)[:k]
        agree = self.ref[top] @ code
        w = np.maximum(sims[top], 0)
        return {"about": float(q @ code), "near1": float(sims[top[0]]), "near5": float(sims[top].mean()),
                "agree1": float(agree[0]), "agree_max": float(agree.max()),
                "agree_w": float((agree * w).sum() / (w.sum() + 1e-9))}

    def history(self, q, labelled, scores, skip=None, k=8):
        """Mean code score on the k nearest judged prompts. labelled: log positions, scores: their values."""
        sims = self.q[labelled] @ q
        if skip is not None:
            sims[labelled == skip] = -1
        top = np.argsort(-sims)[:k]
        w = np.maximum(sims[top], 0) ** 4
        return {"hist_code": float((scores[top] * w).sum() / (w.sum() + 1e-9)), "hist_near": float(sims[top[0]])}


def build(trace_rows, q_emb, code_emb, log, loo, labelled=None, scores=None, mask=None):
    """Feature dicts per group for each row. `mask[i]` = labelled positions row i may see (CV)."""
    out = []
    for i, row in enumerate(trace_rows):
        skip = log.pos[row["id"]] if loo else None
        f = {"base": extract(row["instruction"], row), "inside": inside(row),
             "meaning": log.meaning(q_emb[i], code_emb[i], skip)}
        if labelled is not None:
            keep = mask[i] if mask is not None else np.ones(len(labelled), bool)
            f["history"] = log.history(q_emb[i], labelled[keep], scores[keep], skip)
        out.append(f)
    return out


def matrix(feats, groups):
    names = [(g, n) for g in groups for n in feats[0][g]]
    return np.array([[f[g][n] for g, n in names] for f in feats]), names


SHARES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75)


def curve(rank, code, llm):
    order = np.argsort(-rank, kind="stable")
    out = []
    for share in SHARES:
        defer = np.zeros(len(code), bool)
        defer[order[:int(round(share * len(code)))]] = True
        out.append(100 * np.where(defer, llm, code).mean())
    return out


def models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
            "boost": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.04,
                                                           min_samples_leaf=15, l2_regularization=1.0)}


def main():
    from sklearn.model_selection import KFold
    judged = os.environ.get("JUDGED_DIR", os.path.join(ROOT, "bench", "poc600"))
    llm_run = sys.argv[1] if len(sys.argv) > 1 else "smollm2-360m-brief"
    log = Log()
    T = rows(os.path.join(OUT, "pybot_heldout600_trace.jsonl"))
    J = by_id(os.path.join(judged, "pybot_heldout.judged.jsonl"))
    M = by_id(os.path.join(judged, llm_run + "_heldout.judged.jsonl"))
    T = [r for r in T if r["id"] in J]
    pos = np.array([log.pos[r["id"]] for r in T])
    code = np.array([J[r["id"]]["score"] for r in T])
    both = np.array([r["id"] in M for r in T])
    llm = np.array([M[r["id"]]["score"] if r["id"] in M else np.nan for r in T])
    q_emb, c_emb = log.q[pos], log.code[pos]
    print("%d held-out prompts with code scores, %d with %s scores too" % (len(T), both.sum(), llm_run))
    print("on those: code %.1f%% | model %.1f%% | best of both %.1f%%"
          % (100 * code[both].mean(), 100 * llm[both].mean(), 100 * np.maximum(code, llm)[both].mean()))

    sets = {"base": ("base",), "base+inside": ("base", "inside"), "base+meaning": ("base", "meaning"),
            "base+inside+meaning": ("base", "inside", "meaning"),
            "base+inside+meaning+history": ("base", "inside", "meaning", "history")}
    results = []
    for target in ("code", "regret"):
        for set_name, groups in sets.items():
            for model_name, make in models().items():
                rank = np.zeros(len(T))
                for seed in range(3):
                    for tr, te in KFold(5, shuffle=True, random_state=seed).split(T):
                        mask = np.zeros((len(T), len(T)), bool)
                        mask[:, tr] = True               # neighbours only from the training fold
                        feats = build(T, q_emb, c_emb, log, True, pos, code, mask)
                        X, _ = matrix(feats, groups)
                        if target == "code":
                            m = make().fit(X[tr], code[tr])
                            rank[te] += -m.predict(X[te]) / 3
                        else:
                            trb = tr[both[tr]]
                            m = make().fit(X[trb], (llm - code)[trb])
                            rank[te] += m.predict(X[te]) / 3
                c = curve(rank[both], code[both], llm[both])
                results.append((float(np.mean(c)), "%-6s %-28s %s" % (target, set_name, model_name), c))
    rand = [100 * (code[both].mean() + s * (llm[both].mean() - code[both].mean())) for s in SHARES]
    orac = curve((llm - code)[both], code[both], llm[both])
    print("\n%-44s %s   mean" % ("model share:", "  ".join("%4.0f%%" % (100 * s) for s in SHARES)))
    for mean, name, c in [(np.mean(rand), "random routing", rand)] + sorted(results) + [(np.mean(orac), "oracle", orac)]:
        print("%-44s %s   %.1f" % (name, "  ".join("%5.1f" % v for v in c), mean))


if __name__ == "__main__":
    main()
