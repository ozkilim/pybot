"""Two-sided gate: predict the code's score and the model's score separately, defer on the gap.

  code side   base + inside + meaning (+ cross-encoder) signals        -> expected code score
  model side  question-only signals, question embedding (PCA), and the model's scores on the
              nearest already-judged questions                           -> expected model score
  defer when  expected model score - expected code score  is largest

    JUDGED_DIR=bench/poc600 python bench/two_sided.py [cv|bench]
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench")); sys.path.insert(0, ROOT)
from signals import Log, OUT, SHARES, build, by_id, curve, matrix, rows  # noqa: E402

QUESTION = ("ask_list", "ask_fact", "ask_how", "ask_why", "ask_school", "question_len", "question_marks")
JUDGED = os.environ.get("JUDGED_DIR", os.path.join(ROOT, "bench", "poc600"))
LLM = os.environ.get("LLM_RUN", "smollm2-360m-brief")
ridge = lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a))  # noqa: E731


def code_X(feats, ce, ids, use_ce=True):
    X, _ = matrix(feats, ("base", "inside", "meaning"))
    return np.hstack([X, np.array([[ce[i]] for i in ids])]) if use_ce else X


def llm_X(feats, q_emb, pca, hist):
    q = np.array([[f["base"][n] for n in QUESTION] + [f["inside"][k] for k in f["inside"] if k.startswith("wants_")]
                  + [f["meaning"]["near1"], f["meaning"]["near5"]] for f in feats])
    return np.hstack([q, pca.transform(q_emb), hist[:, None]])


def knn(q, Q, y, k=8):
    """Similarity-weighted mean of y over the k nearest rows of Q, for each row of q."""
    S = q @ Q.T
    out = np.zeros(len(q))
    for i in range(len(q)):
        top = np.argsort(-S[i])[:k]
        w = np.maximum(S[i, top], 0) ** 4
        out[i] = (y[top] * w).sum() / (w.sum() + 1e-9)
    return out


def fit_predict(Ftr, Etr, ce_tr, ids_tr, code_tr, llm_tr, both_tr, Fte, Ete, ce_te, ids_te, same=False):
    mc = ridge(30.0).fit(code_X(Ftr, ce_tr, ids_tr), code_tr)
    p_code = mc.predict(code_X(Fte, ce_te, ids_te))
    Eb, yb = Etr[both_tr], llm_tr[both_tr]
    pca = PCA(12, random_state=0).fit(Etr)
    # history for training rows is leave-one-out, so it is not the row's own label
    S = Eb @ Eb.T; np.fill_diagonal(S, -1)
    hist_tr = np.array([(lambda top: (yb[top] * np.maximum(S[i, top], 0) ** 4).sum() / ((np.maximum(S[i, top], 0) ** 4).sum() + 1e-9))(np.argsort(-S[i])[:8]) for i in range(len(Eb))])
    ml = ridge(30.0).fit(llm_X([f for f, b in zip(Ftr, both_tr) if b], Eb, pca, hist_tr), yb)
    p_llm = ml.predict(llm_X(Fte, Ete, pca, knn(Ete, Eb, yb)))
    return p_code, p_llm


def load_heldout(log):
    T = rows(os.path.join(OUT, "pybot_heldout600_trace.jsonl"))
    J = by_id(os.path.join(JUDGED, "pybot_heldout.judged.jsonl")); M = by_id(os.path.join(JUDGED, LLM + "_heldout.judged.jsonl"))
    T = [r for r in T if r["id"] in J]
    pos = np.array([log.pos[r["id"]] for r in T])
    code = np.array([J[r["id"]]["score"] for r in T]); both = np.array([r["id"] in M for r in T])
    llm = np.array([M[r["id"]]["score"] if r["id"] in M else 0.0 for r in T])
    return T, pos, code, llm, both


def show(title, curves, code, llm):
    print("\n" + title)
    print("%-40s %s" % ("model share:", "  ".join("%4.0f%%" % (100 * s) for s in SHARES)))
    print("%-40s %s" % ("random routing", "  ".join("%5.1f" % (100 * (code.mean() + s * (llm.mean() - code.mean()))) for s in SHARES)))
    for name, rank in curves:
        print("%-40s %s" % (name, "  ".join("%5.1f" % v for v in curve(rank, code, llm))))
    print("%-40s %s" % ("oracle", "  ".join("%5.1f" % v for v in curve(llm - code, code, llm))))
    print("code alone %.1f | model alone %.1f | best of both %.1f" % (100 * code.mean(), 100 * llm.mean(), 100 * np.maximum(code, llm).mean()))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    log = Log()
    T, pos, code, llm, both = load_heldout(log)
    ce = json.load(open(os.path.join(OUT, "ce_heldout.json")))
    ids = [r["id"] for r in T]
    F = build(T, log.q[pos], log.code[pos], log, True)
    E = log.q[pos]
    print("%d held-out with code scores, %d with model scores" % (len(T), both.sum()))
    if mode == "cv":
        pc, pl = np.zeros(len(T)), np.zeros(len(T))
        for seed in range(3):
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(T):
                a, b = fit_predict([F[i] for i in tr], E[tr], ce, [ids[i] for i in tr], code[tr], llm[tr], both[tr],
                                   [F[i] for i in te], E[te], ce, [ids[i] for i in te])
                pc[te] += a / 3; pl[te] += b / 3
        k = both
        from sklearn.metrics import roc_auc_score
        print("out-of-fold AUROC: code win %.3f | model win %.3f" % (roc_auc_score(code >= .5, pc), roc_auc_score(llm[k] >= .5, pl[k])))
        show("held-out, 5-fold x3", [("code side only", -pc[k]), ("two-sided (model - code)", (pl - pc)[k])], code[k], llm[k])
    else:
        B = rows(os.path.join(OUT, "pybot_bench100_trace.jsonl"))
        Jb = by_id(os.path.join(ROOT, "bench", "poc", "pybot_bench.judged.jsonl")); Mb = by_id(os.path.join(ROOT, "bench", "poc", LLM + "_bench.judged.jsonl"))
        allb = [r["id"] for r in rows(os.path.join(OUT, "pybot_bench.jsonl"))]
        Eq, Ec = np.load(os.path.join(OUT, "emb_bench_q.npy")), np.load(os.path.join(OUT, "emb_bench_code.npy"))
        ix = [allb.index(r["id"]) for r in B]
        Fb = build(B, Eq[ix], Ec[ix], log, False)
        ceb = json.load(open(os.path.join(OUT, "ce_bench.json")))
        pc, pl = fit_predict(F, E, ce, ids, code, llm, both, Fb, Eq[ix], ceb, [r["id"] for r in B])
        cb = np.array([Jb[r["id"]]["score"] for r in B]); lb = np.array([Mb[r["id"]]["score"] for r in B])
        show("benchmark sample (100 prompts), gate fitted on held-out only", [("code side only", -pc), ("two-sided (model - code)", pl - pc)], cb, lb)
        json.dump({"ids": [r["id"] for r in B], "p_code": pc.tolist(), "p_llm": pl.tolist()}, open(os.path.join(OUT, "two_sided_bench.json"), "w"))


if __name__ == "__main__":
    main()
