"""Freeze the chooser and router into hybrid/pybot70_assets/ so PyBot-70 can run live.

Everything is fitted on held-out prompts only. The deferral threshold is the 70th percentile of
the router's score on the benchmark prompts: it uses no benchmark labels, only how often turns of
that kind arrive, which is what sets "send about 30% of turns to the model".

    python bench/export_pybot70.py
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench")); sys.path.insert(0, ROOT)
os.environ.setdefault("BENCH", "all")
import chooser  # noqa: E402
from signals import Log, OUT, by_id, rows  # noqa: E402

DST = os.path.join(ROOT, "hybrid", "pybot70_assets")
MODEL_SHARE = 0.30


def main():
    from sentence_transformers import CrossEncoder
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    os.makedirs(DST, exist_ok=True)
    log = Log()
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    H = chooser.Data("heldout", log, ce)
    X, y, _, _ = H.features()
    ok = ~np.isnan(y)
    scorer = chooser.boost().fit(X[ok], y[ok])

    # the model side, exactly as chooser.llm_side fits it
    M = by_id(os.path.join(ROOT, "bench", "poc1198", chooser.LLM_SRC + "_heldout.judged.jsonl"))
    ids = [i for i in log.pos if i in M]
    E = log.q[[log.pos[i] for i in ids]]
    llm_train = np.array([M[i]["score"] for i in ids])
    pca = PCA(12, random_state=0).fit(E)
    S = E @ E.T
    hist = np.zeros(len(E))
    for i in range(len(E)):
        S[i, i] = -1
        top = np.argsort(-S[i])[:8]
        w = np.maximum(S[i, top], 0) ** 4
        hist[i] = (llm_train[top] * w).sum() / (w.sum() + 1e-9)
    ridge = Ridge(alpha=30.0).fit(np.hstack([pca.transform(E), hist[:, None]]), llm_train)

    saved = json.load(open(os.path.join(OUT, "chooser_benchall.json")))
    rank = np.array(saved["p_llm"]) - np.array(saved["p_best"])
    threshold = float(np.quantile(rank, 1 - MODEL_SHARE))

    held = rows(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))
    np.save(os.path.join(DST, "log_q.npy"), log.q)
    np.save(os.path.join(DST, "log_ref.npy"), log.ref)
    json.dump([r["output"] for r in held], open(os.path.join(DST, "log_answers.json"), "w"))
    np.save(os.path.join(DST, "llm_E.npy"), E)
    np.save(os.path.join(DST, "entry_emb.npy"), np.load(os.path.join(OUT, "emb_entries.npy")))
    ents = json.load(open(os.path.join(OUT, "pybot_entries.json")))
    eids = json.load(open(os.path.join(OUT, "emb_entries_ids.json")))
    json.dump({"ids": eids, "text": [ents[e] for e in eids]}, open(os.path.join(DST, "entries.json"), "w"))
    with open(os.path.join(DST, "models.pkl"), "wb") as f:
        pickle.dump({"scorer": scorer, "pca": pca, "ridge": ridge, "llm_train": llm_train,
                     "threshold": threshold, "model_share": MODEL_SHARE, "sources": list(chooser.USE),
                     "all_sources": list(chooser.SOURCES)}, f)
    print("saved to", DST, "| threshold %.4f for %.0f%% model share" % (threshold, 100 * MODEL_SHARE))


if __name__ == "__main__":
    main()
