"""More cache candidates per turn, and question-to-question paraphrase signals.

A cached answer is only right if the cached question is really the user's question. Two small
cross-encoders measure that directly:
  qq_sts   graded similarity (cross-encoder/stsb-distilroberta-base, 0-1)
  qq_dup   strict duplicate-question probability (cross-encoder/quora-distilroberta-base)

Writes, for heldoutall and benchall:
  crystal12-sts   the cached answer whose question has the best qq_sts among the 20 nearest
  crystal12-k2/3  the 2nd and 3rd nearest cached questions by embedding
and adds qq_sts / qq_dup to the existing crystal12 and recall-emb candidate files (replies unchanged).

    python bench/crystal_more.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "out")
TAG, K = "crystal12", 20


def rows(p):
    return [json.loads(line) for line in open(p)]


def write(path, rs):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rs)


def main():
    from sentence_transformers import CrossEncoder
    sts = CrossEncoder("cross-encoder/stsb-distilroberta-base", device="mps")
    dup = CrossEncoder("cross-encoder/quora-distilroberta-base", device="mps")
    cache = rows(os.path.join(OUT, TAG + "_a.jsonl"))
    CQ = np.load(os.path.join(OUT, TAG + "_q_emb.npy"))
    held = rows(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))
    for split, qfile in (("heldoutall", "emb_heldout.npy"), ("benchall", "emb_bench_q.npy")):
        base = rows(os.path.join(OUT, "%s_%s.jsonl" % (TAG, split)))           # same order as the question embeddings
        Q = np.load(os.path.join(OUT, qfile))
        S = Q @ CQ.T
        top = np.argsort(-S, axis=1)[:, :K]
        pairs = [(r["instruction"], cache[j]["instruction"]) for r, t in zip(base, top) for j in t]
        vs = sts.predict(pairs, batch_size=128).reshape(len(base), K)
        vd = dup.predict(pairs, batch_size=128).reshape(len(base), K)
        out = {"%s-sts" % TAG: [], "%s-k2" % TAG: [], "%s-k3" % TAG: []}
        for k, (r, t) in enumerate(zip(base, top)):
            def cand(pos):
                j = int(t[pos])
                return {"id": r["id"], "instruction": r["instruction"], "reply": cache[j]["reply"], "cached_q": cache[j]["instruction"],
                        "similarity": float(S[k, j]), "crystal": j, "cache": TAG, "qq_sts": float(vs[k, pos]), "qq_dup": float(vd[k, pos])}
            out["%s-sts" % TAG].append(cand(int(vs[k].argmax())))
            out["%s-k2" % TAG].append(cand(1))
            out["%s-k3" % TAG].append(cand(2))
            r["qq_sts"], r["qq_dup"] = float(vs[k, 0]), float(vd[k, 0])
        write(os.path.join(OUT, "%s_%s.jsonl" % (TAG, split)), base)
        for name, rs in out.items():
            write(os.path.join(OUT, "%s_%s.jsonl" % (name, split)), rs)
        same = np.mean([a["reply"] == b["reply"] for a, b in zip(base, out["%s-sts" % TAG])])
        print("%s: paraphrase pick differs from nearest on %.0f%%; qq_sts of nearest median %.2f, of pick %.2f"
              % (split, 100 * (1 - same), np.median(vs[:, 0]), np.median(vs.max(1))), flush=True)
        # the logged-answer candidate: compare with the logged question it came from
        rec = rows(os.path.join(OUT, "recall-emb_%s.jsonl" % split))
        pairs = [(r["instruction"], held[r["doc"]]["instruction"]) for r in rec]
        for r, a, b in zip(rec, sts.predict(pairs, batch_size=128), dup.predict(pairs, batch_size=128)):
            r["qq_sts"], r["qq_dup"] = float(a), float(b)
        write(os.path.join(OUT, "recall-emb_%s.jsonl" % split), rec)


if __name__ == "__main__":
    main()
