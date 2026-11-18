"""Cache candidate: the cached model answer to the cached question closest in meaning.

    python bench/crystal_cand.py      # writes bench/out/crystal_{heldoutall,benchall}.jsonl
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "out")


def rows(p):
    return [json.loads(line) for line in open(p)]


def main():
    cache = rows(os.path.join(OUT, "crystal_a.jsonl"))
    CQ = np.load(os.path.join(OUT, "crystal_q_emb.npy"))
    held = rows(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))
    bench = {r["id"]: r for r in rows(os.path.join(ROOT, "bench", "data", "smoltalkbench.jsonl"))}
    border = [r["id"] for r in rows(os.path.join(OUT, "pybot_bench.jsonl"))]
    for tag, items, Q in (("heldoutall", held, np.load(os.path.join(OUT, "emb_heldout.npy"))),
                          ("benchall", [bench[i] for i in border], np.load(os.path.join(OUT, "emb_bench_q.npy")))):
        S = Q @ CQ.T
        best = S.argmax(1)
        out = [{"id": r["id"], "instruction": r["instruction"], "reply": cache[j]["reply"], "cached_q": cache[j]["instruction"],
                "similarity": float(S[k, j]), "crystal": int(j)} for k, (r, j) in enumerate(zip(items, best))]
        with open(os.path.join(OUT, "crystal_%s.jsonl" % tag), "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in out)
        sim = np.array([x["similarity"] for x in out])
        print("%s: %d prompts, cache similarity median %.2f, >=0.8: %.0f%%, >=0.9: %.0f%%"
              % (tag, len(out), np.median(sim), 100 * (sim >= .8).mean(), 100 * (sim >= .9).mean()))


if __name__ == "__main__":
    main()
