"""Python-side candidate replies for every benchmark prompt (no text generation).

  entry-emb-trim2   the PyBot knowledge entry closest in meaning to the question, first two sentences
  recall-emb        the logged reference answer to the closest logged held-out question

The answer log is the 1,198 held-out prompts only, so no benchmark answer is ever recalled.

    python bench/make_candidates.py          # writes bench/out/{entry-emb-trim2,recall-emb}_benchall.jsonl
"""
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "out")


def rows(p):
    return [json.loads(line) for line in open(p)]


def trim(t, n=2):
    return " ".join(re.split(r"(?<=[.!?])\s+", t.strip())[:n])


def main():
    ents = json.load(open(os.path.join(OUT, "pybot_entries.json")))
    eids = json.load(open(os.path.join(OUT, "emb_entries_ids.json")))
    EE = np.load(os.path.join(OUT, "emb_entries.npy"))
    held = rows(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))
    hq = np.load(os.path.join(OUT, "emb_heldout.npy"))
    order = [r["id"] for r in rows(os.path.join(OUT, "pybot_bench.jsonl"))]
    bq = np.load(os.path.join(OUT, "emb_bench_q.npy"))
    bench = {r["id"]: r for r in rows(os.path.join(ROOT, "bench", "data", "smoltalkbench.jsonl"))}
    ent, rec = [], []
    for k, i in enumerate(order):
        q, r = bq[k], bench[i]
        s = EE @ q
        j = int(s.argmax())
        ent.append({"id": i, "instruction": r["instruction"], "reply": trim(ents[eids[j]]), "entry": eids[j], "similarity": float(s[j])})
        s = hq @ q
        j = int(s.argmax())
        rec.append({"id": i, "instruction": r["instruction"], "reply": held[j]["output"], "similarity": float(s[j]), "doc": j})
    for name, o in (("entry-emb-trim2", ent), ("recall-emb", rec)):
        with open(os.path.join(OUT, name + "_benchall.jsonl"), "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in o)
    print("wrote", len(ent), "prompts; recall similarity median %.2f" % np.median([x["similarity"] for x in rec]))


if __name__ == "__main__":
    main()
