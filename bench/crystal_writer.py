"""Rewrite the answers of an existing answer cache with a different (stronger) writer model.

Retrieval is by cached *question* only, so a prompt always lands on the same cache entry whoever wrote the
answers. Only entries that some benchmark or held-out prompt retrieves can ever be served in the evaluation,
so we answer just those (1,791 of 33,133): the result is identical to rewriting the whole cache.

    BASE=crystal12l TAG=crystal12g WRITER_URL=http://127.0.0.1:8095/v1 python bench/crystal_writer.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
from hybrid.llm import BRIEF, OpenAILLM  # noqa: E402

OUT = os.path.join(ROOT, "bench", "out")
BASE, TAG = os.environ.get("BASE", "crystal12l"), os.environ["TAG"]
SPLITS = ("benchall", "heldoutall")


def rows(p):
    return [json.loads(line) for line in open(p)]


def main():
    base = {s: rows(os.path.join(OUT, "%s_%s.jsonl" % (BASE, s))) for s in SPLITS}
    need = {}
    for s in SPLITS:
        for r in base[s]:
            need[r["crystal"]] = r["cached_q"]
    part = os.path.join(OUT, TAG + "_answers.jsonl")
    have = {r["crystal"]: r["reply"] for r in rows(part)} if os.path.exists(part) else {}
    todo = [(j, q) for j, q in sorted(need.items()) if j not in have]
    print("%d entries needed, %d to write" % (len(need), len(todo)), flush=True)
    llm = OpenAILLM(os.environ["WRITER_URL"], os.environ.get("WRITER", "writer"), system=BRIEF)

    def one(item):
        j, q = item
        try:
            return j, q, llm.chat([{"role": "user", "content": q}])
        except Exception as e:  # noqa: BLE001
            print("fail", j, e, flush=True)
            return j, q, ""
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool, open(part, "a") as f:
        for n, (j, q, a) in enumerate(pool.map(one, todo)):
            if a:
                have[j] = a
                f.write(json.dumps({"crystal": j, "instruction": q, "reply": a}) + "\n"); f.flush()
            if n % 100 == 0:
                print("written %d/%d" % (n, len(todo)), flush=True)
    missing = [j for j in need if j not in have]
    assert not missing, "%d entries unanswered; rerun to resume" % len(missing)
    from embed import embed
    ids = sorted(have)
    E = embed([have[j] for j in ids])
    A = np.zeros((len(np.load(os.path.join(OUT, BASE + "_a_emb.npy"), mmap_mode="r")), E.shape[1]), dtype=E.dtype)
    A[ids] = E
    np.save(os.path.join(OUT, TAG + "_a_emb.npy"), A)
    for s in SPLITS:
        with open(os.path.join(OUT, "%s_%s.jsonl" % (TAG, s)), "w") as f:
            for r in base[s]:
                f.write(json.dumps(dict(r, reply=have[r["crystal"]], cache=TAG)) + "\n")
    print("done", TAG, flush=True)


if __name__ == "__main__":
    main()
