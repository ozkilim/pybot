"""Run PyBot over a prompt file, one fresh conversation per prompt.

Records the chosen reply, the module that produced it, every candidate the modules
proposed, and the latency. That trace is what the cascade's gate is built from.

    python bench/run_pybot.py bench/data/smoltalkbench.jsonl bench/out/pybot_bench.jsonl
"""
import json
import os
import random
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_runtime = None


def _init():
    global _runtime
    sys.path.insert(0, ROOT)
    from bot import runtime
    _runtime = runtime


def _one(row):
    random.seed(0)
    start = time.perf_counter()
    try:
        result = _runtime.turn(row["instruction"], {})
        reply = " ".join(str(result["reply"]).split()) or "Hello!"
        module = result.get("module")
        candidates = [{"module": c["module"], "priority": c["priority"],
                       "reply": " ".join(str(c["reply"]).split())} for c in result.get("candidates", [])]
    except Exception:  # noqa: BLE001 - same contract as pybot.py: a turn never fails
        reply, module, candidates = "Hello!", None, []
    return {"id": row["id"], "instruction": row["instruction"], "reply": reply, "module": module,
            "candidates": candidates, "seconds": time.perf_counter() - start}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rows = [json.loads(line) for line in open(src)]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Pool(workers, initializer=_init) as pool, open(dst, "w") as out:
        for i, rec in enumerate(pool.imap(_one, rows, chunksize=4)):
            out.write(json.dumps(rec) + "\n")
            if (i + 1) % 100 == 0:
                print(i + 1, "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
