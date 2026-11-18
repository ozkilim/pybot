"""Crystallize the model into Python: write its answers ahead of time, serve them by lookup.

  questions  Qwen2.5-7B writes everyday questions for each of the 428 topic categories of the
             held-out prompts (never the benchmark's topics or prompts)
  answers    SmolLM2-360M answers each one, exactly as it would at run time (brief prompt, greedy)
  embed      all-MiniLM embeddings of the questions, for nearest-question lookup

All three steps are resumable. Model calls happen here, offline, once; at run time the cache is a
Python dictionary lookup.

    python bench/crystallize.py questions 25      # questions per category
    python bench/crystallize.py answers
    python bench/crystallize.py embed
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
from hybrid.llm import BRIEF, OllamaLLM  # noqa: E402

OUT = os.path.join(ROOT, "bench", "out")
QFILE, AFILE = os.path.join(OUT, "crystal_q.jsonl"), os.path.join(OUT, "crystal_a.jsonl")
WRITER = "qwen2.5:7b"
ANSWERER = "smollm2:360m"


def categories():
    held = [json.loads(line) for line in open(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))]
    return sorted({" / ".join(p.strip() for p in r["topic"].split("/")[:2]).lower() for r in held})


def done(path, field):
    return {json.loads(line)[field] for line in open(path)} if os.path.exists(path) else set()


def questions(per_cat):
    writer = OllamaLLM(WRITER, system=None, max_tokens=900)
    have = done(QFILE, "category")
    todo = [c for c in categories() if c not in have]

    def one(cat):
        prompt = ("Write %d different questions that a person might type to a friendly chat assistant about: %s.\n"
                  "Everyday, casual, varied (how-to, why, advice, recommendations, simple facts). Each under 25 words.\n"
                  "One question per line, no numbering, nothing else." % (per_cat, cat))
        text = _raw(writer, prompt)
        qs = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines()]
        return cat, [q for q in qs if 8 <= len(q) <= 200][:per_cat]

    with ThreadPoolExecutor(2) as pool, open(QFILE, "a") as f:
        for n, (cat, qs) in enumerate(pool.map(one, todo)):
            for q in qs:
                f.write(json.dumps({"category": cat, "instruction": q}) + "\n")
            f.flush()
            if n % 20 == 0:
                print("questions: %d/%d categories" % (n + 1, len(todo)), flush=True)


def _raw(llm, prompt):
    """Keep line breaks (OllamaLLM.chat joins whitespace)."""
    from hybrid.llm import _post
    body = {"model": llm.model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": 0.8, "seed": 0, "num_predict": llm.max_tokens, "num_ctx": 4096}}
    return _post(llm.host + "/api/chat", body)["message"]["content"]


def answers():
    model = OllamaLLM(ANSWERER, system=BRIEF)
    seen = set()
    qs = []
    for line in open(QFILE):
        q = json.loads(line)["instruction"]
        if q.lower() not in seen:
            seen.add(q.lower()); qs.append(q)
    have = done(AFILE, "instruction")
    todo = [q for q in qs if q not in have]
    print("%d unique questions, %d to answer" % (len(qs), len(todo)), flush=True)

    def one(q):
        try:
            return q, model.chat([{"role": "user", "content": q}])
        except Exception:  # noqa: BLE001
            return q, ""
    with ThreadPoolExecutor(4) as pool, open(AFILE, "a") as f:
        for n, (q, a) in enumerate(pool.map(one, todo)):
            if a:
                f.write(json.dumps({"instruction": q, "reply": a}) + "\n")
            if n % 500 == 0:
                f.flush(); print("answers: %d/%d" % (n, len(todo)), flush=True)


def embed_all():
    from embed import embed
    rows = [json.loads(line) for line in open(AFILE)]
    np.save(os.path.join(OUT, "crystal_q_emb.npy"), embed([r["instruction"] for r in rows]))
    np.save(os.path.join(OUT, "crystal_a_emb.npy"), embed([r["reply"] for r in rows]))
    print("embedded", len(rows))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "questions":
        questions(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif step == "answers":
        answers()
    else:
        embed_all()
