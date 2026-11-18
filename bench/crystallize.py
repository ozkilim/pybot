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
TAG = os.environ.get("CRYSTAL", "crystal")      # crystal2 = second, larger cache (questions2)
QFILE, AFILE = os.path.join(OUT, TAG + "_q.jsonl"), os.path.join(OUT, TAG + "_a.jsonl")
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


def questions2(per_topic):
    """Opening messages in the style of real chat openers, one batch per held-out topic (1,198),
    with three held-out prompts from other topics as style examples."""
    import random
    held = [json.loads(line) for line in open(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))]
    writer = OllamaLLM(WRITER, system=None, max_tokens=1200)
    have = done(QFILE, "category")
    todo = [(k, r["topic"]) for k, r in enumerate(held) if r["topic"] not in have]

    def one(item):
        k, topic = item
        rng = random.Random(k)
        ex = [held[j]["instruction"] for j in rng.sample([j for j in range(len(held)) if j != k], 3)]
        prompt = ("Here are three example messages people sent to a friendly chat assistant:\n- %s\n- %s\n- %s\n\n"
                  "Write %d new, different opening messages in the same style about the topic: %s.\n"
                  "Vary them: some are one direct question, some give a short personal context sentence first and then ask. "
                  "Be specific and concrete. One message per line, no numbering, nothing else." % (ex[0], ex[1], ex[2], per_topic, topic))
        qs = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"') for line in _raw(writer, prompt).splitlines()]
        return topic, [q for q in qs if 8 <= len(q) <= 300][:per_topic]

    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool, open(QFILE, "a") as f:
        for n, (topic, qs) in enumerate(pool.map(one, todo)):
            for q in qs:
                f.write(json.dumps({"category": topic, "instruction": q}) + "\n")
            f.flush()
            if n % 50 == 0:
                print("questions2: %d/%d topics" % (n + 1, len(todo)), flush=True)


def _raw(llm, prompt):
    """Keep line breaks (OllamaLLM.chat joins whitespace). WRITER_URL: an OpenAI-compatible server instead."""
    from hybrid.llm import _post
    url = os.environ.get("WRITER_URL")
    if url:
        body = {"model": llm.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8, "seed": 0,
                "max_tokens": llm.max_tokens}
        return _post(url.rstrip("/") + "/chat/completions", body)["choices"][0]["message"]["content"]
    body = {"model": llm.model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": 0.8, "seed": 0, "num_predict": llm.max_tokens, "num_ctx": 4096}}
    return _post(llm.host + "/api/chat", body)["message"]["content"]


def answers():
    from hybrid.llm import OpenAILLM
    url = os.environ.get("ANSWER_URL")
    model = OpenAILLM(url, ANSWERER, system=BRIEF) if url else OllamaLLM(ANSWERER, system=BRIEF)
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
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool, open(AFILE, "a") as f:
        for n, (q, a) in enumerate(pool.map(one, todo)):
            if a:
                f.write(json.dumps({"instruction": q, "reply": a}) + "\n")
            if n % 500 == 0:
                f.flush(); print("answers: %d/%d" % (n, len(todo)), flush=True)


def embed_all():
    from embed import embed
    rows = [json.loads(line) for line in open(AFILE)]
    np.save(os.path.join(OUT, TAG + "_q_emb.npy"), embed([r["instruction"] for r in rows]))
    np.save(os.path.join(OUT, TAG + "_a_emb.npy"), embed([r["reply"] for r in rows]))
    print("embedded", len(rows))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "questions2":
        questions2(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif step == "questions":
        questions(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif step == "answers":
        answers()
    else:
        embed_all()
