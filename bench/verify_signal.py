"""Experiment: a one-token check by the small model on PyBot's reply, as an extra gate signal.

    python bench/verify_signal.py smollm2:1.7b bench/out/pybot_heldout.jsonl bench/out/verify-1.7b_heldout.jsonl IDS.jsonl
"""
import json, math, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ('Question: {q}\n\nAnswer: {a}\n\nDoes the answer directly and helpfully address the question? '
          'Reply with one word: yes or no.')


def p_yes(model, q, a):
    body = {"model": model, "stream": False, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "user", "content": PROMPT.format(q=q, a=a)}],
            "options": {"temperature": 0, "num_predict": 1}}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=120))
    yes = no = 0.0
    for t in data["logprobs"][0]["top_logprobs"]:
        w = t["token"].strip().lower()
        if w.startswith("yes"): yes += math.exp(t["logprob"])
        elif w.startswith("no"): no += math.exp(t["logprob"])
    return yes / (yes + no) if yes + no > 0 else 0.5


model, src, dst, idfile = sys.argv[1:5]
ids = {json.loads(l)["id"] for l in open(idfile)}
rows = [r for r in map(json.loads, open(src)) if r["id"] in ids]
with ThreadPoolExecutor(8) as pool, open(dst, "w") as out:
    for r, p in zip(rows, pool.map(lambda r: p_yes(model, r["instruction"], r["reply"]), rows)):
        out.write(json.dumps({"id": r["id"], "p_yes": p}) + "\n")
print("done", len(rows))
