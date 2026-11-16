"""Answer each prompt once with a small model served by Ollama, as a single user turn.

    python bench/run_llm.py smollm2:360m bench/data/smoltalkbench.jsonl bench/out/smollm2-360m_bench.jsonl
    python bench/run_llm.py smollm2:360m IN OUT --system "Reply in one to three short sentences."
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def chat(model, text, system=None, host="http://localhost:11434", max_tokens=256):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens}}
    request = urllib.request.Request(host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
    except urllib.error.HTTPError:
        body.pop("think")  # models without a thinking mode reject the flag on some versions
        request = urllib.request.Request(host + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
    return data["message"]["content"], data.get("eval_count", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--system", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.src)]
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    done = {json.loads(line)["id"] for line in open(args.dst)} if os.path.exists(args.dst) else set()
    rows = [row for row in rows if row["id"] not in done]   # resume: keep what is already answered

    def work(row):
        start = time.perf_counter()
        text, tokens = chat(args.model, row["instruction"], args.system)
        # Same normalisation as PyBot's interface: one non-empty line.
        return {"id": row["id"], "instruction": row["instruction"], "reply": " ".join(text.split()) or "Hello!",
                "tokens": tokens, "seconds": time.perf_counter() - start}

    with ThreadPoolExecutor(args.workers) as pool, open(args.dst, "a") as out:
        for i, rec in enumerate(pool.map(work, rows)):
            out.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                print(i + 1, "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
