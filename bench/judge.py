"""SmolTalkBench scoring: AlpacaEval 2.0 weighted annotator + WildBench length penalty.

Follows the dataset card (https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench):
  * the judge sees the reply and the reference under AlpacaEval's `alpaca_eval_clf` prompt;
    which one comes first is fixed per prompt;
  * a sample's score is the judge's probability that the reply is better, read from the
    top-5 logprobs of its single output token;
  * a reply more than 500 characters longer than its reference counts as a tie at most.

The card's judge is deepseek-ai/DeepSeek-V4.1-Flash. Any OpenAI-compatible endpoint that returns
logprobs can stand in (--base-url / --model); every arm must then be scored by that same judge.

    python bench/judge.py REFS.jsonl REPLIES.jsonl OUT.jsonl [--base-url ...] [--model ...]

REPLIES rows need {"id", "reply"}. Judgments are cached by (instruction, reply, reference), so
re-scoring a system that shares replies with another costs nothing.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = open(os.path.join(HERE, "prompts", "alpaca_eval_clf.txt")).read()
LENGTH_MARGIN = 500


def _messages(instruction, output_1, output_2):
    text = (TEMPLATE.replace("{instruction}", instruction)
            .replace("{output_1}", output_1).replace("{output_2}", output_2))
    parts = re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", text, flags=re.S)
    return [{"role": role, "content": content.strip()} for role, content in parts]


def switched(instruction):
    """Whether the reference goes first. Fixed per prompt, so every system sees the same order."""
    return hashlib.sha256(("is_switched_outputs" + instruction).encode()).digest()[0] % 2 == 1


def _key(instruction, reply, reference, model):
    return hashlib.sha256(json.dumps([instruction, reply, reference, model]).encode()).hexdigest()


def judge_one(instruction, reply, reference, base_url, model):
    """P(reply better than reference), or None when the judge names neither identifier."""
    flip = switched(instruction)
    first, second = (reference, reply) if flip else (reply, reference)
    mine = "M" if flip else "m"
    body = {"model": model, "messages": _messages(instruction, first, second), "max_tokens": 1,
            "temperature": 0, "logprobs": True, "top_logprobs": 5}
    if os.environ.get("JUDGE_NO_THINK", "1") == "1":  # llama.cpp: answer in one token, no reasoning
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + os.environ.get("JUDGE_API_KEY", "x")})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    mass = {"m": 0.0, "M": 0.0}
    for entry in top:
        token = entry["token"].strip()
        if token in mass:
            mass[token] += math.exp(entry["logprob"])
    total = mass["m"] + mass["M"]
    return mass[mine] / total if total > 0 else None


def penalised(score, reply, reference):
    if score is None:
        return 0.0
    return min(score, 0.5) if len(reply) > len(reference) + LENGTH_MARGIN else score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("refs")
    parser.add_argument("replies")
    parser.add_argument("out")
    parser.add_argument("--base-url", default="http://localhost:8091/v1")
    parser.add_argument("--model", default="gemma-3-12b-it-Q4_K_M")
    parser.add_argument("--cache", default=os.path.join(HERE, "out", "judge_cache.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    refs = {r["id"]: r for r in map(json.loads, open(args.refs))}
    replies = [r for r in map(json.loads, open(args.replies)) if r["id"] in refs]
    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache):
            entry = json.loads(line)
            cache[entry["key"]] = entry["score"]
    cache_file = open(args.cache, "a")

    def work(row):
        ref = refs[row["id"]]
        key = _key(ref["instruction"], row["reply"], ref["output"], args.model)
        if key not in cache:
            score = judge_one(ref["instruction"], row["reply"], ref["output"], args.base_url, args.model)
            cache[key] = score
            cache_file.write(json.dumps({"key": key, "score": score}) + "\n")
            cache_file.flush()
        raw = cache[key]
        return {"id": row["id"], "raw": raw, "score": penalised(raw, row["reply"], ref["output"])}

    done = []
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "w") as out:
        for i, rec in enumerate(pool.map(work, replies)):
            out.write(json.dumps(rec) + "\n")
            done.append(rec["score"])
            if (i + 1) % 100 == 0:
                print(i + 1, "/", len(replies), "win rate so far %.1f%%" % (100 * sum(done) / len(done)),
                      file=sys.stderr, flush=True)
    print("win rate %.2f%% over %d" % (100 * sum(done) / len(done), len(done)))


if __name__ == "__main__":
    main()
