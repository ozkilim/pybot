"""Distill the judge into a small cross-encoder (MiniLM-L6, 22M parameters).

Training pairs: every (held-out question, reply, judged score) we have, whatever wrote the reply:
PyBot, its knowledge entries, logged answers, SmolLM2-360M/1.7B, Gemma-1B. No benchmark prompt is used.
The cross-encoder reads only the question and the reply (no reference answer), so it can run live.

  cv    5-fold, grouped by prompt: every held-out pair gets a score from a model that never saw
        its prompt (bench/out/judge_ce_oof.json), plus AUROC/correlation against the judge
  full  fit on all held-out pairs, save to bench/out/judge_ce/

    python bench/distill_judge.py cv
    python bench/distill_judge.py full
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "out")
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EPOCHS, BATCH, LR, MAXLEN = 3, 32, 3e-5, 256
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# (reply file in bench/out, judged file) - judged rows hold only id and score
PAIRS = [
    ("pybot_heldout.jsonl", "poc1198/pybot_heldout.judged.jsonl"),
    ("smollm2-360m-brief_heldout.jsonl", "poc1198/smollm2-360m-brief_heldout.judged.jsonl"),
    ("entry-emb-trim2_heldoutall.jsonl", "cand/entry-emb-trim2_heldoutall.judged.jsonl"),
    ("recall-emb_heldoutall.jsonl", "cand/recall-emb_heldoutall.judged.jsonl"),
    ("entry-emb-trim2_heldout.jsonl", "cand/entry-emb-trim2_heldout.judged.jsonl"),
    ("recall-emb_heldout.jsonl", "cand/recall-emb_heldout.judged.jsonl"),
    ("smollm2-1.7b-brief_heldout.jsonl", "poc/smollm2-1.7b-brief_heldout.judged.jsonl"),
    ("gemma3-1b-brief_heldout.jsonl", "poc/gemma3-1b-brief_heldout.judged.jsonl"),
    ("pybot-trim2_heldout.jsonl", "poc/pybot-trim2_heldout.judged.jsonl"),
    ("recall_heldout.jsonl", "poc/recall_heldout.judged.jsonl"),
]


def key(pid, reply):
    return pid + "|" + hashlib.sha1(reply.encode()).hexdigest()[:16]


def load():
    q = {json.loads(l)["id"]: json.loads(l)["instruction"] for l in open(os.path.join(ROOT, "bench", "data", "heldout.jsonl"))}
    seen, data = set(), []
    for rep, jud in PAIRS:
        rp, jp = os.path.join(OUT, rep), os.path.join(ROOT, "bench", jud)
        if not (os.path.exists(rp) and os.path.exists(jp)):
            continue
        R = {json.loads(l)["id"]: json.loads(l)["reply"] for l in open(rp)}
        for line in open(jp):
            j = json.loads(line)
            if j["id"] not in R or j["id"] not in q or j.get("score") is None:
                continue
            k = key(j["id"], R[j["id"]])
            if k in seen:
                continue
            seen.add(k)
            data.append({"id": j["id"], "q": q[j["id"]], "reply": R[j["id"]], "score": float(j["score"]), "src": rep.split("_")[0], "key": k})
    return data


def fit(train, seed=0):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        for b in np.array_split(rng.permutation(len(train)), max(1, len(train) // BATCH)):
            enc = tok([train[i]["q"] for i in b], [train[i]["reply"] for i in b], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(DEVICE)
            y = torch.tensor([train[i]["score"] for i in b], dtype=torch.float32, device=DEVICE)
            loss = lossf(model(**enc).logits.squeeze(-1), y)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    return tok, model


@torch.no_grad()
def predict(tok, model, qs, replies, batch=64):
    out = []
    for i in range(0, len(qs), batch):
        enc = tok(qs[i:i + batch], replies[i:i + batch], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(model.device)
        out.extend(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy().tolist())
    return np.array(out)


def main():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    data = load()
    print("%d judged held-out pairs over %d prompts" % (len(data), len({d["id"] for d in data})), flush=True)
    if mode == "full":
        tok, model = fit(data)
        dst = os.path.join(OUT, "judge_ce")
        model.save_pretrained(dst); tok.save_pretrained(dst)
        print("saved", dst)
        return
    groups = np.array([d["id"] for d in data])
    oof = np.zeros(len(data))
    for f, (tr, te) in enumerate(GroupKFold(5).split(data, groups=groups)):
        tok, model = fit([data[i] for i in tr])
        oof[te] = predict(tok, model, [data[i]["q"] for i in te], [data[i]["reply"] for i in te])
        print("fold %d done" % f, flush=True)
    y = np.array([d["score"] for d in data])
    print("out-of-fold: AUROC (score>0.5) %.3f, Pearson %.3f" % (roc_auc_score(y > 0.5, oof), np.corrcoef(y, oof)[0, 1]))
    for s in sorted({d["src"] for d in data}):
        m = np.array([d["src"] == s for d in data])
        if len(set(y[m] > 0.5)) > 1:
            print("  %-20s n=%4d AUROC %.3f" % (s, m.sum(), roc_auc_score(y[m] > 0.5, oof[m])))
    json.dump({d["key"]: float(p) for d, p in zip(data, oof)}, open(os.path.join(OUT, "judge_ce_oof.json"), "w"))


if __name__ == "__main__":
    main()
