"""PyBot-70: about 70% of turns answered in Python, the rest by a small model.

Each turn:
  1. Three Python replies, none generated: PyBot's own, the PyBot knowledge entry closest in
     meaning to the question (first two sentences), and the logged answer to the closest logged
     question.
  2. One scorer (gradient-boosted trees over about 24 signals) predicts how each would be judged
     and picks the best.
  3. A router predicts how the model would be judged on this question. If the model is expected to
     beat the best Python reply by more than a threshold, the turn goes to the model.

The scorer, router and threshold were fitted on held-out prompts (bench/export_pybot70.py) and
are frozen in hybrid/pybot70_assets/. Embeddings come from all-MiniLM through Ollama; relevance
from a small cross-encoder. Same features as bench/chooser.py, so live choices match the benchmark.

    from hybrid.pybot70 import PyBot70
    bot = PyBot70()
    out = bot.turn("how long should I boil an egg?", memory={})
"""
import json
import os
import pickle
import re
import time
import urllib.request

import numpy as np

from .features import ABSTAIN, HANDS_BACK, content_words
from .llm import BRIEF, OllamaLLM

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pybot70_assets")
WANTS = ("options", "yesno", "steps", "meaning", "compare", "pick", "amount", "reason")
LABEL = {"pybot": "PyBot", "entry-emb-trim2": "PyBot knowledge entry", "recall-emb": "logged answer"}
MAX_HISTORY_TURNS = 4


def trim(text, n=2):
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:n])


def embed(texts, model="all-minilm", host="http://localhost:11434"):
    req = urllib.request.Request(host.rstrip("/") + "/api/embed", data=json.dumps({"model": model, "input": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    v = np.array(json.load(urllib.request.urlopen(req, timeout=60))["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def inside(text, trace):
    """How PyBot found its answer: does the knowledge entry it used cover the question's words."""
    q = content_words(text)
    keys = {k.lower() for k in trace.get("keys") or []}
    name = content_words((trace.get("entry_id") or "").replace("-", " "))
    hit = q & content_words(" ".join(keys))
    return {"has_entry": 1.0 if trace.get("entry_id") else 0.0,
            "key_cover": len(hit) / len(q) if q else 0.0,
            "name_hit": len(q & name) / len(name) if name else 0.0}


def _trace(memory, before):
    last = memory.get("last_answer") or {}
    if last == before:                     # no knowledge entry was used on this turn
        last = {}
    return {"entry_id": last.get("id"), "keys": [str(k) for k in (last.get("keys") or [])][:60]}


class PyBot70:
    def __init__(self, llm=None, runtime=None, ollama="http://localhost:11434"):
        from sentence_transformers import CrossEncoder
        if runtime is None:
            from bot import runtime
        self.runtime = runtime
        self.ollama = ollama
        self.llm = llm if llm is not None else OllamaLLM("smollm2:360m", ollama, system=BRIEF)
        with open(os.path.join(ASSETS, "models.pkl"), "rb") as f:
            m = pickle.load(f)
        self.scorer, self.pca, self.ridge = m["scorer"], m["pca"], m["ridge"]
        self.llm_train, self.threshold, self.model_share = m["llm_train"], m["threshold"], m["model_share"]
        self.sources, self.all_sources = m["sources"], m["all_sources"]
        self.log_q = np.load(os.path.join(ASSETS, "log_q.npy"))
        self.log_ref = np.load(os.path.join(ASSETS, "log_ref.npy"))
        self.log_answers = json.load(open(os.path.join(ASSETS, "log_answers.json")))
        self.llm_E = np.load(os.path.join(ASSETS, "llm_E.npy"))
        self.entry_emb = np.load(os.path.join(ASSETS, "entry_emb.npy"))
        ents = json.load(open(os.path.join(ASSETS, "entries.json")))
        self.entry_ids, self.entry_text = ents["ids"], ents["text"]
        self.ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    # --- the Python side -------------------------------------------------------------------
    def python_candidates(self, text, memory):
        before = dict(memory.get("last_answer") or {})
        try:
            result = self.runtime.turn(text, memory)
            reply = " ".join(str(result["reply"]).split()) or "Hello!"
            module = result.get("module")
        except Exception:  # noqa: BLE001 - a turn never fails
            reply, module = "Hello!", None
        trace = _trace(memory, before)
        q, code_emb = embed([text, reply], host=self.ollama)
        s = self.entry_emb @ q
        j = int(s.argmax())
        sims = self.log_q @ q
        d = int(sims.argmax())
        return q, sims, [
            {"source": "pybot", "reply": reply, "module": module, "emb": code_emb, "sim": 0.0, "doc": None, "trace": trace},
            {"source": "entry-emb-trim2", "reply": trim(self.entry_text[j]), "entry": self.entry_ids[j],
             "emb": self.entry_emb[j], "sim": float(s[j]), "doc": None},
            {"source": "recall-emb", "reply": self.log_answers[d], "emb": self.log_ref[d], "sim": float(sims[d]), "doc": d}]

    def features(self, text, q, sims, cands):
        top = np.argsort(-sims)[:6]
        embs = np.array([c["emb"] for c in cands])
        pair = embs @ embs.T
        np.fill_diagonal(pair, -1)
        ce = self.ce.predict([(text, c["reply"]) for c in cands])
        qw = content_words(text)
        X = []
        for j, c in enumerate(cands):
            near = [t for t in top if t != c["doc"]][:5]
            agree = self.log_ref[near] @ c["emb"]
            w = np.maximum(sims[near], 0)
            rw = content_words(c["reply"])
            ins = inside(text, c["trace"]) if c["source"] == "pybot" else {}
            f = [float(q @ c["emb"]), float(ce[j]), float(agree[0]), float(agree.max()), float((agree * w).sum() / (w.sum() + 1e-9)),
                 float(sims[top[0]]), float(sims[top[:5]].mean()), c["sim"],
                 float(pair[j].max()), float(np.sort(pair[j])[-2]),
                 len(c["reply"]) / 600.0, len(qw & rw) / len(qw) if qw else 0.0,
                 1.0 if ABSTAIN.search(c["reply"]) else 0.0, 1.0 if HANDS_BACK.search(c["reply"]) else 0.0,
                 1.0 if c["reply"].rstrip().endswith("?") else 0.0,
                 1.0 if c["source"] == "pybot" and c.get("module") == "answers" else 0.0,
                 ins.get("key_cover", 0.0), ins.get("name_hit", 0.0), ins.get("has_entry", 0.0)]
            f += [1.0 if c["source"] == t else 0.0 for t in self.all_sources]
            X.append(f)
        return np.array(X)

    def expected_model(self, q):
        S = self.llm_E @ q
        top = np.argsort(-S)[:8]
        w = np.maximum(S[top], 0) ** 4
        hist = (self.llm_train[top] * w).sum() / (w.sum() + 1e-9)
        return float(self.ridge.predict(np.hstack([self.pca.transform(q[None]), [[hist]]]))[0])

    # --- one turn ----------------------------------------------------------------------------
    def decide(self, text, memory):
        """Python side and routing decision, no model call."""
        t0 = time.perf_counter()
        q, sims, cands = self.python_candidates(text, memory)
        pred = self.scorer.predict(self.features(text, q, sims, cands))
        best = int(pred.argmax())
        p_llm = self.expected_model(q)
        margin = p_llm - float(pred[best])
        out = {"reply": cands[best]["reply"], "tier": "python", "source": cands[best]["source"],
               "source_label": LABEL[cands[best]["source"]], "module": cands[0]["module"],
               "entry": cands[best].get("entry"), "defer": margin > self.threshold,
               "p_python": float(pred[best]), "p_model": p_llm, "margin": margin, "threshold": self.threshold,
               "candidates": [{"source": LABEL[c["source"]], "reply": c["reply"], "predicted": float(p)}
                              for c, p in zip(cands, pred)],
               "python_ms": (time.perf_counter() - t0) * 1000, "model_ms": 0.0, "model": None}
        return out

    def turn(self, text, memory):
        return self.answer(self.decide(text, memory), text, memory)

    def answer(self, out, text, memory):
        """Call the model when decide() deferred; keep the conversation record true to the reply."""
        if out["defer"] and self.llm is not None:
            t0 = time.perf_counter()
            try:
                reply = self.llm.chat(_history(memory, text))
            except Exception:  # noqa: BLE001 - the model is optional: keep the Python reply
                reply = ""
            out["model_ms"] = (time.perf_counter() - t0) * 1000
            if reply:
                out["python_reply"] = out["reply"]
                out.update(reply=reply, tier="model", model=getattr(self.llm, "name", "model"))
        turns = memory.get("turns")
        if isinstance(turns, list) and turns:        # keep the thread true to what was said
            turns[-1] = dict(turns[-1], bot=out["reply"])
        return out


def _history(memory, text):
    messages = []
    turns = memory.get("turns") if isinstance(memory, dict) else None
    for t in (turns[:-1] if isinstance(turns, list) else [])[-MAX_HISTORY_TURNS:]:
        messages += [{"role": "user", "content": str(t.get("user", ""))}, {"role": "assistant", "content": str(t.get("bot", ""))}]
    return messages + [{"role": "user", "content": text}]
