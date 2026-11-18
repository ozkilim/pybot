"""Chooser: several Python-side replies per turn, a scorer picks one, and the turn goes to the
model only when even the best Python reply is expected to lose to it.

Python-side candidates (none of them generates text):
  pybot             PyBot's own reply
  entry-emb-trim2   the PyBot knowledge entry closest in meaning to the question (first two sentences)
  entry-ce-trim2    same, re-ranked among the 20 closest by a small cross-encoder
  recall-emb        the logged answer to the closest logged question
  recall-ce         same, re-ranked among the 10 closest by the cross-encoder

One scorer, shared by all candidates, predicts the judged score of a (question, reply) pair.

    python bench/chooser.py cv      # grouped cross-validation on held-out prompts
    python bench/chooser.py bench   # fit on held-out, score the 100 benchmark prompts once
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench")); sys.path.insert(0, ROOT)
from hybrid.features import ABSTAIN, HANDS_BACK, content_words  # noqa: E402
from signals import Log, OUT, SHARES, by_id, curve, inside, rows  # noqa: E402

SOURCES = ("pybot", "entry-emb-trim2", "entry-ce-trim2", "recall-emb", "recall-ce", "crystal", "crystal12", "crystal12-sts", "crystal12-k2", "crystal12-k3")
CAND = os.path.join(ROOT, "bench", "cand")
USE = [s for s in os.environ.get("SOURCES", "pybot,entry-emb-trim2,recall-emb").split(",") if s]
EXTRA = os.environ.get("EXTRA", "1") == "1"   # also train the scorer on the small model's judged replies
LLM_SRC = "smollm2-360m-brief"
FULL = os.environ.get("BENCH", "100") == "all"   # BENCH=all: score all 1,000 benchmark prompts (judged in bench/full)
BTAG = "benchall" if FULL else "bench"
JCE = os.environ.get("JCE", "0") == "1"          # add the distilled judge (bench/distill_judge.py) as a feature
REL = os.environ.get("REL", "0") == "1"          # add within-prompt relative features
REL_COLS = 12


_CA = {}


def crystal_emb(tag="crystal"):
    if tag not in _CA:
        _CA[tag] = np.load(os.path.join(OUT, tag + "_a_emb.npy"))
    return _CA[tag]


def judge_ce_scores(rows_, held):
    """Distilled-judge score per (question, reply). Held-out: out-of-fold predictions; benchmark: the full model."""
    import hashlib
    if held:
        oof = json.load(open(os.path.join(OUT, "judge_ce_oof.json")))
        return [oof.get(r["id"] + "|" + hashlib.sha1(r["reply"].encode()).hexdigest()[:16], np.nan) for r in rows_]
    from distill_judge import predict
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = os.path.join(OUT, "judge_ce")
    tok, model = AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval()
    return list(predict(tok, model, [r["instruction"] for r in rows_], [r["reply"] for r in rows_]))


def boost():
    return HistGradientBoostingRegressor(max_depth=4, max_iter=250, learning_rate=0.04, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


class Data:
    """Everything about one split: candidates, their embeddings, features and (where judged) scores."""

    def __init__(self, split, log, ce):
        self.log = log
        held = split == "heldout"
        base = rows(os.path.join(ROOT, "bench", "data", "heldout.jsonl" if held else ("smoltalkbench.jsonl" if FULL else "bench100.jsonl")))
        self.ids = [r["id"] for r in base]
        trace = by_id(os.path.join(OUT, "pybot_heldout_trace.jsonl" if held else ("pybot_benchall_trace.jsonl" if FULL else "pybot_bench100_trace.jsonl")))
        if held:
            self.q = log.q[[log.pos[i] for i in self.ids]]
            code_emb = log.code[[log.pos[i] for i in self.ids]]
        else:
            allb = [r["id"] for r in rows(os.path.join(OUT, "pybot_bench.jsonl"))]
            ix = [allb.index(i) for i in self.ids]
            self.q = np.load(os.path.join(OUT, "emb_bench_q.npy"))[ix]
            code_emb = np.load(os.path.join(OUT, "emb_bench_code.npy"))[ix]
        ent_ids = json.load(open(os.path.join(OUT, "emb_entries_ids.json")))
        ent_pos = {e: k for k, e in enumerate(ent_ids)}
        EE = np.load(os.path.join(OUT, "emb_entries.npy"))
        pj = by_id(os.path.join(ROOT, "bench", "poc1198" if held else ("full" if FULL else "poc"), "pybot_%s.judged.jsonl" % split))
        self.cands = {}                                   # source -> list of dicts aligned with self.ids
        for s in USE:
            if s == "pybot":
                rs = [dict(trace[i], emb=code_emb[k], sim=0.0, doc=None) for k, i in enumerate(self.ids)]
                sc = pj
            else:
                tag = os.environ.get("RECALL_LOG") if (s == "recall-emb" and not held) else None   # log-size ablation
                name = s + ("-log" + tag if tag else "")
                src = by_id(os.path.join(OUT, "%s_%s.jsonl" % (name, "heldoutall" if held else BTAG)))
                rs = []
                own = os.path.join(OUT, "%s_%s_emb.npy" % (name, split))      # replies from outside the held-out log
                own_emb = dict(zip(json.load(open(own.replace(".npy", "_ids.json"))), np.load(own))) if os.path.exists(own) else {}
                for i in self.ids:
                    r = dict(src[i])
                    r["emb"] = EE[ent_pos[r["entry"]]] if "entry" in r else (log.ref[r["doc"]] if r.get("doc") is not None else None)
                    if "crystal" in r:                                    # cached model answer (bench/crystallize.py)
                        r["emb"] = crystal_emb(r.get("cache", "crystal"))[r["crystal"]]
                    if r["emb"] is None:
                        r["emb"] = own_emb[i]
                    r["sim"], r["doc"] = r["similarity"], r.get("doc")
                    rs.append(r)
                path = os.path.join(CAND, "%s_heldoutall.judged.jsonl" % name) if held else \
                    os.path.join(CAND, "%s_%s.judged.jsonl" % (name, split)) if not FULL else \
                    os.path.join(ROOT, "bench", "full", "%s_benchall.judged.jsonl" % name)
                sc = by_id(path) if os.path.exists(path) else {}
            for r, i in zip(rs, self.ids):
                r["score"] = sc[i]["score"] if i in sc else np.nan
                r["source"] = s
            self.cands[s] = rs
        self.all = list(USE)
        if held and EXTRA:                                 # training-only rows: never offered as a candidate
            L = by_id(os.path.join(OUT, LLM_SRC + "_heldout.jsonl")); LJ = by_id(os.path.join(ROOT, "bench", "poc1198", LLM_SRC + "_heldout.judged.jsonl"))
            LE = np.load(os.path.join(OUT, "emb_heldout_360m.npy"))
            self.cands["llm"] = [dict(L[i], emb=LE[log.pos[i]], sim=0.0, doc=None, source="llm",
                                      score=LJ[i]["score"] if i in LJ else np.nan) for i in self.ids]
            self.all.append("llm")
        # cross-encoder relevance for every (question, candidate reply)
        pairs = [(r["instruction"], r["reply"]) for s in self.all for r in self.cands[s]]
        vals = ce.predict(pairs, batch_size=64)
        k = 0
        for s in self.all:
            for r in self.cands[s]:
                r["ce"] = float(vals[k]); k += 1
        if JCE:
            allrows = [r for s in self.all for r in self.cands[s]]
            for r, v in zip(allrows, judge_ce_scores(allrows, held)):
                r["jce"] = float(v)
        self.held = held

    def features(self):
        """X (rows = prompt x source), scores, prompt index, source index."""
        X, y, g, src = [], [], [], []
        log = self.log
        for k, pid in enumerate(self.ids):
            q = self.q[k]
            sims = log.q @ q
            if self.held:
                sims[log.pos[pid]] = -1
            top = np.argsort(-sims)[:6]
            embs = np.array([self.cands[s][k]["emb"] for s in USE])
            pair = embs @ embs.T
            np.fill_diagonal(pair, -1)
            qw = content_words(self.cands[USE[0]][k]["instruction"])
            for j, s in enumerate(self.all):
                r = self.cands[s][k]
                extra = j >= len(USE)
                pj = np.sort(embs @ r["emb"])[::-1] if extra else None
                near = [t for t in top if t != r["doc"]][:5]      # do not let a logged answer agree with itself
                agree = log.ref[near] @ r["emb"]
                w = np.maximum(sims[near], 0)
                rw = content_words(r["reply"])
                ins = inside(r) if s == "pybot" else {}
                f = [float(q @ r["emb"]), r["ce"], float(agree[0]), float(agree.max()), float((agree * w).sum() / (w.sum() + 1e-9)),
                     float(sims[top[0]]), float(sims[top[:5]].mean()), r["sim"],
                     float(pj[0]) if extra else float(pair[j].max()),                       # agreement with the other candidates
                     float(pj[1]) if extra else (float(np.sort(pair[j])[-2]) if len(USE) > 2 else 0.0),
                     len(r["reply"]) / 600.0, len(qw & rw) / len(qw) if qw else 0.0,
                     1.0 if ABSTAIN.search(r["reply"]) else 0.0, 1.0 if HANDS_BACK.search(r["reply"]) else 0.0,
                     1.0 if r["reply"].rstrip().endswith("?") else 0.0,
                     1.0 if s == "pybot" and r.get("module") == "answers" else 0.0,
                     ins.get("key_cover", 0.0), ins.get("name_hit", 0.0), ins.get("has_entry", 0.0)]
                f += [r.get("qq_sts", np.nan), r.get("qq_dup", np.nan)]      # is the matched question the same question
                if JCE:
                    f.append(r["jce"])
                f += [1.0 if s == t else 0.0 for t in SOURCES]
                X.append(f); y.append(r["score"]); g.append(k); src.append(j)
        X, g, src = np.array(X), np.array(g), np.array(src)
        if REL:                                            # each candidate relative to the others for the same question
            cont = X[:, :REL_COLS]
            sel = src < len(USE)
            mean = np.zeros((g.max() + 1, REL_COLS)); cnt = np.zeros(g.max() + 1)
            np.add.at(mean, g[sel], np.nan_to_num(cont[sel])); np.add.at(cnt, g[sel], 1)
            mx = np.full((g.max() + 1, REL_COLS), -np.inf)
            np.maximum.at(mx, g[sel], np.nan_to_num(cont[sel], nan=-np.inf))
            X = np.hstack([X, cont - (mean / cnt[:, None])[g], cont - mx[g]])
        return X, np.array(y), g, src


def llm_side(log, q_train_pos, llm_train, q_test, loo_pos=None):
    """Expected model score for each test question: ridge on question embedding PCA + neighbours' scores."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    E = log.q[q_train_pos]
    pca = PCA(12, random_state=0).fit(E)

    def hist(Q, exclude_self):
        S = Q @ E.T
        out = np.zeros(len(Q))
        for i in range(len(Q)):
            if exclude_self:
                S[i, i] = -1
            top = np.argsort(-S[i])[:8]
            w = np.maximum(S[i, top], 0) ** 4
            out[i] = (llm_train[top] * w).sum() / (w.sum() + 1e-9)
        return out
    m = Ridge(alpha=30.0).fit(np.hstack([pca.transform(E), hist(E, True)[:, None]]), llm_train)
    return m.predict(np.hstack([pca.transform(q_test), hist(q_test, False)[:, None]]))


def pick(pred, g, src, n):
    """Per prompt: index of the best-scored candidate row, and its predicted score."""
    best_row, best = np.zeros(n, int), np.full(n, -9.0)
    for row in range(len(pred)):
        if pred[row] > best[g[row]]:
            best[g[row]], best_row[g[row]] = pred[row], row
    return best_row, best


def two_stage(rank, chosen, best, llm, pred_llm_reply, margin=0.0):
    """Defer by rank; on deferred turns keep the model's reply only if the scorer rates it above
    the best Python reply (plus a margin). Returns the win rate at each share of model calls."""
    order = np.argsort(-rank, kind="stable")
    out = []
    for share in SHARES:
        d = np.zeros(len(chosen), bool)
        d[order[:int(round(share * len(chosen)))]] = True
        use_llm = d & (pred_llm_reply >= best + margin)
        out.append(100 * np.where(use_llm, llm, chosen).mean())
    return out


def report(title, y, g, src, pred, p_llm, llm, n, pred_llm_reply=None):
    Y = np.full((n, len(USE)), np.nan)
    Y[g, src] = y
    best_row, best = pick(pred, g, src, n)
    chosen = y[best_row]
    print("\n" + title)
    for j, s in enumerate(USE):
        print("  %-18s alone %5.1f%%   chosen on %4.1f%% of prompts" % (s, 100 * np.nanmean(Y[:, j]), 100 * np.mean(src[best_row] == j)))
    print("  chooser, Python only (no model): %.1f%%   | hindsight best Python candidate: %.1f%%"
          % (100 * chosen.mean(), 100 * np.nanmax(Y, 1).mean()))
    print("  model alone: %.1f%%   | hindsight best of everything: %.1f%%" % (100 * llm.mean(), 100 * np.maximum(np.nanmax(Y, 1), llm).mean()))
    print("  %-36s %s" % ("model share:", "  ".join("%4.0f%%" % (100 * s) for s in SHARES)))
    print("  %-36s %s" % ("chooser + deferral (model - best)", "  ".join("%5.1f" % v for v in curve(p_llm - best, chosen, llm))))
    print("  %-36s %s" % ("chooser + deferral (best only)", "  ".join("%5.1f" % v for v in curve(-best, chosen, llm))))
    if pred_llm_reply is not None:
        for mg in (0.0, -0.05, -0.1):
            print("  %-36s %s" % ("two-stage, margin %+.2f" % mg, "  ".join("%5.1f" % v for v in two_stage(p_llm - best, chosen, best, llm, pred_llm_reply, mg))))
        allc = two_stage(np.ones(n), chosen, best, llm, pred_llm_reply, 0.0)[-1]
        print("  model on every turn, scorer picks between its reply and Python: %.1f%%" % (100 * np.where(pred_llm_reply >= best, llm, chosen).mean()))
    rank = p_llm - best
    order = np.argsort(-rank, kind="stable")
    fine = [100 * np.where(np.isin(np.arange(n), order[:int(round(f * n))]), llm, chosen).mean() for f in np.arange(0, 1.001, 0.01)]
    par = next((k for k, v in enumerate(fine) if v >= 100 * llm.mean()), None)
    print("  PyBot-90 (10%% model): %.1f%%   | model share needed to match the model alone: %s"
          % (fine[10], "%d%%" % par if par is not None else "never"))
    print("  %-36s %s" % ("oracle deferral over chooser", "  ".join("%5.1f" % v for v in curve(llm - chosen, chosen, llm))))
    return chosen, best


def main():
    from sentence_transformers import CrossEncoder
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    llm_run = os.environ.get("LLM_RUN", "smollm2-360m-brief")
    log = Log()
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    H = Data("heldout", log, ce)
    X, y, g, src = H.features()
    M = by_id(os.path.join(ROOT, "bench", "poc1198", llm_run + "_heldout.judged.jsonl"))
    llm_all_ids = [i for i in log.pos if i in M]
    llm_h = np.array([M[i]["score"] for i in H.ids])
    sel = src < len(USE)                                   # rows that may be chosen at run time
    ok = ~np.isnan(y)
    full = np.array([ok[(g == k) & sel].all() for k in range(len(H.ids))])     # prompts with every candidate judged
    print("held-out prompts: %d, with all %d candidates judged: %d" % (len(H.ids), len(USE), full.sum()))
    if mode == "cv":
        pred, p_llm = np.zeros(len(y)), np.zeros(len(H.ids))
        hpos = np.array([log.pos[i] for i in H.ids])
        for tr, te in GroupKFold(5).split(X, y, g):
            trp, tep = np.unique(g[tr]), np.unique(g[te])
            m = boost().fit(X[tr][ok[tr]], y[tr][ok[tr]])
            pred[te] = m.predict(X[te])
            # the model side may also learn from held-out prompts outside the 600, but never from the test fold
            test_ids = {H.ids[k] for k in tep}
            tr_ids = [i for i in llm_all_ids if i not in test_ids]
            p_llm[tep] = llm_side(log, np.array([log.pos[i] for i in tr_ids]), np.array([M[i]["score"] for i in tr_ids]), log.q[hpos[tep]])
        keep = np.isin(g, np.where(full)[0]) & sel
        remap = {k: n for n, k in enumerate(np.where(full)[0])}
        g2 = np.array([remap[k] for k in g[keep]])
        plr = None
        if EXTRA:
            plr = np.full(len(H.ids), np.nan)
            plr[g[src == len(USE)]] = pred[src == len(USE)]
            plr = plr[full]
        report("held-out, grouped 5-fold", y[keep], g2, src[keep], pred[keep], p_llm[full], llm_h[full], int(full.sum()), plr)
    else:
        B = Data("bench", log, ce)
        Xb, yb, gb, sb = B.features()
        m = boost().fit(X[ok], y[ok])
        pred = m.predict(Xb)
        Mb = by_id(os.path.join(ROOT, "bench", "full" if FULL else "poc", llm_run + "_bench.judged.jsonl"))
        llm_b = np.array([Mb[i]["score"] for i in B.ids])
        p_llm = llm_side(log, np.array([log.pos[i] for i in llm_all_ids]), np.array([M[i]["score"] for i in llm_all_ids]), B.q)
        chosen, best = report("benchmark (%d prompts), fitted on held-out only" % len(B.ids), yb, gb, sb, pred, p_llm, llm_b, len(B.ids))
        rng = np.random.default_rng(0)
        order = np.argsort(-(p_llm - best), kind="stable")
        for share in (0.0, 0.2, 0.3, 0.4, 0.5):
            d = np.zeros(len(chosen), bool); d[order[:int(round(share * len(chosen)))]] = True
            mix = np.where(d, llm_b, chosen); diff = mix - llm_b
            bt = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(1)
            print("  model share %3.0f%%: hybrid %.1f%%, against model alone %+.1f (95%% CI %+.1f to %+.1f)"
                  % (100 * share, 100 * mix.mean(), 100 * diff.mean(), 100 * np.quantile(bt, .025), 100 * np.quantile(bt, .975)))
        json.dump({"ids": B.ids, "chosen_source": [USE[j] for j in sb[pick(pred, gb, sb, len(B.ids))[0]]],
                   "p_best": best.tolist(), "p_llm": p_llm.tolist()}, open(os.path.join(OUT, "chooser_%s.json" % BTAG), "w"))


if __name__ == "__main__":
    main()
