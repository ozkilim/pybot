# PyBot Hybrid: code first, a small model only when the code should not answer

A fork of [huggingface/pybot](https://github.com/huggingface/pybot). `bot/` is untouched. `hybrid/`
adds a cascade around it:

1. **Code tier.** PyBot proposes its reply, as before.
2. **Gate.** A 27-number formula (pure Python, weights in `hybrid/gate_weights.json`) reads what the
   code did: did it admit not knowing, is the reply about the question, is it a stock line, how many
   modules proposed something. It estimates the chance the code's reply is good and the chance a small
   model would do better.
3. **Model tier.** If the model is expected to do better by more than a margin, the turn goes to a
   small language model (SmolLM2 by default, through Ollama or any OpenAI-compatible endpoint).

The margin is set by one dial: the share of everyday turns the model may take (1%, 10%, 30%, ...).
Every reply carries its tier, module or model, the gate's numbers and its reasons.

## Headline: the full 1,000-prompt benchmark (`BENCH=all python bench/chooser.py bench`)

All 1,000 SmolTalkBench prompts, local Qwen3.6-35B-A3B judge (same recipe as the benchmark card,
different judge, so compare rows with each other, not with the published 26.7%). The chooser's scorer
and the deferral rule were fitted on held-out prompts only; the answer log it recalls from is the
1,198 held-out prompts only, so no benchmark answer is ever recalled.

| System (n=1,000) | Win rate | vs SmolLM2-360M alone (95% CI) | Turns sent to a model |
|---|---:|---|---:|
| PyBot, code only | 23.0% | | 0% |
| Chooser, Python only (PyBot + knowledge entry + logged answer) | 30.5% | −8.3 (−11.5 to −5.2) | 0% |
| Chooser + SmolLM2-360M | 37.4% | −1.4 (−4.3 to +1.5) | 20% |
| Chooser + SmolLM2-360M | 39.8% | +1.0 (−1.6 to +3.6) | 30% |
| Chooser + SmolLM2-360M | 41.8% | **+3.1 (+0.6 to +5.5)** | 40% |
| Chooser + SmolLM2-360M | 43.0% | **+4.2 (+2.0 to +6.6)** | 50% |
| Chooser + SmolLM2-360M | 44.4% | | 75% |
| SmolLM2-360M (brief prompt) alone | 38.8% | | 100% |

- With 70% of turns answered in Python, the hybrid matches the model on every turn (+1.0, CI
  includes zero). With 60% in Python it beats it, and the confidence interval excludes zero.
- Python alone goes from 23.0% to 30.5% just by choosing among three Python replies.
- Hindsight ceilings: best Python reply 36.5%; oracle deferral over the chooser 51.8% at 30% model
  share; best of everything 56.0%. The scorer leaves a lot on the table, mostly coin-flip judgements.
- Candidate alone: entry 19.7%, logged answer 12.9%; chosen on 32% and 25% of prompts.

### Pushing toward PyBot-90: a crystallized answer cache

Goal: match the model with as few model calls as possible. The limit was the Python replies, not the
picking: before this, even the best Python reply in hindsight could not match the model.

- **Distilled judge (no gain).** A 22M cross-encoder trained on 5,138 judged held-out replies
  predicts the judge well (AUROC 0.87 on PyBot's replies) but added nothing to the chooser.
- **Crystallized cache.** Offline, Qwen2.5-7B wrote 33,133 everyday questions from the held-out
  topics only (`bench/crystallize.py`; never the benchmark's prompts or topics), and SmolLM2-360M
  answered each exactly as at run time. At run time the cached answer to the closest cached question
  is a fourth Python candidate: a lookup, no model call. Building it took about 33,000 model calls, once.
  Closest-question similarity on the benchmark: median 0.77, 37% of prompts at 0.8 or more.

| Full benchmark (n=1,000) | 3 Python candidates | + answer cache |
|---|---:|---:|
| Python only | 30.5% | 31.6% |
| PyBot-90 (10% of turns to the model) | 34.4% | 35.2% |
| 30% to the model, vs model alone | +1.0 | +2.0 (CI −0.5 to +4.5) |
| 40% to the model, vs model alone | +3.1 | +3.4 (CI +1.1 to +5.8) |
| Model share needed to match the model alone | ~28% | **25%** |
| Best Python reply in hindsight | 36.5% | **41.2%** (model alone: 38.8%) |

The ceiling moved: with perfect picking, Python alone would now beat the model. The gap left is
picking (31.6% chosen vs 41.2% possible) and routing (oracle routing at 10% model share: 41.4%).
Judging moved to janos (2 × Quadro RTX 8000, same judge file, Pearson 0.99 with the Mac judge).

Better picking, tried next (`bench/crystal_more.py`), none of which moved the full benchmark:
- three more cached answers per turn (2nd/3rd nearest cached question, and the best paraphrase among
  the 20 nearest by a small STS cross-encoder): the hindsight ceiling rose to 44.7%, but the chooser
  picked no better (Python only 31.4%, PyBot-90 35.1%, parity at 24% model share). Cross-validation
  on held-out had shown +2 points; it did not transfer.
- question-to-question paraphrase scores (STS and Quora-duplicate cross-encoders) as features: no gain.
- each candidate's signals relative to the other candidates for the same question: no gain.
The picking problem is that the judge's preferences between plausible replies are hard to predict
from anything visible at run time; more options only help if the scorer can tell them apart.

## Status: proof of concept on a 10x smaller sample

Scored on **100 of SmolTalkBench's 1,000 prompts** (random, seed 0). The gate was fitted on up to
**600 held-out prompts** from the same source that are not in the benchmark. Same scoring recipe as
the benchmark card (AlpacaEval 2.0 weighted prompt, logprob score, WildBench length penalty), but
with a **local judge, Qwen3.6-35B-A3B**, not the card's DeepSeek-V4.1-Flash. So compare rows in this
table with each other, not with the published 26.7%.

| System | Win rate | 95% CI | Turns answered by a model |
|---|---:|---|---:|
| PyBot, code only | 17.3% | 10.8–24.4 | 0% |
| Hybrid → SmolLM2-360M | 17.6% | 11.0–24.5 | 2% |
| Hybrid → SmolLM2-360M | 21.0% | 14.3–28.1 | 13% |
| Hybrid → SmolLM2-360M | 27.2% | 19.9–34.6 | 34% |
| Hybrid → SmolLM2-360M | 34.2% | 26.7–42.0 | 52% |
| SmolLM2-360M alone | 35.2% | 28.4–41.9 | 100% |
| SmolLM2-360M alone, brief prompt | 38.8% | 30.9–46.6 | 100% |
| Gemma3-1B alone, brief prompt | 50.8% | 42.7–58.9 | 100% |
| SmolLM2-1.7B alone, brief prompt | 55.3% | 47.3–62.9 | 100% |

Full table: [bench/results/RESULTS.md](bench/results/RESULTS.md). Chart: `bench/results/tradeoff.png`.

## Update: a much better gate (experiment, not yet wired into `hybrid/`)

`bench/two_sided.py` predicts the code's score and the model's score separately and defers on the gap.
It is fitted on 600 held-out prompts and adds signals the first gate did not have, none of which
generate text:

- **PyBot's own record of how it answered** (`memory["last_answer"]`): which knowledge entry, how much
  of the question that entry's keywords cover, whether the entry's name is in the question.
- **Meaning** (all-MiniLM embeddings, 23M parameters): is the reply about the question, and does it
  agree with logged answers to the nearest logged questions. These two alone separate PyBot's good
  real answers from its bad ones with AUROC 0.79.
- **A model side**: question type, question embedding, and how the model scored on the nearest
  already-judged questions, to spot prompts the model will fail too (keeping the code there is free).

Out of fold on held-out prompts the code-side AUROC went from 0.69 to 0.885.

| Turns answered by the model | 30% | 40% | 50% | 60% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| Held-out, cross-validated (n=600) | 32.4% | 36.1% | 38.3% | 38.7% | 39.5% | 36.9% |
| Benchmark sample (n=100), first gate | 27.2% | 32.5% | 34.2% | – | 36.4% | 38.8% |
| Benchmark sample (n=100), new gate | 29.4% | 33.5% | 37.2% | 39.8% | 40.9% | 38.8% |

On the benchmark sample the hybrid now matches SmolLM2-360M alone with 40 to 50% of turns answered
by Python: 39.8% at 60% model share (+1.1 against the model alone, 95% CI −4.2 to +6.5) and 37.2% at
50% (−1.5, CI −7.3 to +4.2). With 100 prompts neither difference is distinguishable from zero.
A perfect gate would reach 40.2% with 30% of turns on the model.

### Fitting on all 1,198 held-out prompts

Doubling the fitting data from 600 to 1,198 prompts changed nothing: code-side AUROC stayed at 0.885,
model-side went from 0.68 to 0.70, and the curves moved by less than a point. The gate is no longer
short of data.

| Turns answered by the model | 30% | 40% | 50% | 60% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| Held-out, cross-validated (n=1,198) | 33.7% | 36.7% | 38.6% | 39.2% | 40.7% | 37.8% |
| Benchmark sample (n=100) | 30.1% | 32.8% | 36.9% | 38.3% | 40.8% | 38.8% |
| Perfect gate, held-out | 44.3% | 46.7% | 47.3% | 47.4% | 47.4% | 37.8% |

The cross-validated row is the most reliable number so far: the hybrid is level with SmolLM2-360M
alone when Python answers a little over half the turns, and about 3 points above it when Python
answers a quarter.

### A chooser over several Python replies (`bench/chooser.py`)

Instead of only gating PyBot's one reply, the Python side now offers three, none of which generates
text: PyBot's reply, the PyBot knowledge entry closest in meaning to the question (6,262 entries,
first two sentences), and the logged answer to the closest logged question. One scorer predicts each
reply's score from meaning-level signals and the candidates' agreement with each other; the best one
is kept, and the turn goes to the model only when even that one is expected to lose.

| | Python only | 30% model | 40% | 50% | 60% | 75% | model alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| Held-out, grouped 5-fold (n=600) | 25.8% | 35.5% | 38.0% | 39.2% | 40.8% | 40.3% | 36.9% |
| Benchmark sample (n=100) | 27.1% | 34.1% | 37.4% | 42.1% | 44.0% | 46.0% | 38.8% |

PyBot alone is 19.7% / 17.3% on the same prompts, so choosing among Python replies adds 6 to 10
points with no model call. The hybrid passes SmolLM2-360M alone from about 40% model share. On the
sample at 50%: +3.3 against the model alone (95% CI −3.5 to +10.8). Picking the best Python reply
with hindsight would give 33 to 34%, and deferring perfectly on top of the chooser 48% at 30% model
share, so the scorer is still the limit.

### Latest numbers and what limits them

With the scorer trained on all judged held-out replies (PyBot's 1,198, the two other Python candidates'
600 each, and the small model's 1,198 as extra examples):

| Benchmark sample (n=100) | Python only | 20% model | 30% | 40% | 50% | 60% | 75% | model alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Win rate | 27.7% | 34.8% | 37.5% | 38.7% | 40.1% | 42.0% | 43.5% | 38.8% |

PyBot alone: 17.3%. At 30% model share the difference from the model alone is −1.2 (95% CI −9.3 to
+7.0); at 50% it is +1.3 (−6.0 to +8.5). Cross-validated on 600 held-out prompts the same system
gives 25.1% Python only, 34.8% at 30%, 37.9% at 40%, 39.7% at 50%, against 36.9% for the model alone,
so the sample is a little kinder than the held-out estimate.

Things tried on top that did not help:

- More Python candidates (second and third closest entry or logged answer, cross-encoder re-ranking,
  untrimmed entries): the hindsight ceiling rises from 34% to 39% with eight candidates but each one
  is weak alone (4 to 9%) and the chooser does not benefit.
- More training examples for the scorer: no change.
- A second check after the model has answered (keep its reply only if the scorer rates it above the
  best Python reply): under half a point.
- Trimming PyBot's replies to two sentences: slightly worse (19.9% against 21.7% on held-out).

Why the scorer has stopped improving (`bench/look_at_mistakes.py`): many of its misses are PyBot
answers that are correct and on topic ("20 minutes, before 3pm" for a nap; a clear definition of a
web browser or a fraction) where the judge preferred the reference's wording. Those are close to
coin flips and no signal predicts them. The other kind is an honest "I don't know" that wins because
the reference invented a fact (a fixture for "the Wildcats", a founding date for "my hometown").

How it moves with the size of the answer log (same 100 prompts, only the logged-answer candidate
changes; the 2,098 row adds the benchmark's other 900 prompts as log, which is only valid for this
100-prompt experiment):

| Logged question/answer pairs | 300 | 600 | 1,198 | 2,098 |
|---|---:|---:|---:|---:|
| Prompts with a close logged match | 12% | 16% | 24% | 30% |
| Python only | 22.7% | 24.4% | 27.7% | 28.1% |
| Hybrid at 30% model share | 32.4% | 33.9% | 37.5% | 36.3% |

More logged answers help up to about 1,200 pairs and then flatten on this sample.

## What the sample says

- **"99% Python, 1% model" does not buy much.** At 2% model share the gain is +0.3 points.
- **Every small model alone beats the hybrid at every setting.** Gemma3-1B was not trained on the
  benchmark's source conversations (SmolLM2 was), and it scores three times PyBot.
- **A second local judge agrees on the order**: Gemma-3-12B gives PyBot 21.3% and brief-prompt
  SmolLM2-360M 55.6% on the same 100 prompts.
- **The gate is not the bottleneck.** `bench/gate_search.py` cross-validates 12 gate designs on
  held-out prompts (rules, 10 to 27 signals, three regularisation strengths, predicting the gain
  directly). All land 1 to 3.5 points above random routing; the shipped one is the best of them.
- **Even a perfect gate adds little over the model alone**: picking the better reply per prompt with
  hindsight gives 44.8% against SmolLM2-360M's 38.8%, and 58.6% against SmolLM2-1.7B's 55.3%. The
  code's good answers mostly fall on prompts the model also answers well.
- **Where the code loses.** About a third of turns come from PyBot's fallback modules ("I don't know
  enough to answer that"); those score about 4% as code and 36% as model, and the gate finds them.
  On the rest, where PyBot does give a real answer, it scores about 26% against the model's 38 to 45%,
  and no signal we tried separates the good ones.

Two other ideas were tried and did not help:

- **A recall tier** (`hybrid/recall.py`): answer from the 1,198 logged held-out question/answer pairs
  when a TF-IDF match is close. Only 11% of benchmark prompts have a close match (similarity ≥ 0.7),
  and there it scores 32%, below the model.
- **A one-token check** (`bench/verify_signal.py`): ask the small model "does this reply answer the
  question, yes or no" about PyBot's reply. With SmolLM2-1.7B it predicts PyBot's score well
  (AUROC 0.81) and adds about 1 point to the gate; with SmolLM2-360M it adds nothing.

- **Recall by meaning** (`bench/embed.py`, all-MiniLM, 23M parameters, no generation): match the
  question to the 1,198 logged ones by embedding and return the logged answer when similarity is
  at least 0.8. That covers 24% of benchmark prompts and scores about what SmolLM2-360M scores on
  them. Recall first, SmolLM2-360M otherwise: 38.3% with 76% of turns on the model, against 38.8%
  for the model alone. So it saves a quarter of model calls at the same quality; it does not raise
  quality, and it is below SmolLM2-1.7B. Below 0.8 similarity the logged answer is close to useless.
  Putting PyBot between the two (recall 24%, PyBot 51%, model 25%) gives 27.1%.

100 prompts is a small sample (intervals of about ±7 points), and the judge is not the benchmark's.

## Run it

```sh
ollama pull smollm2:360m
HYBRID_LLM_MODEL=smollm2:360m python -m hybrid            # terminal chat, replies tagged python/model
HYBRID_LLM_MODEL=smollm2:360m python demo/server.py       # web demo on http://localhost:7860
python -m unittest tests.test_hybrid
```

`Dockerfile` packages PyBot, Ollama and the model into one image (port 7860, the layout a Hugging Face
Docker Space expects). It has not been built or deployed yet.

## Reproduce the numbers

```sh
python bench/run_pybot.py bench/data/smoltalkbench.jsonl bench/out/pybot_bench.jsonl      # PyBot's replies + trace
python bench/run_llm.py smollm2:360m bench/data/smoltalkbench.jsonl bench/out/smollm2-360m-brief_bench.jsonl \
    --system "Answer in one to three short, plain sentences. No lists, no markdown."
python bench/judge.py REFS.jsonl REPLIES.jsonl OUT.judged.jsonl --base-url http://localhost:8091/v1 --model NAME
JUDGED_DIR=bench/poc python bench/fit_gate.py smollm2-360m-brief       # held-out prompts only
JUDGED_DIR=bench/poc python bench/evaluate.py                          # table + chart
```

Replies for all 1,000 benchmark prompts and 600 to 1,198 held-out prompts are already in `bench/out/`
for PyBot, SmolLM2-360M, SmolLM2-1.7B and Gemma3-1B; only the judging was cut to the small sample.
The hybrid's reply is always one of two already-judged replies, so scoring any gate setting needs no
further judge calls.
