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
