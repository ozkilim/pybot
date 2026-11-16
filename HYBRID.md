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

Scored on **100 of SmolTalkBench's 1,000 prompts** (random, seed 0). The gate was fitted on **150
held-out prompts** from the same source that are not in the benchmark. Same scoring recipe as the
benchmark card (AlpacaEval 2.0 weighted prompt, logprob score, WildBench length penalty), but with a
**local judge, Qwen3.6-35B-A3B**, not the card's DeepSeek-V4.1-Flash. So compare rows in this table
with each other, not with the published 26.7%.

| System | Win rate | 95% CI | Turns answered by a model |
|---|---:|---|---:|
| PyBot, code only | 17.3% | 10.8–24.4 | 0% |
| Hybrid | 17.5% | 10.8–24.5 | 3% |
| Hybrid | 21.3% | 14.5–28.7 | 14% |
| Hybrid | 26.9% | 19.7–34.9 | 36% |
| Hybrid | 34.6% | 26.9–42.5 | 53% |
| SmolLM2-360M alone (brief prompt) | 38.8% | 30.9–46.6 | 100% |

Full table: [bench/results/RESULTS.md](bench/results/RESULTS.md). Chart: `bench/results/tradeoff.png`.

What this says so far:

- **"99% Python, 1% model" does not buy much here.** At 3% model share the gain is +0.2 points.
- **The small model alone beats the hybrid at every setting.** With a "keep it brief" system prompt,
  SmolLM2-360M scores about twice PyBot under this judge. (SmolLM2 was trained on the conversations the
  benchmark is drawn from, so its score is inflated; a model that was not is still to be scored.)
- **The gate does add something.** From about 10% model share it beats routing the same number of
  turns at random, by 1 to 6 points. Turns it defers score about 10% as code and about 37% as model.
- **There is headroom.** Picking the better of the two replies per prompt, with hindsight, gives 44.8%,
  above either system alone. A gate fitted on 150 prompts (AUROC 0.69) is far from that.
- 100 prompts is a small sample: the intervals are wide and none of this is settled.

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
