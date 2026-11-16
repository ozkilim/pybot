# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0", "huggingface_hub>=0.30"]
# ///
"""Build SmolTalkBench: single-turn chat prompts with reference replies, AlpacaEval style.

    uv run scripts/create_dataset.py            # build and print stats
    uv run scripts/create_dataset.py --push     # also push the data, card and this script

This builds the `gold` config, whose references come from the source conversations. Variants
with other reference models go in their own configs, keeping the same ids and instructions.

Source: the everyday-conversations subset of HuggingFaceTB/smoltalk (train + test), written
whole by Llama-3.1-70B-Instruct. Every conversation opens with a bare greeting and the same
"How can I help you today?", so the prompt is the second user turn and the reference is the
assistant's reply to it. Prompts are deduplicated, then sampled in proportion to their parent
topic (full_topic without its leaf), largest remainders rounding the allocation to exactly N.
Columns follow AlpacaEval's reference outputs (instruction, output, generator, dataset).
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi

SOURCE = "HuggingFaceTB/smoltalk"
SUBSET = "everyday-conversations"
SOURCE_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
GENERATOR = "meta-llama/Meta-Llama-3.1-70B-Instruct"
REPO = "HuggingFaceSwarms/SmolTalkBench"
CONFIG = "gold"
SCRIPT = "scripts/create_dataset.py"
META = {"license": "apache-2.0", "language": ["en"], "pretty_name": "SmolTalkBench",
        "tags": ["chatbot", "evaluation", "alpaca-eval"]}


def candidates(source_revision):
    """One row per distinct second user turn, first occurrence kept (train, then test)."""
    ds = load_dataset(SOURCE, SUBSET, revision=source_revision)
    rows, seen = [], set()
    for split in ("train", "test"):
        for index, record in enumerate(ds[split]):
            msgs = record["messages"]
            assert [m["role"] for m in msgs[:4]] == ["user", "assistant", "user", "assistant"]
            prompt = msgs[2]["content"].strip()
            key = " ".join(prompt.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            topic = record["full_topic"]
            rows.append({"id": f"{split}-{index:04d}", "instruction": prompt,
                         "output": msgs[3]["content"].strip(), "generator": GENERATOR,
                         "dataset": f"{SOURCE}/{SUBSET}", "topic": topic,
                         "category": topic.split("/")[0],
                         # Capitalised three-level topics are everyday requests; lowercase
                         # two-level ones are grade-school science and general knowledge.
                         "group": "everyday" if topic.count("/") == 2 else "school",
                         "source_split": split, "source_index": index})
    return rows


def stratified(rows, n, seed):
    """Proportional allocation over parent topics, rounded by largest remainder."""
    strata = defaultdict(list)
    for row in rows:
        strata[row["topic"].rsplit("/", 1)[0]].append(row)
    names = sorted(strata)
    quotas = {s: n * len(strata[s]) / len(rows) for s in names}
    alloc = {s: int(q) for s, q in quotas.items()}
    for s in sorted(names, key=lambda s: (alloc[s] - quotas[s], s))[:n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(seed)
    picked = [row for s in names for row in rng.sample(strata[s], alloc[s])]
    return sorted(picked, key=lambda row: (row["source_split"] != "train", row["source_index"]))


def card(args, total):
    public = " --public" if args.public else ""
    return f"""| subset | description | num_samples |
| --- | --- | ---: |
| `{CONFIG}` (default) | Single-turn everyday chat prompts from smoltalk's `{SUBSET}`, sampled in proportion to their parent topic, with the source conversations' replies by `{GENERATOR}` as references | {args.n} |

{args.n} single-turn everyday chat prompts with reference replies, for AlpacaEval-style pairwise
win rates.

```python
from datasets import load_dataset

ds = load_dataset("{args.repo}", "{CONFIG}", split="test")
```

Variants with other reference models go in their own configs, keeping the same `id`s and
instructions so that results compare across them.

## What it's for

SmolTalkBench was built to measure [PyBot](https://huggingface.co/spaces/HuggingFaceSwarms/PyBot),
a chatbot that a swarm of agents wrote entirely in Python: no weights, no training, just
if-else branches, regex patterns and lookup tables. Every version of PyBot was scored on
SmolTalkBench, with small instruction-tuned LMs as reference points. Its win rate climbed from
0% to 26% over more than 6,000 commits, on par with SmolLM-360M-Instruct.

![PyBot's win rate on SmolTalkBench over its commits, with three SmolLM-Instruct baselines as horizontal lines](https://huggingface.co/datasets/{args.repo}/resolve/main/assets/progress.png)

SmolLM and SmolLM2 Instruct models were trained on the source conversations, so the SmolLM
baselines' scores are inflated.

## How it's scored

- The model answers each `instruction` once, as a single user turn.
- An LLM judge (`deepseek-ai/DeepSeek-V4.1-Flash`, with AlpacaEval 2.0's weighted annotator
  prompt) compares the reply with the reference `output`. Which one it sees first is fixed per
  prompt, as in AlpacaEval. A sample's score is the judge's probability that the reply is
  better, read from its top-5 logprobs.
- The win rate is the mean score with [WildBench](https://github.com/allenai/WildBench)'s
  length penalty: a reply more than 500 characters longer than its reference counts as a tie at
  most. PyBot's win rates above use it.

## Construction

- Source: the `{SUBSET}` subset of [`{SOURCE}`](https://huggingface.co/datasets/{SOURCE})
  at revision `{args.source_revision[:12]}`, train and test splits. Each conversation was
  written whole by [`{GENERATOR}`](https://huggingface.co/{GENERATOR}).
- Every source conversation opens with a bare greeting ("Hi", "Hello", ...) answered by
  "Hello! How can I help you today?". The greeting is dropped: `instruction` is the second
  user turn and `output` the assistant's reply to it.
- Prompts are deduplicated (case and whitespace insensitive), leaving {total} rows.
- {args.n} rows are sampled with seed {args.seed}, in proportion to each parent topic (`topic`
  without its last level), so every subcategory keeps its share.

## Columns

| column | description |
|---|---|
| `id` | `<source split>-<source index>` |
| `instruction` | the prompt (second user turn) |
| `output` | reference reply by `generator` |
| `generator`, `dataset` | reference model and source, as in AlpacaEval's reference outputs |
| `topic` | the source `full_topic` |
| `category` | first level of `topic` |
| `group` | `everyday` (three-level topics) or `school` (two-level grade-school knowledge topics) |
| `source_split`, `source_index` | the row in the source subset |

## Reproducing

[`{SCRIPT}`]({SCRIPT}) builds the `{CONFIG}` config and pushes it with this card and itself:

```sh
hf download {args.repo} {SCRIPT} --repo-type dataset --local-dir .
uv run {SCRIPT} --n {args.n} --seed {args.seed} \\
  --source_revision {args.source_revision} --repo {args.repo} --push{public}
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_revision", default=SOURCE_REVISION,
                        help=f"commit of {SOURCE}")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--public", action="store_true", help="push as a public dataset")
    args = parser.parse_args()
    rows = candidates(args.source_revision)
    picked = stratified(rows, args.n, args.seed)
    assert len(picked) == args.n and len({r["id"] for r in picked}) == args.n
    print(f"{len(rows)} distinct prompts -> {len(picked)} sampled")
    print("groups:", dict(Counter(r["group"] for r in picked)))
    print("categories:", dict(Counter(r["category"] for r in picked).most_common()))
    if not args.push:
        return
    Dataset.from_list(picked).push_to_hub(args.repo, config_name=CONFIG, split="test",
                                          private=not args.public)
    # push_to_hub wrote the card's dataset_info and configs; keep them, add ours.
    pushed = DatasetCard.load(args.repo, repo_type="dataset")
    for key, value in META.items():
        pushed.data[key] = value
    # push_to_hub(set_default=True) drops or trips over the flag when re-pushing the default
    # config (datasets 5.1), so make gold the default here.
    for entry in pushed.data["configs"]:
        if entry["config_name"] == CONFIG:
            entry["default"] = True
    pushed.text = card(args, len(rows))
    # The card's chart, assets/progress.png, is uploaded separately and left in place here.
    api = HfApi()
    api.create_commit(args.repo, repo_type="dataset", operations=[
        CommitOperationAdd("README.md", str(pushed).encode()),
        CommitOperationAdd(SCRIPT, Path(__file__).read_bytes()),
    ], commit_message="Publish the card and the script that built this dataset")
    print(f"pushed https://huggingface.co/datasets/{args.repo} @ "
          f"{api.dataset_info(args.repo).sha}")


if __name__ == "__main__":
    main()
