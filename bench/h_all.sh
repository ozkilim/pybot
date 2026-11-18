#!/bin/sh
# Judge the knowledge-entry and logged-answer candidates on all 1,198 held-out prompts (600 are cached).
cd /Users/ozkilim/pybot
until curl -s localhost:8091/health | grep -q ok; do sleep 3; done
for t in entry-emb-trim2 recall-emb; do
 .venv/bin/python bench/judge.py bench/data/heldout.jsonl bench/out/${t}_heldoutall.jsonl bench/cand/${t}_heldoutall.judged.jsonl --model qwen3.6-35b-a3b-q4_k_m --workers 4 2>&1 | tail -1; echo "DONE $t $(date +%T)"
done
echo HALL_DONE
