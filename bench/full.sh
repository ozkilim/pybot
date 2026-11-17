#!/bin/sh
# Judge every reply the full-benchmark hybrid can pick from, on all 1,000 benchmark prompts.
cd /Users/ozkilim/pybot
until curl -s localhost:8091/health | grep -q ok; do sleep 3; done
j() { .venv/bin/python bench/judge.py bench/data/smoltalkbench.jsonl $1 bench/full/$2.judged.jsonl --model qwen3.6-35b-a3b-q4_k_m --workers 4 2>&1 | tail -2; echo "DONE $2 $(date +%T)"; }
j bench/out/pybot_bench.jsonl pybot_bench
j bench/out/smollm2-360m-brief_bench.jsonl smollm2-360m-brief_bench
j bench/out/entry-emb-trim2_benchall.jsonl entry-emb-trim2_benchall
j bench/out/recall-emb_benchall.jsonl recall-emb_benchall
echo FULL_DONE
