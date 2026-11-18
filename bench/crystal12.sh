#!/bin/sh
# Second cache: generated on janos, merged with the first, embedded and judged.
cd /Users/ozkilim/pybot
CRYSTAL=crystal12 .venv/bin/python bench/crystal_cand.py
J() { .venv/bin/python bench/judge.py "$1" "$2" "$3" --base-url http://localhost:$4/v1 --model qwen3.6-35b-a3b-q4_k_m --workers 8 --cache "$5" 2>&1 | tail -1; }
J bench/data/smoltalkbench.jsonl bench/out/crystal12_benchall.jsonl bench/full/crystal12_benchall.judged.jsonl 8191 bench/out/judge_cache_janos_c.jsonl &
J bench/data/heldout.jsonl bench/out/crystal12_heldoutall.jsonl bench/cand/crystal12_heldoutall.judged.jsonl 8192 bench/out/judge_cache_janos_d.jsonl &
wait
cat bench/out/judge_cache_janos_c.jsonl bench/out/judge_cache_janos_d.jsonl >> bench/out/judge_cache.jsonl
SOURCES=pybot,entry-emb-trim2,recall-emb,crystal12 .venv/bin/python bench/chooser.py cv 2>&1 | grep -v -i -E "warn|redirects|loading weights|two-stage|every turn"
echo C12_DONE
