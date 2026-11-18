#!/bin/sh
# Judge on janos (two GPUs, reached through an SSH tunnel on local ports 8191/8192).
cd /Users/ozkilim/pybot
S=/private/tmp/claude-501/-Users-ozkilim--ssh/155060f6-62fd-4324-90f0-35d5081735b1/scratchpad
J() { .venv/bin/python bench/judge.py "$1" "$2" "$3" --base-url http://localhost:$4/v1 --model qwen3.6-35b-a3b-q4_k_m --workers 8 --cache "$5" 2>&1 | tail -1; }
until curl -s localhost:8191/health | grep -q ok && curl -s localhost:8192/health | grep -q ok; do sleep 5; done
# 1. consistency: re-judge 200 replies the Mac already scored, with an empty cache
head -200 bench/out/pybot_bench.jsonl > $S/check200.jsonl; rm -f $S/check_cache.jsonl
t0=$(date +%s); J bench/data/smoltalkbench.jsonl $S/check200.jsonl $S/check200.judged.jsonl 8191 $S/check_cache.jsonl; echo "CHECK $(( $(date +%s) - t0 ))s for 200"
# 2. cache candidates, one GPU each
J bench/data/smoltalkbench.jsonl bench/out/crystal_benchall.jsonl bench/full/crystal_benchall.judged.jsonl 8191 bench/out/judge_cache_janos_a.jsonl &
J bench/data/heldout.jsonl bench/out/crystal_heldoutall.jsonl bench/cand/crystal_heldoutall.judged.jsonl 8192 bench/out/judge_cache_janos_b.jsonl &
wait
echo JJ_DONE
