#!/bin/sh
cd /Users/ozkilim/pybot
# retry until every reply is judged (a tunnel drop just resumes from the cache)
J() { n=$(wc -l < "$2"); until [ -f "$3" ] && [ $(wc -l < "$3") -ge $n ]; do .venv/bin/python bench/judge.py "$1" "$2" "$3" --base-url http://localhost:$4/v1 --model qwen3.6-35b-a3b-q4_k_m --workers 8 --cache "$5" 2>&1 | tail -1; sleep 5; done; }
( for s in crystal12-sts crystal12-k2 crystal12-k3; do J bench/data/smoltalkbench.jsonl bench/out/${s}_benchall.jsonl bench/full/${s}_benchall.judged.jsonl 8191 bench/out/judge_cache_janos_e.jsonl; done ) &
( for s in crystal12-sts crystal12-k2 crystal12-k3; do J bench/data/heldout.jsonl bench/out/${s}_heldoutall.jsonl bench/cand/${s}_heldoutall.judged.jsonl 8192 bench/out/judge_cache_janos_f.jsonl; done ) &
wait
cat bench/out/judge_cache_janos_e.jsonl bench/out/judge_cache_janos_f.jsonl >> bench/out/judge_cache.jsonl
echo MORE_DONE
