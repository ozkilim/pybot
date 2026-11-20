#!/bin/sh
# Finish judging the Gemma-written cache on janos (judges on GPUs 0/1, tunnelled to 8191/8192).
cd /Users/ozkilim/pybot
( while true; do ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N -L 8191:127.0.0.1:8091 -L 8192:127.0.0.1:8092 janos; sleep 5; done ) &
TUN=$!
until curl -s localhost:8191/health | grep -q ok && curl -s localhost:8192/health | grep -q ok; do sleep 3; done
base=$(wc -l < bench/out/judge_cache.jsonl)
cp bench/out/judge_cache.jsonl bench/out/judge_cache_janos_k.jsonl; cp bench/out/judge_cache.jsonl bench/out/judge_cache_janos_l.jsonl
J() { n=$(wc -l < "$2"); until [ -f "$3" ] && [ $(wc -l < "$3") -ge $n ]; do .venv/bin/python bench/judge.py "$1" "$2" "$3" --base-url http://localhost:$4/v1 --model qwen3.6-35b-a3b-q4_k_m --workers 8 --cache "$5" 2>&1 | tail -1; sleep 5; done; }
( J bench/data/smoltalkbench.jsonl bench/out/crystal12g_benchall.jsonl bench/full/crystal12g_benchall.judged.jsonl 8191 bench/out/judge_cache_janos_k.jsonl &
  J bench/data/heldout.jsonl bench/out/crystal12g_heldoutall.jsonl bench/cand/crystal12g_heldoutall.judged.jsonl 8192 bench/out/judge_cache_janos_l.jsonl &
  wait )
tail -n +$((base+1)) bench/out/judge_cache_janos_k.jsonl >> bench/out/judge_cache.jsonl
tail -n +$((base+1)) bench/out/judge_cache_janos_l.jsonl >> bench/out/judge_cache.jsonl
kill $TUN; pkill -f "ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N -L 8191"
echo G_JANOS_DONE
