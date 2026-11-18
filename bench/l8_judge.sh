#!/bin/sh
# Judge Llama-3.1-8B (the stronger comparator) and the cache it wrote. Usage: bench/l8_judge.sh model|cache
cd /Users/ozkilim/pybot
J() { n=$(wc -l < "$2"); until [ -f "$3" ] && [ $(wc -l < "$3") -ge $n ]; do .venv/bin/python bench/judge.py "$1" "$2" "$3" --base-url http://localhost:$4/v1 --model qwen3.6-35b-a3b-q4_k_m --workers 8 --cache "$5" 2>&1 | tail -1; sleep 5; done; }
if [ "$1" = model ]; then
  J bench/data/smoltalkbench.jsonl bench/out/llama8b-brief_bench.jsonl bench/full/llama8b-brief_bench.judged.jsonl 8191 bench/out/judge_cache_janos_g.jsonl &
  J bench/data/heldout.jsonl bench/out/llama8b-brief_heldout.jsonl bench/poc1198/llama8b-brief_heldout.judged.jsonl 8192 bench/out/judge_cache_janos_h.jsonl &
  wait; cat bench/out/judge_cache_janos_g.jsonl bench/out/judge_cache_janos_h.jsonl >> bench/out/judge_cache.jsonl
else
  J bench/data/smoltalkbench.jsonl bench/out/crystal12l_benchall.jsonl bench/full/crystal12l_benchall.judged.jsonl 8191 bench/out/judge_cache_janos_i.jsonl &
  J bench/data/heldout.jsonl bench/out/crystal12l_heldoutall.jsonl bench/cand/crystal12l_heldoutall.judged.jsonl 8192 bench/out/judge_cache_janos_j.jsonl &
  wait; cat bench/out/judge_cache_janos_i.jsonl bench/out/judge_cache_janos_j.jsonl >> bench/out/judge_cache.jsonl
fi
echo L8_$1_DONE
