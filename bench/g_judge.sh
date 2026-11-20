#!/bin/sh
# After the Gemma cache is written: swap the Gemma server for the local judge, judge both splits.
cd /Users/ozkilim/pybot
until [ -f bench/out/crystal12g_heldoutall.jsonl ] && [ -f bench/out/crystal12g_benchall.jsonl ]; do sleep 30; done
pkill -f "llama-server.*--port 8095"; sleep 5
nohup /Users/ozkilim/.cache/llama-metal-build/bin/llama-server -m /Users/ozkilim/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8091 -ngl 999 -c 16384 -np 4 --jinja -fa on > bench/judge_server.log 2>&1 &
until curl -s localhost:8091/health | grep -q ok; do sleep 5; done
J() { n=$(wc -l < "$2"); until [ -f "$3" ] && [ $(wc -l < "$3") -ge $n ]; do .venv/bin/python bench/judge.py "$1" "$2" "$3" --base-url http://localhost:8091/v1 --model qwen3.6-35b-a3b-q4_k_m --workers 4 2>&1 | tail -1; sleep 5; done; }
J bench/data/smoltalkbench.jsonl bench/out/crystal12g_benchall.jsonl bench/full/crystal12g_benchall.judged.jsonl
J bench/data/heldout.jsonl bench/out/crystal12g_heldoutall.jsonl bench/cand/crystal12g_heldoutall.judged.jsonl
echo G_JUDGE_DONE
