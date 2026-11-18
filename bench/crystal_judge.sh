#!/bin/sh
# After the cache is built: make the cache candidate, judge it on held-out and benchmark prompts.
cd /Users/ozkilim/pybot
until grep -q CRYSTAL_DONE bench/crystal_run.log; do sleep 30; done
.venv/bin/python bench/crystal_cand.py
ollama stop smollm2:360m; ollama stop all-minilm
nohup ~/.cache/llama-metal-build/bin/llama-server -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8091 -ngl 999 -c 16384 -np 4 --jinja -fa on > bench/judge_server.log 2>&1 &
until curl -s localhost:8091/health | grep -q ok; do sleep 3; done
.venv/bin/python bench/judge.py bench/data/smoltalkbench.jsonl bench/out/crystal_benchall.jsonl bench/full/crystal_benchall.judged.jsonl --model qwen3.6-35b-a3b-q4_k_m --workers 4 2>&1 | tail -1
echo "DONE bench $(date +%T)"
.venv/bin/python bench/judge.py bench/data/heldout.jsonl bench/out/crystal_heldoutall.jsonl bench/cand/crystal_heldoutall.judged.jsonl --model qwen3.6-35b-a3b-q4_k_m --workers 4 2>&1 | tail -1
echo "DONE heldout $(date +%T)"
echo CJ_DONE
