#!/bin/sh
# questions (already running) -> answers -> embeddings
cd /Users/ozkilim/pybot
true
ollama stop qwen2.5:7b
.venv/bin/python bench/crystallize.py answers
.venv/bin/python bench/crystallize.py embed
echo CRYSTAL_DONE
