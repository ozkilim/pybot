#!/bin/sh
# Pull the 8B-written cache from janos, embed it, build candidates, judge them.
cd /Users/ozkilim/pybot
until ssh janos 'grep -q DONE ~/pybot_judge/work/l8a.log && grep -q DONE ~/pybot_judge/work/l8b.log'; do sleep 60; done
rsync -a janos:~/pybot_judge/work/bench/out/crystal12la_a.jsonl janos:~/pybot_judge/work/bench/out/crystal12lb_a.jsonl bench/out/
cat bench/out/crystal12la_a.jsonl bench/out/crystal12lb_a.jsonl > bench/out/crystal12l_a.jsonl
wc -l bench/out/crystal12l_a.jsonl
CRYSTAL=crystal12l .venv/bin/python bench/crystallize.py embed
CRYSTAL=crystal12l .venv/bin/python bench/crystal_cand.py
bench/l8_judge.sh cache
