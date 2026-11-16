#!/bin/sh
# Container entry point: start the model server, load the model once, then serve the demo.
set -e
ollama serve > /tmp/ollama.log 2>&1 &
until curl -sf "http://${OLLAMA_HOST:-127.0.0.1:11434}/api/version" > /dev/null; do sleep 1; done
# Warm the model so the first deferred turn is not the slow one.
curl -s "http://${OLLAMA_HOST:-127.0.0.1:11434}/api/generate" -d "{\"model\":\"${MODEL:-smollm2:1.7b}\",\"prompt\":\"hi\",\"options\":{\"num_predict\":1}}" > /dev/null || true
exec python demo/server.py
