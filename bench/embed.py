"""Sentence embeddings from a small local model (all-MiniLM, 23M parameters) through Ollama."""
import json
import urllib.request

import numpy as np


def embed(texts, model="all-minilm", host="http://localhost:11434", batch=64):
    out = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(host + "/api/embed", data=json.dumps({"model": model, "input": texts[i:i + batch]}).encode(),
                                     headers={"Content-Type": "application/json"})
        out.extend(json.load(urllib.request.urlopen(req, timeout=300))["embeddings"])
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
