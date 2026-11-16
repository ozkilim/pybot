"""The model tier: a small language model behind one function, chat(messages) -> text.

Standard library only (urllib). Two backends:

  OllamaLLM("smollm2:1.7b")                           a local Ollama server
  OpenAILLM(base_url, model, api_key)                 anything that speaks /v1/chat/completions:
                                                      llama.cpp's llama-server, vLLM, a hosted API

`from_env()` builds one from HYBRID_LLM_* environment variables, so the demo and the benchmark
share a configuration.
"""
import json
import os
import urllib.request

BRIEF = "Answer in one to three short, plain sentences. No lists, no markdown."


def _post(url, body, headers=None, timeout=120):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class OllamaLLM:
    def __init__(self, model="smollm2:1.7b", host="http://localhost:11434", system=BRIEF, max_tokens=256):
        self.model, self.host, self.system, self.max_tokens = model, host.rstrip("/"), system, max_tokens
        self.name = model

    def chat(self, messages):
        if self.system:
            messages = [{"role": "system", "content": self.system}] + list(messages)
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "seed": 0, "num_predict": self.max_tokens}}
        data = _post(self.host + "/api/chat", body)
        return " ".join(data["message"]["content"].split())


class OpenAILLM:
    def __init__(self, base_url, model, api_key=None, system=BRIEF, max_tokens=256):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.system, self.max_tokens = system, max_tokens
        self.name = model

    def chat(self, messages):
        if self.system:
            messages = [{"role": "system", "content": self.system}] + list(messages)
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        data = _post(self.base_url + "/chat/completions", body, headers)
        return " ".join((data["choices"][0]["message"].get("content") or "").split())


def from_env():
    """HYBRID_LLM_BACKEND=ollama|openai, HYBRID_LLM_MODEL, HYBRID_LLM_URL, HYBRID_LLM_KEY."""
    backend = os.environ.get("HYBRID_LLM_BACKEND", "ollama")
    model = os.environ.get("HYBRID_LLM_MODEL", "smollm2:1.7b")
    if backend == "openai":
        return OpenAILLM(os.environ.get("HYBRID_LLM_URL", "http://localhost:8080/v1"), model,
                         os.environ.get("HYBRID_LLM_KEY"))
    return OllamaLLM(model, os.environ.get("HYBRID_LLM_URL", "http://localhost:11434"))
