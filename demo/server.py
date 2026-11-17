"""Live side-by-side demo: PyBot, a small model, and PyBot-70. Standard library server.

    python demo/server.py                 # http://localhost:7860  (needs Ollama: smollm2:360m, all-minilm)

One message goes to all three. Each column keeps its own conversation:
  pybot     PyBot alone (640,000 lines of Python, no model)
  model     SmolLM2-360M on every turn
  pybot70   three Python replies + a scorer; the model only when it is expected to do better
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from hybrid.llm import BRIEF, OllamaLLM  # noqa: E402
from hybrid.pybot70 import PyBot70, _history  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("HYBRID_LLM_MODEL", "smollm2:360m")
SESSION_TTL, MAX_SESSIONS, MAX_TEXT = 3600, 500, 1000
BENCH = {"pybot": 23.0, "model": 38.8, "pybot70": 39.8}     # full SmolTalkBench, local judge (HYBRID.md)

print("Loading PyBot (about 640,000 lines of Python) and the PyBot-70 router...", flush=True)
LLM = OllamaLLM(MODEL, OLLAMA, system=BRIEF)
BOT = PyBot70(llm=LLM, ollama=OLLAMA)
LOCK = threading.Lock()          # PyBot's modules share global state
SESSIONS = {}


def _session(key):
    now = time.time()
    if len(SESSIONS) > MAX_SESSIONS:
        for old in [k for k, v in SESSIONS.items() if now - v["seen"] > SESSION_TTL]:
            del SESSIONS[old]
    s = SESSIONS.setdefault(key, {"seen": now, "pybot": {}, "pybot70": {}, "model": [],
                                  "count": {"python": 0, "model": 0}})
    s["seen"] = now
    return s


def turn(payload):
    text = str(payload.get("text", "")).strip()[:MAX_TEXT]
    system = payload.get("system")
    if not text or system not in BENCH:
        return {"error": "bad request"}
    s = _session(str(payload.get("session", "anon"))[:64])
    t0 = time.perf_counter()
    if system == "pybot":
        with LOCK:
            try:
                r = BOT.runtime.turn(text, s["pybot"])
                reply, module = " ".join(str(r["reply"]).split()) or "Hello!", r.get("module")
            except Exception:  # noqa: BLE001
                reply, module = "Hello!", None
        return {"reply": reply, "tier": "python", "label": "Python · " + (module or "pybot"),
                "ms": (time.perf_counter() - t0) * 1000}
    if system == "model":
        history = s["model"][-8:] + [{"role": "user", "content": text}]
        try:
            reply = LLM.chat(history)
        except Exception as e:  # noqa: BLE001
            return {"error": "model unavailable: %s" % e}
        s["model"] = history + [{"role": "assistant", "content": reply}]
        return {"reply": reply, "tier": "model", "label": "Model · " + MODEL, "ms": (time.perf_counter() - t0) * 1000}
    with LOCK:                                   # Python side and routing; the model call happens outside the lock
        out = BOT.decide(text, s["pybot70"])
    out = BOT.answer(out, text, s["pybot70"])
    s["count"][out["tier"]] += 1
    out["label"] = ("Model · " + MODEL) if out["tier"] == "model" else ("Python · " + out["source_label"])
    out["ms"] = (time.perf_counter() - t0) * 1000
    out["count"] = dict(s["count"])
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as handle:
                self._send(200, handle.read(), "text/html; charset=utf-8")
        elif self.path == "/moneyshot.png":
            with open(os.path.join(ROOT, "bench", "results", "moneyshot.png"), "rb") as handle:
                self._send(200, handle.read(), "image/png")
        elif self.path == "/api/info":
            self._send(200, {"model": MODEL, "bench": BENCH, "model_share": BOT.model_share})
        elif self.path == "/healthz":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            payload = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 20000)) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if self.path == "/api/turn":
            self._send(200, turn(payload))
        elif self.path == "/api/reset":
            SESSIONS.pop(str(payload.get("session", "anon"))[:64], None)
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "7860"))
    print("Demo on http://localhost:%d  (model: %s)" % (port, MODEL), flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
