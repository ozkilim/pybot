"""A small web demo of the cascade. Standard library only.

    python demo/server.py                 # http://localhost:7860
    PORT=8000 HYBRID_LLM_MODEL=smollm2:360m python demo/server.py

Each reply shows which tier answered (Python or the model), the module or model behind it,
the gate's numbers, and what the code would have said when the turn was deferred.
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

from hybrid import Cascade  # noqa: E402

SESSION_TTL = 3600
MAX_SESSIONS = 500
MAX_TEXT = 1000

print("Loading PyBot (about 640,000 lines of Python)...", flush=True)
BOT = Cascade()
LOCK = threading.Lock()
SESSIONS = {}          # id -> {"memory": {}, "seen": time, "code": n, "llm": n}
TOTAL = {"code": 0, "llm": 0}


def _results():
    path = os.path.join(ROOT, "bench", "results", "results.json")
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _session(key):
    now = time.time()
    if len(SESSIONS) > MAX_SESSIONS:
        for old in [k for k, v in SESSIONS.items() if now - v["seen"] > SESSION_TTL]:
            del SESSIONS[old]
        if len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.clear()
    session = SESSIONS.setdefault(key, {"memory": {}, "seen": now, "code": 0, "llm": 0})
    session["seen"] = now
    return session


def chat(payload):
    text = str(payload.get("text", "")).strip()[:MAX_TEXT]
    if not text:
        return {"error": "empty message"}
    budget = str(payload.get("budget", "0.10"))
    with LOCK:  # PyBot's modules share global state, and the gate setting is per request
        session = _session(str(payload.get("session", "anon"))[:64])
        if budget in ("0", "0.00"):
            BOT.gate.set_budget("0.01", tau=2.0)      # never defer: PyBot alone
        else:
            BOT.gate.set_budget(budget)
        out = BOT.turn(text, session["memory"])
        session[out["tier"]] += 1
        TOTAL[out["tier"]] += 1
        out["session"] = {"code": session["code"], "llm": session["llm"]}
        out["all_sessions"] = dict(TOTAL)
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
        elif self.path == "/api/info":
            self._send(200, {"model": getattr(BOT.llm, "name", None), "gate_model": BOT.gate.llm,
                             "budgets": ["0"] + BOT.gate.budgets(), "results": _results(), "total": TOTAL})
        elif self.path == "/healthz":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 20000)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if self.path == "/api/chat":
            self._send(200, chat(payload))
        elif self.path == "/api/reset":
            with LOCK:
                SESSIONS.pop(str(payload.get("session", "anon"))[:64], None)
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):  # quiet
        pass


def main():
    port = int(os.environ.get("PORT", "7860"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("Demo on http://localhost:%d  (model tier: %s)" % (port, getattr(BOT.llm, "name", "none")), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
