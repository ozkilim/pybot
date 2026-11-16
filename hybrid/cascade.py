"""PyBot first, a small model only when the code should not answer.

    from hybrid import Cascade
    bot = Cascade()                       # gate weights from hybrid/gate_weights.json, LLM from env
    memory = {}
    out = bot.turn("why is the sky blue?", memory)
    out["reply"], out["tier"]             # tier is "code" or "llm"

Every turn enters at the code tier. PyBot (bot/, untouched) proposes its reply; the gate reads
what the code did and either lets the reply through or defers the turn to the model tier.
The response carries the tier, the module, the gate's numbers and its reasons, so every answer
can be audited.
"""
import time

from .gate import Gate
from .llm import from_env

MAX_HISTORY_TURNS = 4


def _load_runtime():
    from bot import runtime
    return runtime


class Cascade:
    def __init__(self, llm=None, gate=None, budget=None, runtime=None):
        self.runtime = runtime or _load_runtime()
        self.llm = llm if llm is not None else from_env()
        self.gate = gate or Gate(llm=_gate_name(self.llm), budget=budget)
        self.stats = {"turns": 0, "code": 0, "llm": 0, "llm_failed": 0}

    def turn(self, text, memory, budget=None):
        if budget is not None:
            self.gate.set_budget(budget)
        start = time.perf_counter()
        try:
            result = self.runtime.turn(text, memory)
            code_reply = " ".join(str(result["reply"]).split()) or "Hello!"
        except Exception:  # noqa: BLE001 - same contract as pybot.py: a turn never fails
            result, code_reply = {"reply": "Hello!", "module": None, "candidates": []}, "Hello!"
        code_ms = (time.perf_counter() - start) * 1000
        decision = self.gate.decide(text, result)
        out = {"reply": code_reply, "tier": "code", "module": result.get("module"), "code_reply": code_reply,
               "code_ms": code_ms, "llm_ms": 0.0, "model": None, **decision}
        if decision["defer"] and self.llm is not None:
            started = time.perf_counter()
            try:
                reply = self.llm.chat(_history(memory, text))
            except Exception:  # noqa: BLE001 - the model tier is optional: fall back to the code
                reply = ""
                self.stats["llm_failed"] += 1
            out["llm_ms"] = (time.perf_counter() - started) * 1000
            if reply:
                out.update(reply=reply, tier="llm", model=getattr(self.llm, "name", "llm"))
                turns = memory.get("turns")
                if isinstance(turns, list) and turns:   # keep the thread true to what was said
                    turns[-1] = dict(turns[-1], bot=reply, module="llm")
        self.stats["turns"] += 1
        self.stats[out["tier"]] += 1
        return out

    def reply(self, text, memory):
        return self.turn(text, memory)["reply"]


def _history(memory, text):
    """The last few turns as chat messages, ending with this user message."""
    messages = []
    turns = memory.get("turns") if isinstance(memory, dict) else None
    previous = turns[:-1] if isinstance(turns, list) else []   # the runtime already appended this turn
    for turn in previous[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": str(turn.get("user", ""))})
        messages.append({"role": "assistant", "content": str(turn.get("bot", ""))})
    messages.append({"role": "user", "content": text})
    return messages


def _gate_name(llm):
    return getattr(llm, "name", None)
