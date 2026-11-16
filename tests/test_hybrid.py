"""The cascade's own logic, with a stub code tier and a stub model: python -m unittest tests.test_hybrid"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid.cascade import Cascade, _history  # noqa: E402
from hybrid.features import VECTOR, extract  # noqa: E402
from hybrid.gate import Gate  # noqa: E402


class StubRuntime:
    def __init__(self, reply, module="answers"):
        self.reply, self.module = reply, module

    def turn(self, text, memory):
        memory.setdefault("turns", []).append({"user": text, "bot": self.reply, "module": self.module})
        return {"reply": self.reply, "module": self.module,
                "candidates": [{"module": self.module, "priority": 0, "reply": self.reply}]}


class StubLLM:
    name = "stub"

    def __init__(self, reply="from the model"):
        self.reply, self.seen = reply, []

    def chat(self, messages):
        self.seen.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def weights(tmp):
    """A gate that distrusts replies that admit not knowing, and nothing else."""
    code = {name: 0.0 for name in VECTOR}
    code.update(bias=2.0, abstains=-6.0)
    data = {"features": list(VECTOR), "stock": {}, "code": code, "default_llm": "stub", "default_budget": "0.10",
            "llm": {"stub": {"weights": {"bias": 0.0}, "taus": {"0.10": 0.0, "1.00": -2.0}}}}
    path = os.path.join(tmp, "gate.json")
    with open(path, "w") as handle:
        json.dump(data, handle)
    return path


class Features(unittest.TestCase):
    def test_guard_signals(self):
        f = extract("What is camouflage?", {"reply": "I don't know enough to answer that honestly.",
                                            "module": "greeting", "candidates": []})
        self.assertEqual(f["abstains"], 1.0)
        self.assertEqual(f["module_greeting"], 1.0)
        self.assertEqual(f["no_overlap"], 1.0)

    def test_support_signal(self):
        f = extract("How do I descale my kettle?", {"reply": "Half fill the kettle with vinegar and boil it.",
                                                    "module": "answers", "candidates": []})
        self.assertEqual(f["abstains"], 0.0)
        self.assertGreater(f["overlap_question"], 0.0)

    def test_vector_is_stable(self):
        self.assertEqual(tuple(extract("a", {"reply": "b"}).keys()), VECTOR)


class Routing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = weights(self.tmp)

    def bot(self, reply, llm=None, module="answers"):
        return Cascade(llm=llm or StubLLM(), gate=Gate(self.path, llm="stub"), runtime=StubRuntime(reply, module))

    def test_confident_code_answers(self):
        bot = self.bot("Boil it for six minutes.")
        out = bot.turn("How long should I boil an egg?", {})
        self.assertEqual((out["tier"], out["reply"]), ("code", "Boil it for six minutes."))
        self.assertEqual(bot.llm.seen, [])                      # the model was never called

    def test_abstention_defers(self):
        memory = {}
        out = self.bot("I don't know enough to answer that honestly.").turn("What is camouflage?", memory)
        self.assertEqual((out["tier"], out["reply"]), ("llm", "from the model"))
        self.assertIn("abstains", out["why"])
        self.assertEqual(out["code_reply"], "I don't know enough to answer that honestly.")
        self.assertEqual(memory["turns"][-1]["bot"], "from the model")   # the thread keeps what was said

    def test_model_failure_falls_back_to_code(self):
        bot = self.bot("I'm not sure about that one, sorry.", llm=StubLLM(RuntimeError("down")))
        out = bot.turn("What is camouflage?", {})
        self.assertEqual(out["tier"], "code")
        self.assertEqual(bot.stats["llm_failed"], 1)

    def test_budget_moves_the_threshold(self):
        gate = Gate(self.path, llm="stub", budget="1.00")
        self.assertEqual(gate.tau, -2.0)                        # everything defers
        gate.set_budget("0.10")
        self.assertEqual(gate.tau, 0.0)

    def test_history_goes_to_the_model(self):
        memory = {"turns": [{"user": "hi", "bot": "hello"}, {"user": "next", "bot": "code reply"}]}
        self.assertEqual([m["content"] for m in _history(memory, "next")], ["hi", "hello", "next"])


if __name__ == "__main__":
    unittest.main()
