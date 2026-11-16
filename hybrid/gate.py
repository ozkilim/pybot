"""The gate between the code tier and the model tier. Standard library only.

Two tiny logistic models over the signals in features.py, fitted offline on held-out prompts
(bench/fit_gate.py) and shipped as JSON:

  p_code   probability that PyBot's reply beats the reference answer
  p_llm    probability that the small model's reply would, judged from the question alone

A turn is deferred when the model is expected to do better by more than a margin tau:

    defer  <=>  p_llm - p_code > tau

which is the sequential form of a cost-aware routing rule: tau is the price of a model call in
units of answer quality. tau is picked for a target share of model calls on the calibration
prompts, so "1% LLM" or "10% LLM" is a setting, not a retrain.
"""
import json
import math
import os

from .features import VECTOR, extract

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "gate_weights.json")


def _sigmoid(z):
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class Gate:
    def __init__(self, path=DEFAULT_WEIGHTS, llm=None, budget=None, tau=None):
        with open(path) as handle:
            data = json.load(handle)
        self.stock = data.get("stock", {})
        self.code_w = data["code"]
        self.models = data["llm"]                      # name -> {"weights": {...}, "taus": {budget: tau}}
        self.llm = llm if llm in self.models else data.get("default_llm") or next(iter(self.models))
        self.set_budget(budget if budget is not None else data.get("default_budget", "0.10"), tau)

    def budgets(self):
        return sorted(self.models[self.llm]["taus"], key=float)

    def set_budget(self, budget, tau=None):
        """budget: target share of turns sent to the model, e.g. "0.01". tau overrides it."""
        taus = self.models[self.llm]["taus"]
        key = min(taus, key=lambda k: abs(float(k) - float(budget)))
        self.budget = key
        self.tau = float(tau) if tau is not None else float(taus[key])

    def score(self, text, result):
        features = extract(text, result, self.stock)
        llm_w = self.models[self.llm]["weights"]
        z_code = sum(self.code_w.get(name, 0.0) * features[name] for name in VECTOR)
        z_llm = sum(llm_w.get(name, 0.0) * features[name] for name in VECTOR)
        return features, _sigmoid(z_code), _sigmoid(z_llm)

    def decide(self, text, result):
        """{"defer": bool, "p_code", "p_llm", "margin", "tau", "why": [signal names]}"""
        features, p_code, p_llm = self.score(text, result)
        # The audit trail: the signals that pushed hardest against the code's reply.
        pulls = sorted(((self.code_w.get(name, 0.0) * features[name], name) for name in VECTOR
                        if name != "bias" and features[name]), key=lambda item: item[0])
        why = [name for pull, name in pulls[:3] if pull < 0]
        return {"defer": (p_llm - p_code) > self.tau, "p_code": p_code, "p_llm": p_llm,
                "margin": p_llm - p_code, "tau": self.tau, "why": why}
