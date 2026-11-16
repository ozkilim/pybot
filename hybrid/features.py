"""Gate signals for the code tier, computed from one PyBot turn. Standard library only.

Code has no probabilities, so the gate reads what the code did instead:

  guard     did PyBot itself admit it has no answer, hand the question back, or point elsewhere?
  twin      how many modules proposed a reply, and do the proposals agree?
  support   is the reply about the question (shared content words), or a stock line that
            PyBot gives to many unrelated prompts?

Each signal is a number in a flat dict; `VECTOR` fixes their order for the tiny model in gate.py.
"""
import math
import re

STOP = frozenset("""a an the and or but if so to of in on at by for with from into about as is are was were be been being
am do does did have has had having i me my mine we us our you your yours he she it its they them their this that these
those there here what which who whom whose when where why how can could should would will shall may might must not no
yes any some many much more most other such only own same than too very just also then now really please thanks thank
hi hello hey im ive id ill dont cant like get got want need looking look tell know think going go lot lots good best
great new help tips tip ideas idea ways way thing things something anything""".split())

ABSTAIN = re.compile(
    r"\b(i (do not|don't|dont) know|not sure about|beyond what i know|i'd rather not guess|rather not guess|"
    r"would be guessing|without making it up|made-up|nothing in front of me|i can'?t (check|see|give|tell|look)|"
    r"no live|no record|no connection|no browsing|a (quick )?search|searching for it|search online|"
    r"wikipedia page|map app|depends on your postcode|can answer that better)\b", re.I)
HANDS_BACK = re.compile(
    r"\b(what have you tried|tell me (what|the|more)|say more|what else have you noticed|what is the first thing|"
    r"what's the first thing|which (one|of)|what would you|once i know)\b", re.I)
SORRY = re.compile(r"\bsorry\b", re.I)
ASK_LIST = re.compile(r"\b(recommend|suggest|suggestions?|some (popular|good|common|examples?)|what are some|"
                      r"options|examples?|ideas|types of|kinds of)\b", re.I)
ASK_FACT = re.compile(r"^\s*(what|who|when|where|which)\b|\bwhat (is|are|was|does|do)\b", re.I)
ASK_HOW = re.compile(r"\bhow (do|does|can|should|to|long|much|many|often)\b", re.I)
ASK_WHY = re.compile(r"\bwhy\b", re.I)
SCHOOL = re.compile(r"\b(learning about|studying|in (school|class)|homework|my (teacher|science|math|maths) )\b", re.I)
WORD = re.compile(r"[a-z][a-z'-]+")

FALLBACK_MODULES = ("greeting", "the_shape_of_the_answer", "chat")
MODULES = ("answers", "greeting", "the_shape_of_the_answer", "chat")


def _stem(word):
    word = word.replace("'", "")
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def content_words(text):
    return {_stem(w) for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def reply_key(reply):
    """A reply with its wording normalised, to recognise stock lines."""
    return re.sub(r"[^a-z ]", "", reply.lower())[:80]


def extract(text, result, stock=None):
    """Signals for one turn. `result` is bot.runtime.turn's return value; `stock` maps
    reply_key -> how many different calibration prompts drew that same reply."""
    reply = str(result.get("reply") or "")
    module = result.get("module") or "none"
    candidates = result.get("candidates") or []
    question, answer = content_words(text), content_words(reply)
    shared = question & answer
    others = [c for c in candidates if c.get("module") not in FALLBACK_MODULES]
    distinct = {reply_key(str(c.get("reply", ""))) for c in candidates}
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    last = sentences[-1] if sentences else text
    features = {
        "bias": 1.0,
        # guard
        "abstains": 1.0 if ABSTAIN.search(reply) else 0.0,
        "hands_back": 1.0 if HANDS_BACK.search(reply) else 0.0,
        "sorry": 1.0 if SORRY.search(reply) else 0.0,
        "reply_is_question": 1.0 if reply.rstrip().endswith("?") else 0.0,
        # twin
        "n_candidates": min(len(candidates), 5) / 5.0,
        "n_knowledge_candidates": min(len(others), 3) / 3.0,
        "candidates_disagree": 1.0 if len(distinct) > 1 else 0.0,
        # support
        "overlap_question": len(shared) / len(question) if question else 0.0,
        "overlap_reply": len(shared) / len(answer) if answer else 0.0,
        "no_overlap": 1.0 if not shared else 0.0,
        "stock_reply": min(math.log1p((stock or {}).get(reply_key(reply), 0)), 4.0) / 4.0,
        "reply_len": min(len(reply), 600) / 600.0,
        "reply_short": 1.0 if len(reply) < 60 else 0.0,
        "reply_has_digit": 1.0 if re.search(r"\d", reply) else 0.0,
        # the question itself: what kind of request the code tends to get right
        "ask_list": 1.0 if ASK_LIST.search(text) else 0.0,
        "ask_fact": 1.0 if ASK_FACT.search(last) else 0.0,
        "ask_how": 1.0 if ASK_HOW.search(text) else 0.0,
        "ask_why": 1.0 if ASK_WHY.search(text) else 0.0,
        "ask_school": 1.0 if SCHOOL.search(text) else 0.0,
        "question_len": min(len(text), 300) / 300.0,
        "question_marks": 1.0 if "?" in text else 0.0,
    }
    for name in MODULES:
        features["module_" + name] = 1.0 if module == name else 0.0
    features["module_other"] = 1.0 if module not in MODULES else 0.0
    return features


VECTOR = tuple(extract("x", {"reply": "x", "module": "answers", "candidates": []}).keys())
