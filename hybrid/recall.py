"""A recall tier in plain Python: answer from logged question/answer pairs when the question
has been seen before in other words. Standard library only.

This is the "compile the dense head of the traffic" step: a strong model's past answers are kept
as a table, a TF-IDF match finds the nearest logged question, and the tier answers only when the
match is close enough. Otherwise it abstains and the turn moves on.

    index = Recall.load()                       # hybrid/recall_index.json
    hit = index.lookup("what is camouflage?")   # {"reply", "similarity", "matched"} or None
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.path.join(HERE, "recall_index.json")
WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")
# Only the emptiest words are dropped: "what", "how", "why", "best" carry the question's intent.
STOP = frozenset("a an the and or but to of in on at by for with from as is are was were be been am do does did "
                 "i me my we you your it its this that there here can could would should will im ive id "
                 "hi hello hey please thanks some any about so just really".split())


def _stem(word):
    word = word.replace("'", "")
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def terms(text):
    """Stemmed words plus adjacent pairs, so word order counts a little."""
    words = [_stem(w) for w in WORD.findall(text.lower()) if w not in STOP]
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


class Recall:
    def __init__(self, pairs):
        """pairs: [(question, answer)]"""
        self.pairs = list(pairs)
        counts = [Counter(terms(q)) for q, _ in self.pairs]
        df = Counter(t for c in counts for t in c)
        n = max(len(self.pairs), 1)
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.default_idf = math.log(n + 1) + 1
        self.postings = defaultdict(list)          # term -> [(doc, weight)]
        for doc, c in enumerate(counts):
            vec = {t: (1 + math.log(f)) * self.idf[t] for t, f in c.items()}
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            for t, w in vec.items():
                self.postings[t].append((doc, w / norm))

    def lookup(self, text, skip=None):
        """The closest logged question: {"reply", "similarity", "matched", "doc"}, or None.
        `skip` leaves one logged row out (used to calibrate on the log itself)."""
        c = Counter(terms(text))
        vec = {t: (1 + math.log(f)) * self.idf.get(t, self.default_idf) for t, f in c.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        scores = defaultdict(float)
        for t, w in vec.items():
            for doc, dw in self.postings.get(t, ()):
                if doc != skip:
                    scores[doc] += (w / norm) * dw
        if not scores:
            return None
        doc = max(scores, key=scores.get)
        return {"reply": self.pairs[doc][1], "similarity": scores[doc], "matched": self.pairs[doc][0], "doc": doc}

    def save(self, path=DEFAULT_INDEX):
        with open(path, "w") as handle:
            json.dump({"pairs": self.pairs}, handle)

    @classmethod
    def load(cls, path=DEFAULT_INDEX):
        with open(path) as handle:
            return cls(json.load(handle)["pairs"])
