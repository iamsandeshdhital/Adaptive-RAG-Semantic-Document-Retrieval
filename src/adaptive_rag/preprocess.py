"""Text preprocessing for lexical retrieval.

Only the sparse retriever uses this module. Dense retrieval feeds raw text to
the sentence-embedding model, because the model was trained on natural text and
stripping stopwords from its input would remove signal it expects.

The stemmer is deliberately a short list of suffix rules rather than a full
Porter implementation. It is enough to make plurals and simple verb inflections
match, it is easy to explain in a report, and it adds no dependency.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

#: A compact English stopword list. Terms that carry retrieval signal in this
#: corpus (such as "not" or numerals) are intentionally absent.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are as at
    be because been before being below between both but by
    can could did do does doing down during
    each few for from further
    had has have having he her here hers herself him himself his how
    i if in into is it its itself
    just me more most my myself
    no nor now of off on once only or other ought our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then
    there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves
    """.split()
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


@lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    """Strip a few common English suffixes.

    The rules are applied in stages -- plural, then verb inflection, then a
    small set of derivational endings -- and the result finally has any silent
    trailing "e" removed. That last step is what makes the inflected and
    uninflected forms of a word agree: without it "waves" would reduce to
    "wave" while "wave" stayed as it was, and the two would never match.

    Short tokens are left alone so that words such as "gas" are not mangled
    into something that matches nothing.
    """
    if len(token) <= 3:
        return token

    word = token

    # Plurals.
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("es") and len(word) > 4 and word[-3] in "sxzh":
        # A sibilant stem takes "-es", so both letters belong to the suffix:
        # "gases" -> "gas", "boxes" -> "box".
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]

    # Verb inflection.
    if word.endswith("ied") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]

    # A few derivational endings.
    if word.endswith("ally") and len(word) > 6:
        word = word[:-2]
    elif word.endswith("ly") and len(word) > 4:
        word = word[:-2]
    if word.endswith("ness") and len(word) > 6:
        word = word[:-4]
    elif word.endswith("ment") and len(word) > 6:
        word = word[:-4]
    elif word.endswith("ion") and len(word) > 6 and word[-4] in "st":
        # "eruption" -> "erupt", so a question about eruptions matches a
        # passage that says a volcano erupts.
        word = word[:-3]

    # Undo a consonant doubled before "-ing" or "-ed", as in "stopping".
    if (
        len(word) >= 4
        and word[-1] == word[-2]
        and word[-1] not in "aeiouls"
    ):
        word = word[:-1]

    # Normalise a silent final "e" so that "wave" and "waves" agree.
    if len(word) >= 5 and word.endswith("e"):
        word = word[:-1]

    return word or token


def preprocess(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
) -> list[str]:
    """Turn raw text into the token list used to build the BM25 index."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if apply_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences on terminal punctuation.

    Abbreviations are handled well enough for this corpus by refusing to split
    after a single capital letter followed by a full stop, which covers initials
    such as "J. F. Kennedy".
    """
    flattened = " ".join(text.split())
    if not flattened:
        return []
    protected = re.sub(r"\b([A-Z])\.\s", r"\1<DOT> ", flattened)
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def contains_all(tokens: Iterable[str], required: Iterable[str]) -> bool:
    """True when every required token appears in ``tokens`` after stemming."""
    pool = {stem(t) for t in tokens}
    return all(stem(t) in pool for t in required)
