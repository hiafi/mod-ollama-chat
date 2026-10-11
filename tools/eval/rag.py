"""Port of the module's retrieval: src/mod-ollama-chat_rag.cpp (``OllamaRAGSystem``).

Bag-of-words cosine similarity, no IDF. Kept deliberately literal, including
its quirks (punctuation is deleted rather than replaced by a space, so "it's"
becomes "its"; only ASCII is lower-cased and only ASCII punctuation is removed,
as in the C locale).
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass, field
from pathlib import Path

# OllamaChat.RAGIgnoreCommonWords: dropped from entries and queries when the option is on (kCommonWords in the C++).
COMMON_WORDS = frozenset(
    "a an the to of in on for and or is are was be been am i im you we he she it they me my your our his her its "
    "their this that these those with at by from as do does did can could would should will what where who how when "
    "why which there here so if not no yes ok just".split())

_PUNCT = frozenset(string.punctuation)          # std::ispunct in the C locale
_WS = " \t\n\v\f\r"                             # what ``stream >> token`` splits on


def preprocess_text(text: str) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            ch = chr(ord(ch) + 32)
        if ch in _PUNCT:
            continue
        out.append(ch)
    return "".join(out)


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch in _WS:
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


def text_to_tf_vector(text: str, vocabulary: list[str]) -> list[float]:
    freq: dict[str, int] = {}
    for tok in tokenize_text(text):
        freq[tok] = freq.get(tok, 0) + 1
    return [float(freq.get(term, 0)) for term in vocabulary]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RagEntry:
    id: str
    title: str
    content: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class RagResult:
    entry: RagEntry
    similarity: float


class RagSystem:
    def __init__(self, entries: list[RagEntry], ignore_common_words: bool = False):
        self.entries = entries
        self.ignore_common_words = ignore_common_words
        vocab: set[str] = set()
        for e in entries:
            vocab.update(self._tokens(e.title + " " + e.content))
            for kw in e.keywords:
                vocab.update(self._tokens(kw))
        self.vocabulary = sorted(vocab)       # C++ order is unspecified; cosine does not depend on it
        self._vocab_set = vocab
        # Sparse term frequencies and norms, as the scorer in the module keeps them (see retrieve()).
        self._tf = [self._entry_tf(e) for e in entries]
        self._norm = [math.sqrt(sum(v * v for v in tf.values())) for tf in self._tf]

    def _tokens(self, text: str) -> list[str]:
        toks = tokenize_text(preprocess_text(text))
        return [t for t in toks if t not in COMMON_WORDS] if self.ignore_common_words else toks

    def _entry_tf(self, entry: "RagEntry") -> dict[str, int]:
        text = entry.title + " " + entry.content
        for kw in entry.keywords:
            text += " " + kw
        tf: dict[str, int] = {}
        for tok in self._tokens(text):
            tf[tok] = tf.get(tok, 0) + 1
        return tf

    @classmethod
    def from_directory(cls, directory: Path, ignore_common_words: bool = False) -> "RagSystem":
        entries: list[RagEntry] = []
        files = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix == ".json")
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for item in data:
                eid = item.get("id", "") or ""
                content = item.get("content", "") or ""
                if not eid or not content:
                    continue
                kws = item.get("keywords")
                entries.append(RagEntry(
                    id=eid,
                    title=item.get("title", "") or "",
                    content=content,
                    keywords=[str(k) for k in kws] if isinstance(kws, list) else [],
                ))
        if not entries:
            raise ValueError(f"no RAG entries loaded from {directory}")
        return cls(entries, ignore_common_words)

    def similarity(self, query: str, entry: RagEntry) -> float:
        """Dense reference: the module's original scorer, O(vocabulary) per entry. Slow; used to check retrieve()."""
        entry_text = entry.title + " " + entry.content
        for kw in entry.keywords:
            entry_text += " " + kw
        qv = text_to_tf_vector(" ".join(self._tokens(query)), self.vocabulary)
        ev = text_to_tf_vector(" ".join(self._tokens(entry_text)), self.vocabulary)
        return cosine_similarity(qv, ev)

    def retrieve(self, query: str, max_results: int, threshold: float) -> list[RagResult]:
        if not query:
            return []
        # Sparse cosine, score-identical to similarity(): query terms outside the vocabulary are dropped, as the
        # dense vector did, and only shared terms contribute to the dot product.
        qtf: dict[str, int] = {}
        for tok in self._tokens(query):
            if tok in self._vocab_set:
                qtf[tok] = qtf.get(tok, 0) + 1
        qnorm = math.sqrt(sum(v * v for v in qtf.values()))
        results = []
        for e, tf, norm in zip(self.entries, self._tf, self._norm):
            if qnorm == 0.0 or norm == 0.0:
                sim = 0.0
            else:
                sim = sum(c * tf[t] for t, c in qtf.items() if t in tf) / (qnorm * norm)
            if sim >= threshold:
                results.append(RagResult(e, sim))
        results.sort(key=lambda r: -r.similarity)      # stable; std::sort is not, ties are arbitrary there
        return results[:max_results]

    @staticmethod
    def format_info(results: list[RagResult]) -> str:
        return "\n".join(f"- {r.entry.title}: {r.entry.content}" for r in results)
