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
    def __init__(self, entries: list[RagEntry]):
        self.entries = entries
        vocab: set[str] = set()
        for e in entries:
            vocab.update(tokenize_text(preprocess_text(e.title + " " + e.content)))
            for kw in e.keywords:
                vocab.update(tokenize_text(preprocess_text(kw)))
        self.vocabulary = sorted(vocab)       # C++ order is unspecified; cosine does not depend on it

    @classmethod
    def from_directory(cls, directory: Path) -> "RagSystem":
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
        return cls(entries)

    def similarity(self, query: str, entry: RagEntry) -> float:
        entry_text = entry.title + " " + entry.content
        for kw in entry.keywords:
            entry_text += " " + kw
        qv = text_to_tf_vector(preprocess_text(query), self.vocabulary)
        ev = text_to_tf_vector(preprocess_text(entry_text), self.vocabulary)
        return cosine_similarity(qv, ev)

    def retrieve(self, query: str, max_results: int, threshold: float,
                 location_names: list[str] | None = None, location_boost: float = 0.0) -> list[RagResult]:
        """Threshold on the raw score; rank by raw * (1 + boost) for entries mentioning a location."""
        if not query:
            return []
        locations = []
        if location_boost > 0.0:
            locations = [p for p in (preprocess_text(n) for n in location_names or []) if p]
        scored: list[tuple[float, RagResult]] = []
        for e in self.entries:
            sim = self.similarity(query, e)
            if sim < threshold:
                continue
            rank = sim
            if locations:
                text = preprocess_text(" ".join([e.title, e.content, *e.keywords]))
                if any(loc in text for loc in locations):
                    rank = sim * (1.0 + location_boost)
            scored.append((rank, RagResult(e, sim)))
        # stable; ties on rank fall back to raw score, then load order
        scored.sort(key=lambda t: (-t[0], -t[1].similarity))
        return [r for _, r in scored[:max_results]]

    @staticmethod
    def format_info(results: list[RagResult]) -> str:
        return "\n".join(f"- {r.entry.title}: {r.entry.content}" for r in results)
