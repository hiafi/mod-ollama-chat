"""Automatic metrics over processed replies. They catch regressions; the user is the judge."""

from __future__ import annotations

import difflib
import re
import statistics
from collections import Counter
from pathlib import Path

DEFAULT_OPENERS = ["lol", "lmao", "yo", "dude", "haha", "bro"]


def load_banned_terms(path: Path | None) -> list[str]:
    """One term (word or phrase) per line; '#' starts a comment, blank lines are skipped."""
    if path is None:
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"(^|\s)#.*$", "", line).strip()
        if line:
            terms.append(line)
    return terms


def _banned_regex(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(p) for p in term.split()) + r"(?!\w)", re.I)


def first_word(text: str) -> str:
    m = re.search(r"[^\W_]+(?:'[^\W_]+)*", text.lower())
    return m.group(0) if m else ""


# A reply that shows the bot did not understand a plain callout or explains itself instead of answering.
CONFUSED_RE = re.compile(
    r"^\W*(?:what|huh|wut|wat|hm+|eh|um+)\b[^a-z]*(?:\?|$)|\bwdym\b|what do you mean|what does that mean|"
    r"i (?:don'?t|do not) (?:get|understand)|not sure what you mean|don'?t need mana",
    re.I)


def compute(replies: list[str], failures: int, openers: list[str], banned: list[str], skipped: int = 0) -> dict:
    """``replies`` are the non-empty processed replies of one variant, in run order. ``skipped`` counts the samples
    where the model answered with the skip token: no line, so not in ``replies`` and not a failure."""
    n = len(replies)
    opener_re = re.compile(r"^\W*(?:" + "|".join(re.escape(o) for o in openers) + r")\b", re.I) if openers else None
    slang = sum(1 for r in replies if opener_re and opener_re.search(r))
    words = [len(r.split()) for r in replies]

    banned_counter: Counter = Counter()
    for term in banned:
        rx = _banned_regex(term)
        for r in replies:
            hits = len(rx.findall(r))
            if hits:
                banned_counter[term] += hits

    dups = 0
    for i, r in enumerate(replies):
        if any(difflib.SequenceMatcher(None, r.lower(), prev.lower()).ratio() >= 0.8 for prev in replies[:i]):
            dups += 1

    return {
        "replies": n,
        "slang_open_pct": round(100.0 * slang / n, 1) if n else 0.0,
        "distinct_opener_ratio": round(len({first_word(r) for r in replies}) / n, 3) if n else 0.0,
        "words_mean": round(statistics.fmean(words), 1) if n else 0.0,
        "words_stdev": round(statistics.pstdev(words), 1) if n > 1 else 0.0,
        "banned_hits": sum(banned_counter.values()),
        "banned_top": banned_counter.most_common(5),
        "near_dup_pct": round(100.0 * dups / n, 1) if n else 0.0,
        "confused_pct": round(100.0 * sum(1 for r in replies if CONFUSED_RE.search(r)) / n, 1) if n else 0.0,
        "skipped": skipped,
        "skip_pct": round(100.0 * skipped / (n + skipped), 1) if (n + skipped) else 0.0,
        "failures": failures,
    }
