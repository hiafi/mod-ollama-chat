"""Port of the module's reply clean-up: ``ProcessLlmResponse`` in src/mod-ollama-chat_response.cpp.

The C++ works on ``std::string`` bytes, so this port works on ``bytes`` too:
``MaxReplyLength`` is a byte count, and the UTF-8 handling (including how
malformed sequences are dropped) only comes out the same if the bytes are what
is processed. Emote-tag extraction (``ExtractEmoteTag`` in
src/mod-ollama-chat_expression.cpp) runs inside it, because it edits the text.
"""

from __future__ import annotations

from emotes import EMOTE_NAMES

_WS = b" \t\n\r\f\v"


def _is_space(c: int) -> bool:
    return c in _WS


def _trim(s: bytes) -> bytes:
    start, end = 0, len(s)
    while start < end and _is_space(s[start]):
        start += 1
    while end > start and _is_space(s[end - 1]):
        end -= 1
    return s[start:end]


def _lower(s: bytes) -> bytes:
    return bytes(c + 32 if 65 <= c <= 90 else c for c in s)


def _is_alpha(c: int) -> bool:
    return 65 <= c <= 90 or 97 <= c <= 122


# ---------------------------------------------------------------------------
# Emote tags (expression.cpp)
# ---------------------------------------------------------------------------


def lookup_text_emote(name: bytes) -> bool:
    letters = bytes(c for c in _lower(name) if _is_alpha(c))
    if len(letters) > 4 and letters.endswith(b"ing"):
        letters = letters[:-3]
    elif len(letters) > 3 and letters.endswith(b"s"):
        letters = letters[:-1]
    if not letters:
        return False
    return letters.decode("ascii") in EMOTE_NAMES


def extract_emote_tag(response: bytes) -> tuple[bytes, str]:
    """Returns (text with gesture tags removed, first resolved emote word or '')."""
    found = ""
    r = response

    def consider(word: bytes) -> None:
        nonlocal found
        if not found and lookup_text_emote(word):
            found = word.decode("utf-8", "replace")

    # form 1: [emote:name]
    opener = b"[emote:"
    while True:
        start = r.find(opener)
        if start < 0:
            break
        close = r.find(b"]", start)
        if close < 0:
            r = r[:start]
            break
        consider(r[start + len(opener):close])
        r = r[:start] + r[close + 1:]

    # form 2: *waves*
    while True:
        start = r.find(b"*")
        if start < 0:
            break
        close = r.find(b"*", start + 1)
        if close < 0:
            r = r[:start] + r[start + 1:]
            break

        inner = r[start + 1:close]
        one_word = len(inner) > 0 and len(inner) <= 24 and b" " not in inner
        resolves = one_word and lookup_text_emote(inner)

        after_end = close + 1
        while after_end < len(r) and _is_space(r[after_end]):
            after_end += 1
        before_start = start
        while before_start > 0 and _is_space(r[before_start - 1]):
            before_start -= 1

        at_edge = before_start == 0 or after_end >= len(r)
        whole_line = before_start == 0 and after_end >= len(r)

        if whole_line and not one_word:
            r = r[:start] + r[close + 1:]
            continue

        if resolves or (one_word and at_edge):
            consider(inner)
            r = r[:start] + r[close + 1:]
        else:
            r = r[:close] + r[close + 1:]
            r = r[:start] + r[start + 1:]

    # form 3: a bare /wave
    scan_from = 0
    while True:
        slash = r.find(b"/", scan_from)
        if slash < 0:
            break
        end = slash + 1
        while end < len(r) and _is_alpha(r[end]):
            end += 1
        word = r[slash + 1:end]
        if not lookup_text_emote(word):
            scan_from = slash + 1
            continue
        consider(word)
        r = r[:slash] + r[end:]
        scan_from = slash

    r = r.lstrip(_WS).rstrip(_WS)
    return r, found


# ---------------------------------------------------------------------------
# response.cpp
# ---------------------------------------------------------------------------


def strip_think_tags(text: bytes) -> bytes:
    open_tag, close_tag = b"<think>", b"</think>"
    result = text
    while True:
        o = result.find(open_tag)
        if o < 0:
            break
        c = result.find(close_tag, o)
        if c < 0:
            result = result[:o]
            break
        result = result[:o] + result[c + len(close_tag):]
    orphan = result.find(close_tag)
    if orphan >= 0:
        result = result[orphan + len(close_tag):]
    return _trim(result)


def unwrap_quoted_reply(text: bytes) -> bytes:
    s = _trim(text)
    if len(s) < 2:
        return s
    first, last = s[0:1], s[-1:]
    if not ((first == b'"' and last == b'"') or (first == b"'" and last == b"'")):
        return s
    inner = s[1:-1]
    if first in inner:
        return s
    return _trim(inner)


def strip_speaker_prefix(text: bytes, bot_name: bytes) -> bytes:
    s = _trim(text)
    if not s:
        return s

    if s[0:1] in (b"<", b"["):
        closer = b">" if s[0:1] == b"<" else b"]"
        end = s.find(closer)
        if end >= 0 and end <= len(bot_name) + 2:
            inner = _trim(s[1:end])
            if bot_name and _lower(inner) == _lower(bot_name):
                return _trim(s[end + 1:])

    if bot_name and len(s) > len(bot_name):
        if _lower(s[:len(bot_name)]) == _lower(bot_name):
            p = len(bot_name)
            while p < len(s) and _is_space(s[p]):
                p += 1
            if p < len(s) and s[p:p + 1] in (b":", b"-"):
                return _trim(s[p + 1:])
    return s


def collapse_whitespace(text: bytes) -> bytes:
    out = bytearray()
    pending = False
    for ch in text:
        if _is_space(ch):
            pending = len(out) > 0
            continue
        if pending:
            out.append(32)
            pending = False
        out.append(ch)
    return bytes(out)


def strip_markdown(text: bytes) -> bytes:
    s = text
    while True:
        o = s.find(b"```")
        if o < 0:
            break
        c = s.find(b"```", o + 3)
        if c < 0:
            s = s[:o]
            break
        s = s[:o] + s[c + 3:]

    out = bytearray()
    at_line_start = True
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if at_line_start:
            if c in (ord("#"), ord(">")):
                i += 1
                continue
            if c in (ord("-"), ord("*"), ord("+")) and i + 1 < n and s[i + 1] == 32:
                i += 2
                continue
            if not _is_space(c):
                at_line_start = False
        if c == 10:
            at_line_start = True
            out.append(c)
            i += 1
            continue
        if c in (ord("*"), ord("_"), ord("`"), ord("~")):
            i += 1
            continue
        out.append(c)
        i += 1
    return bytes(out)


def _decode_utf8(s: bytes, i: int) -> tuple[int, int]:
    """(codepoint, next index); 0xFFFD and +1 for malformed input."""
    c = s[i]
    if c <= 0x7F:
        return c, i + 1

    def cont(off: int) -> bool:
        return i + off < len(s) and (s[i + off] & 0xC0) == 0x80

    if (c & 0xE0) == 0xC0 and cont(1):
        return ((c & 0x1F) << 6) | (s[i + 1] & 0x3F), i + 2
    if (c & 0xF0) == 0xE0 and cont(1) and cont(2):
        return ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F), i + 3
    if (c & 0xF8) == 0xF0 and cont(1) and cont(2) and cont(3):
        return (((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12)
                | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F)), i + 4
    return 0xFFFD, i + 1


def _append_utf8(out: bytearray, cp: int) -> None:
    if cp <= 0x7F:
        out.append(cp)
    elif cp <= 0x7FF:
        out.append(0xC0 | (cp >> 6))
        out.append(0x80 | (cp & 0x3F))
    elif cp <= 0xFFFF:
        out.append(0xE0 | (cp >> 12))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))


def strip_decorative_unicode(text: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(text):
        cp, i = _decode_utf8(text, i)
        if cp in (0x2018, 0x2019, 0x201B):
            out.append(ord("'"))
            continue
        if cp in (0x201C, 0x201D, 0x201F):
            out.append(ord('"'))
            continue
        if cp in (0x2013, 0x2014, 0x2212):
            out.append(ord("-"))
            continue
        if cp == 0x2026:
            out += b"..."
            continue
        if cp in (0x00A0, 0x2007, 0x202F):
            out.append(32)
            continue
        if cp == 0xFFFD:
            continue
        if cp <= 0x7F or 0xA0 <= cp <= 0x24F:
            _append_utf8(out, cp)
    return bytes(out)


def clamp_reply_length(text: bytes, max_len: int) -> bytes:
    if max_len == 0 or len(text) <= max_len:
        return text
    cut = max_len
    while cut > 0 and (text[cut] & 0xC0) == 0x80:
        cut -= 1
    head = text[:cut]

    min_sentence = len(head) * 2 // 3 if len(head) > 40 else 0
    best = -1
    i = len(head)
    while i > min_sentence:
        if head[i - 1] in (ord("."), ord("!"), ord("?")):
            best = i
            break
        i -= 1
    if best >= 0:
        return _trim(head[:best])

    last_space = head.rfind(b" ")
    if last_space >= 0 and last_space > len(head) // 2:
        head = head[:last_space]
    return _trim(head)


def process_llm_response(raw: str, bot_name: str, *, strip_markdown_on: bool = True,
                         strip_decorative_on: bool = True, max_reply_length: int = 240) -> tuple[str, str]:
    """Returns (processed text, emote word or ''). Empty string means the reply would be dropped."""
    if not raw:
        return "", ""
    s = raw.encode("utf-8")

    s = strip_think_tags(s)
    if not s:
        return "", ""

    s, emote = extract_emote_tag(s)

    if strip_markdown_on:
        s = strip_markdown(s)
    s = collapse_whitespace(s)
    s = unwrap_quoted_reply(s)
    s = strip_speaker_prefix(s, bot_name.encode("utf-8"))
    if strip_decorative_on:
        s = strip_decorative_unicode(s)
    s = unwrap_quoted_reply(s)
    s = collapse_whitespace(s)
    s = clamp_reply_length(s, max_reply_length)
    return _trim(s).decode("utf-8", "replace"), emote
