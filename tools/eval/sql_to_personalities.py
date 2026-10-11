#!/usr/bin/env python3
"""Turn the personality INSERT rows in the module's SQL into the TOML file run_eval.py reads.

    sql_to_personalities.py data/sql/characters/base/2025_05_31_personality_template.sql > personalities.toml

Reads ``('KEY', 'prompt text'[, manual_only]),`` tuples (SQL '' escapes a quote, \\ a backslash). Output
is one ``KEY = "text"`` line per row, or ``KEY = { prompt = "text", manual_only = true }`` for a
manual_only row, which is the format of --personalities.
"""

from __future__ import annotations

import re
import sys

ROW = re.compile(r"\(\s*'((?:[^']|'')*)'\s*,\s*'((?:[^']|'')*)'\s*(?:,\s*(\d+)\s*)?\)")


def parse_rows(sql: str) -> list[tuple[str, str, bool]]:
    rows = []
    for key, prompt, manual in ROW.findall(sql):
        rows.append((_unescape(key), _unescape(prompt), manual == "1"))
    return rows


def _unescape(s: str) -> str:
    return re.sub(r"''|\\\\", lambda m: "'" if m.group() == "''" else "\\", s)


_TOML_ESC = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def toml_string(s: str) -> str:
    return '"' + "".join(_TOML_ESC.get(c) or (f"\\u{ord(c):04x}" if ord(c) < 0x20 or ord(c) == 0x7F else c)
                         for c in s) + '"'


def to_toml(rows: list[tuple[str, str, bool]]) -> str:
    lines = ["# Personality prompts: KEY = \"prompt text\". A key the module never assigns at random is written",
             "# KEY = { prompt = \"prompt text\", manual_only = true }."]
    for key, prompt, manual in rows:
        lines.append(f"{key} = {{ prompt = {toml_string(prompt)}, manual_only = true }}" if manual
                     else f"{key} = {toml_string(prompt)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rows: list[tuple[str, str, bool]] = []
    for path in argv[1:]:
        with open(path, encoding="utf-8") as fh:
            rows += parse_rows(fh.read())
    sys.stdout.write(to_toml(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
