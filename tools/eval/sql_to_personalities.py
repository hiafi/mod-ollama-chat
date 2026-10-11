#!/usr/bin/env python3
"""Turn the personality INSERT rows in the module's SQL into the TOML file run_eval.py reads.

    sql_to_personalities.py data/sql/characters/base/2025_05_31_personality_template.sql > personalities.toml

Reads ``('KEY', 'prompt text'),`` tuples (SQL '' escapes a quote). Output is one
``KEY = "text"`` line per row, which is the format of --personalities.
"""

from __future__ import annotations

import re
import sys

ROW = re.compile(r"\(\s*'((?:[^']|'')*)'\s*,\s*'((?:[^']|'')*)'\s*(?:,\s*(\d+)\s*)?\)")


def parse_rows(sql: str) -> list[tuple[str, str, bool]]:
    rows = []
    for key, prompt, manual in ROW.findall(sql):
        rows.append((key.replace("''", "'"), prompt.replace("''", "'"), manual == "1"))
    return rows


def toml_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_toml(rows: list[tuple[str, str, bool]]) -> str:
    lines = ["# Personality prompts: KEY = \"prompt text\". A trailing '# manual_only' marks keys the module",
             "# never assigns at random."]
    for key, prompt, manual in rows:
        lines.append(f"{key} = {toml_string(prompt)}" + ("  # manual_only" if manual else ""))
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
