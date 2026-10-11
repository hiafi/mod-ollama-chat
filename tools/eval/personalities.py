"""Load a --personalities TOML file: KEY -> (prompt, manual_only).

Two value forms are accepted::

    CASUAL = "Relaxed, types in lowercase..."                                  # manual_only = false
    GLITCHED_AI = { prompt = "Respond in fragmented...", manual_only = true }  # inline table

The old trailing ``# manual_only`` comment is invisible to a TOML parser and is not read.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

Personalities = dict[str, tuple[str, bool]]


def parse_personalities(data: dict, where: str = "personalities") -> Personalities:
    out: Personalities = {}
    for key, value in data.items():
        if isinstance(value, str):
            out[key] = (value, False)
        elif isinstance(value, dict):
            unknown = sorted(set(value) - {"prompt", "manual_only"})
            if unknown:
                raise ValueError(f"{where}: personality '{key}' has unknown field(s): {', '.join(unknown)}")
            if "prompt" not in value:
                raise ValueError(f"{where}: personality '{key}' is a table without 'prompt'")
            if not isinstance(value["prompt"], str):
                raise ValueError(f"{where}: personality '{key}': 'prompt' must be a string")
            manual = value.get("manual_only", False)
            if not isinstance(manual, bool):
                raise ValueError(f"{where}: personality '{key}': 'manual_only' must be true or false")
            out[key] = (value["prompt"], manual)
        else:
            raise ValueError(f"{where}: personality '{key}' must be a string or a table with 'prompt', "
                             f"not {type(value).__name__}")
    return out


def load_personalities(path: Path) -> Personalities:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return parse_personalities(data, str(path))


def prompts_only(p: Personalities) -> dict[str, str]:
    return {k: v[0] for k, v in p.items()}
