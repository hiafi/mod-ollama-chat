#!/usr/bin/env python3
"""Build a mod-ollama-chat config from the shipped .conf.dist plus a small TOML overrides file.

    build_conf.py --dist conf/mod_ollama_chat.conf.dist --overrides my.toml --out live.conf [--check] [--diff]

Why this exists
---------------
AzerothCore loads exactly one config file per module and reads it one line per
key, so a long prompt cannot be wrapped, and appending an override at the end
of the file silently does nothing (the FIRST occurrence of a key wins). This
tool keeps your changes in a small, commented, multi-line-friendly TOML file
and rewrites only the overridden lines of the stock .conf.dist in place, so the
upstream comments and key order are kept and a module update is a re-run.

Overrides file (TOML)
---------------------
    "OllamaChat.Provider" = "anthropic"       # keys are quoted: they contain dots
    "OllamaChat.WorkerThreads" = 2
    "OllamaChat.SystemPrompt" = \"\"\"
    A long prompt, written over
    several lines.
    \"\"\"

Strings are folded to one line (a run of whitespace containing a real newline
becomes one space; at the two ends such a run is dropped, but a plain edge space is
kept and the value is written in quotes). In TOML ``"..."`` and multi-line basic strings
a ``\\n`` escape IS a real newline, so it folds to a space. To pass the two characters
backslash and n through (stock templates use them as separators), write the value as a
literal ``'''...'''`` string or as ``\\\\n``.

What is validated (nothing is written if any check fails)
---------------------------------------------------------
* every override key exists as a real line in the .dist
* the key is not duplicated in the .dist (the first one would win)
* string values contain no double quote (the config parser deletes every ``"``)
* no newline survives folding
* template placeholders: the module runs templates through ``fmt::vformat``,
  where an unknown ``{name}`` or a stray brace turns the whole prompt into the
  literal text ``[Format Error]``. Each override may only use placeholders the
  C++ actually supplies for that key.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import os
import re
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Config parsing, mirroring src/common/Configuration/Config.cpp ParseFile
# ---------------------------------------------------------------------------


@dataclass
class DistLine:
    index: int          # position in the file (0-based)
    key: str
    raw_value: str      # value text exactly as written, after the first '=' and a trim
    indent: str         # leading whitespace of the line
    key_text: str       # the key as written (whitespace around '=' is normalised on rewrite)


def _is_key_line(line: str) -> bool:
    """AzerothCore skips blank lines, lines starting with '#' or '[', and lines without '='."""
    s = line.strip()
    return bool(s) and s[0] not in "#[" and "=" in s


def scan_dist(text: str) -> tuple[list[str], dict[str, list[DistLine]]]:
    lines = text.split("\n")
    keys: dict[str, list[DistLine]] = {}
    for i, line in enumerate(lines):
        if not _is_key_line(line):
            continue
        stripped = line.strip()
        eq = stripped.index("=")
        key = stripped[:eq].strip()
        raw = stripped[eq + 1:].strip()
        indent = line[: len(line) - len(line.lstrip())]
        keys.setdefault(key, []).append(DistLine(i, key, raw, indent, key))
    return lines, keys


def conf_value(raw_value: str) -> str:
    """What the core hands to the module: the trimmed text with every double quote deleted."""
    return raw_value.replace('"', "")


def parse_conf_text(text: str) -> dict[str, str]:
    """Key -> value as AzerothCore would load it (first occurrence of a key wins)."""
    _, keys = scan_dist(text)
    return {k: conf_value(v[0].raw_value) for k, v in keys.items()}


# ---------------------------------------------------------------------------
# Value handling
# ---------------------------------------------------------------------------


def fold(value: str) -> str:
    """Fold a multi-line string to the single line the config format needs.

    * A whitespace run that contains a newline becomes one space.
    * At the two ends, only a whitespace run that contains a newline is dropped (so the
      newline after an opening triple quote and before the closing one vanish), while a
      deliberate edge space survives; ``build`` then writes such a value inside quotes.

    Mind how TOML spells the escape: in ``"..."`` and multi-line basic strings a ``\\n`` is a real
    newline, which this function folds into a space. To keep the two characters
    backslash and n (stock templates use them as separators) write ``'''...'''`` or ``\\\\n``.
    """
    for pattern in (r"\A\s+", r"\s+\Z"):
        m = re.search(pattern, value)
        if m and "\n" in m.group(0):
            value = value[: m.start()] + value[m.end():]
    return re.sub(r"\s+", lambda m: " " if "\n" in m.group(0) else m.group(0), value)


def render_value(value) -> str:
    """Override value -> text as it appears after the config parser (before quoting)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return fold(value)
    raise TypeError(f"unsupported value type {type(value).__name__}")


def load_overrides(path: Path) -> dict[str, object]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    for key, value in data.items():
        if isinstance(value, dict):
            raise ValueError(
                f"{path}: '{key}' is a table. Keys must be quoted full names such as "
                f'"OllamaChat.Provider" = "x" (an unquoted dotted key becomes a nested table).'
            )
    return data


# ---------------------------------------------------------------------------
# Placeholder validation
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass
class Token:
    kind: str           # "lit" | "field"
    text: str = ""      # literal text, or the field name
    spec: str = ""      # format spec for fields


def parse_template(template: str) -> tuple[list[Token], list[str]]:
    """Tokenise a fmt-style template. Returns (tokens, errors)."""
    tokens: list[Token] = []
    errors: list[str] = []
    lit: list[str] = []
    i, n = 0, len(template)

    def flush() -> None:
        if lit:
            tokens.append(Token("lit", "".join(lit)))
            lit.clear()

    while i < n:
        c = template[i]
        if c == "{":
            if i + 1 < n and template[i + 1] == "{":
                lit.append("{")
                i += 2
                continue
            close = template.find("}", i + 1)
            if close < 0:
                errors.append(f"unclosed '{{' at offset {i}: ...{template[i:i + 24]!r}")
                i += 1
                continue
            body = template[i + 1:close]
            name, _, spec = body.partition(":")
            if "{" in body:
                errors.append(f"nested or stray '{{' inside a replacement field at offset {i}: {template[i:close + 1]!r}")
            elif not name:
                errors.append(f"empty placeholder '{{}}' at offset {i} (the module only passes named arguments)")
            elif not _IDENT.match(name):
                errors.append(f"malformed placeholder {{{name}}} at offset {i}")
            else:
                flush()
                tokens.append(Token("field", name, spec))
            i = close + 1
        elif c == "}":
            if i + 1 < n and template[i + 1] == "}":
                lit.append("}")
                i += 2
                continue
            errors.append(f"stray '}}' at offset {i}: ...{template[max(0, i - 16):i + 8]!r}")
            i += 1
        else:
            lit.append(c)
            i += 1
    flush()
    return tokens, errors


def placeholders_of(template: str) -> tuple[set[str], list[str]]:
    tokens, errors = parse_template(template)
    return {t.text for t in tokens if t.kind == "field"}, errors


# Placeholders each key is formatted with, taken from the SafeFormat(...) call
# sites in src/. A key mapped to an empty set is never formatted (or only has
# a fixed set of arguments that it does not use), so a placeholder in it would
# reach the model literally. test_build_conf.py checks that every placeholder
# in the shipped .dist is covered by this table, so drift is caught.
_PLAYER_STATE = {"player_name", "player_class", "player_race", "player_level", "player_state", "level_gap"}
_BOT_COMMON = {
    "bot_name", "bot_level", "bot_class", "bot_race", "bot_gender", "bot_role", "bot_faction",
    "bot_area", "bot_zone", "bot_map", "bot_personality", "bot_personality_name",
}

ALLOWED_PLACEHOLDERS: dict[str, set[str]] = {
    # handler.cpp GenerateBotPrompt
    "OllamaChat.ChatPromptTemplate": {
        "bot_name", "bot_level", "bot_class", "bot_personality", "bot_personality_name",
        "player_level", "player_class", "player_name", "player_message",
        "extra_info", "chat_history", "sentiment_info",
    },
    "OllamaChat.ChatExtraInfoTemplate": {
        "bot_race", "bot_gender", "bot_role", "bot_faction", "bot_guild", "bot_group_status", "bot_gold",
        "player_race", "player_gender", "player_role", "player_faction", "player_guild",
        "player_group_status", "player_gold", "player_distance", "bot_area", "bot_zone", "bot_map",
    },
    "OllamaChat.ChatHistoryHeaderTemplate": {"player_name"},
    "OllamaChat.ChatHistoryLineTemplate": {"player_name", "player_message", "bot_reply"},
    "OllamaChat.ChatHistoryFooterTemplate": {"player_name", "player_message"},
    "OllamaChat.ChatBotSnapshotTemplate": {"combat", "group", "spells", "quests", "los", "players"},
    "OllamaChat.RAGPromptTemplate": {"rag_info"},
    "OllamaChat.EmoteReactionPromptTemplate": {
        "bot_name", "bot_level", "bot_class", "bot_race", "bot_personality", "bot_personality_name",
        "player_name", "player_class", "player_race", "emote_name",
    },
    # random.cpp BuildRandomChatterPrompt
    "OllamaChat.RandomChatterPromptTemplate": _BOT_COMMON | {"environment_info"},
    # events.cpp BuildPrompt (also used for guild events)
    "OllamaChat.EventChatterPromptTemplate": _BOT_COMMON | {
        "event_type", "event_detail", "actor_name", "sentiment_info",
    },
    # sentiment.cpp
    "OllamaChat.SentimentAnalysisPrompt": {"message"},
    "OllamaChat.SentimentPromptTemplate": {"player_name", "sentiment_value"},
    # memory.cpp (empty value = built-in default)
    "OllamaChat.Memory.PromptTemplate": {"memories"},
    "OllamaChat.Relationship.PromptTemplate": {"relationships"},
    "OllamaChat.Memory.CondensePrompt": {"bot_name", "history"},
    "OllamaChat.Relationship.UpdatePrompt": {"bot_name", "other_name", "existing", "history"},
    # topics.cpp gatherers
    "OllamaChat.EnvCommentNearbyPlayer": _PLAYER_STATE,
    "OllamaChat.EnvCommentGroupMember": {"member_name", "member_class", "member_state", "member_level"},
    "OllamaChat.EnvCommentGroupNeed": {"member_name", "need"},
    "OllamaChat.EnvCommentGuildMemberOnline": {"member_name", "member_class", "member_level"},
    "OllamaChat.EnvCommentRecentEvent": {"event_text"},
    "OllamaChat.EnvCommentCreature": {"creature_name", "creature_level", "level_delta"},
    "OllamaChat.EnvCommentCorpse": {"creature_name", "creature_level", "level_delta"},
    "OllamaChat.EnvCommentDanger": {"creature_name", "level_delta"},
    "OllamaChat.EnvCommentNamedNpc": {"npc_name", "npc_role"},
    "OllamaChat.EnvCommentVendor": {"vendor_name"},
    "OllamaChat.EnvCommentQuestgiver": {"questgiver_name", "quest_count"},
    "OllamaChat.EnvCommentGameObject": {"object_name"},
    "OllamaChat.EnvCommentZoneLandmark": {"area_name", "zone_name"},
    "OllamaChat.EnvCommentTimeOfDay": {"time_of_day"},
    "OllamaChat.EnvCommentDungeon": {"dungeon_name"},
    "OllamaChat.EnvCommentQuestObjective": {"quest_name", "objective"},
    "OllamaChat.EnvCommentUnfinishedQuest": {"quest_name"},
    "OllamaChat.EnvCommentQuestArea": {"quest_area"},
    "OllamaChat.EnvCommentEquippedItem": {"item_name"},
    "OllamaChat.EnvCommentSpell": {"spell_name", "spell_cost"},
    "OllamaChat.EnvCommentBagItem": {"item_name"},
    "OllamaChat.EnvCommentBagItemSell": {"item_name", "item_count"},
    "OllamaChat.EnvCommentBagSlots": {"bag_slots"},
    "OllamaChat.GuildEnvCommentGuildMOTD": {"guild_motd"},
    "OllamaChat.GuildEnvCommentGuildBank": {"bank_gold"},
    # Loaded by config.cpp but no gatherer reads them (dead config); the stock
    # placeholder set is all that is known.
    "OllamaChat.GuildEnvCommentGuildMember": {"member_name"},
    "OllamaChat.GuildEnvCommentGuildRank": {"guild_rank"},
    "OllamaChat.GuildEnvCommentGuildInfo": {"guild_info"},
    "OllamaChat.GuildEnvCommentGuildOnlineMembers": {"online_count", "total_count"},
    # Plain lists, picked with no formatting at all.
    "OllamaChat.SystemPrompt": set(),
    "OllamaChat.RandomChatterPromptVariations": set(),
    "OllamaChat.RandomChatterQuestionVariations": set(),
    "OllamaChat.Roleplay.PromptVariations": set(),
    "OllamaChat.Roleplay.QuestionVariations": set(),
    "OllamaChat.GuildEnvCommentGuildRaid": set(),
    "OllamaChat.GuildEnvCommentGuildEndgame": set(),
    "OllamaChat.GuildEnvCommentGuildStrategy": set(),
    "OllamaChat.GuildEnvCommentGuildGroup": set(),
    "OllamaChat.GuildEnvCommentGuildPvP": set(),
    "OllamaChat.GuildEnvCommentGuildCommunity": set(),
    "OllamaChat.DefaultPersonalityPrompt": set(),
}

# A key not in the table falls back to "the stock value's own placeholder set".


def allowed_placeholders(key: str, stock_value: str) -> tuple[set[str], str]:
    """(allowed set, how it was derived)."""
    stock_set, _ = placeholders_of(stock_value)
    if key in ALLOWED_PLACEHOLDERS:
        return set(ALLOWED_PLACEHOLDERS[key]) | stock_set, "SafeFormat call site"
    return stock_set, "stock value"


def validate_placeholders(key: str, stock_value: str, new_value: str) -> list[str]:
    errs: list[str] = []
    new_set, parse_errors = placeholders_of(new_value)
    stock_set, _ = placeholders_of(stock_value)
    allowed, source = allowed_placeholders(key, stock_value)

    if key in ALLOWED_PLACEHOLDERS and not ALLOWED_PLACEHOLDERS[key]:
        # Never run through fmt::vformat: braces are harmless, but a {name}
        # would be sent to the model verbatim.
        for name in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", new_value):
            errs.append(f"{key}: '{{{name}}}' will not be substituted: this key is not run through the "
                        f"template formatter, so the model would see the braces literally")
        return errs

    for e in parse_errors:
        errs.append(f"{key}: {e} (the whole prompt would become '[Format Error]')")
    if key not in ALLOWED_PLACEHOLDERS and not stock_set and new_set:
        errs.append(f"{key}: the stock value has no placeholders, so the module does not format it; "
                    f"{sorted(new_set)} would be sent literally (or break the format)")
        return errs
    for name in sorted(new_set - allowed):
        errs.append(f"{key}: unknown placeholder {{{name}}} (allowed, from {source}: "
                    f"{', '.join(sorted(allowed)) or 'none'}); it would turn the whole prompt into '[Format Error]'")
    return errs


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    text: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed_keys: list[str] = field(default_factory=list)
    dist_keys: list[str] = field(default_factory=list)


def _needs_quotes(dl: DistLine, new_text: str) -> bool:
    # Keep the quoting style the .dist used for that line. The parser trims the line
    # and only then deletes the quotes, so quoting is what lets a deliberate leading
    # or trailing space reach the module.
    if dl.raw_value.startswith('"') or dl.raw_value == "":
        return True
    return new_text != new_text.strip()


def build(dist_text: str, overrides: dict[str, object]) -> BuildResult:
    lines, keys = scan_dist(dist_text)
    res = BuildResult(text="", dist_keys=list(keys))

    for k, occ in keys.items():
        if len(occ) > 1 and k not in overrides:
            res.warnings.append(
                f"{k}: appears {len(occ)} times in the .dist (lines "
                f"{', '.join(str(o.index + 1) for o in occ)}); the core keeps the first")

    for key, value in overrides.items():
        if key not in keys:
            hint = ""
            commented = re.search(rf"^\s*#\s*{re.escape(key)}\s*=", dist_text, re.M)
            if commented:
                hint = (" It exists only as a commented-out example; the module has a built-in default for it. "
                        "Uncomment it in the .dist (a feat/ change) to override it with this tool.")
            res.errors.append(f"{key}: not found in the .dist - renamed or removed upstream?{hint}")
            continue
        occ = keys[key]
        if len(occ) > 1:
            res.errors.append(
                f"{key}: appears {len(occ)} times in the .dist (lines "
                f"{', '.join(str(o.index + 1) for o in occ)}); refusing to guess which one wins")
            continue
        try:
            text = render_value(value)
        except TypeError as exc:
            res.errors.append(f"{key}: {exc}")
            continue
        if isinstance(value, str):
            if '"' in text:
                res.errors.append(
                    f"{key}: value contains a double quote; the config parser deletes every '\"'. "
                    f"Use single quotes in prompts.")
                continue
            if "\n" in text or "\r" in text or "\x00" in text:
                res.errors.append(f"{key}: value still contains a newline or control character after folding")
                continue
            stock = conf_value(occ[0].raw_value)
            res.errors.extend(validate_placeholders(key, stock, text))
            if "\\n" in stock and "\\n" not in text:
                res.warnings.append(
                    f"{key}: stock value uses a literal \\n separator; yours has none - did TOML turn it into "
                    f"a newline? use '''...''' or \\\\n")
        dl = occ[0]
        out_value = f'"{text}"' if (isinstance(value, str) and _needs_quotes(dl, text)) else text
        lines[dl.index] = f"{dl.indent}{key} = {out_value}"
        res.changed_keys.append(key)

    res.text = "\n".join(lines)
    return res


def effective_config(dist_text: str, overrides: dict[str, object] | None) -> dict[str, str]:
    """The key -> value map the module would load. Raises ValueError on invalid overrides."""
    built = build(dist_text, overrides or {})
    if built.errors:
        raise ValueError("invalid overrides:\n  " + "\n  ".join(built.errors))
    return parse_conf_text(built.text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", required=True, type=Path, help="conf/mod_ollama_chat.conf.dist")
    ap.add_argument("--overrides", required=True, type=Path, help="TOML file of overrides")
    ap.add_argument("--out", required=True, type=Path, help="config file to write (the live one)")
    ap.add_argument("--check", action="store_true",
                    help="validate only, write nothing; also list .dist keys new since the last build")
    ap.add_argument("--diff", action="store_true",
                    help="print a unified diff of --out against what would be written; write nothing")
    args = ap.parse_args(argv)

    try:
        dist_text = args.dist.read_text(encoding="utf-8")
        overrides = load_overrides(args.overrides)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    res = build(dist_text, overrides)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)

    keys_file = args.out.with_name(args.out.name + ".keys")
    if args.check:
        if keys_file.exists():
            prev = set(keys_file.read_text(encoding="utf-8").split())
            new = [k for k in res.dist_keys if k not in prev]
            gone = sorted(prev - set(res.dist_keys))
            if new:
                print(f"{len(new)} key(s) in the .dist are new since the last build:")
                for k in new:
                    print(f"  + {k}")
            if gone:
                print(f"{len(gone)} key(s) from the last build are no longer in the .dist:")
                for k in gone:
                    print(f"  - {k}")
            if not new and not gone:
                print("no .dist keys added or removed since the last build")
        else:
            print(f"note: {keys_file} does not exist yet; no baseline to list new keys against")

    if res.errors:
        print(f"{len(res.errors)} error(s); nothing written:", file=sys.stderr)
        for e in res.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    if args.diff:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        diff = list(difflib.unified_diff(current.splitlines(), res.text.splitlines(),
                                         fromfile=str(args.out), tofile="(would write)", lineterm=""))
        print("\n".join(diff) if diff else "no differences")
        return 0

    if args.check:
        print(f"ok: {len(res.changed_keys)} override(s) valid against {len(res.dist_keys)} .dist keys")
        return 0

    if args.out.exists():
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.out.with_name(f"{args.out.name}.bak-{stamp}")
        backup.write_bytes(args.out.read_bytes())
        print(f"backed up {args.out} -> {backup}")
    _write_atomic(args.out, res.text)
    _write_atomic(keys_file, "\n".join(res.dist_keys) + "\n")
    print(f"wrote {args.out} ({len(res.changed_keys)} override(s) applied)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
