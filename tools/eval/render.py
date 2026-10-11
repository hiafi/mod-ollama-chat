"""Prompt renderer: builds the text the module would send, from a config and a case.

Mirrors, function by function:

* reply  - ``GenerateBotPrompt``            (src/mod-ollama-chat_handler.cpp)
           ``GetBotHistoryPrompt``           (same file)
           ``GetSentimentPromptAddition``    (src/mod-ollama-chat_sentiment.cpp)
* random - ``BuildRandomChatterPrompt``      (src/mod-ollama-chat_random.cpp)
           the topic text (``Topics_Pick``)  (src/mod-ollama-chat_topics.cpp)
* event  - ``OllamaBotEventChatter::BuildPrompt`` (src/mod-ollama-chat_events.cpp)

Anything the C++ reads from a live ``Player`` comes from the case file instead.
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from build_conf import parse_template
from rag import RagSystem


class CaseError(Exception):
    """A case (or the config) cannot be rendered. Fatal for the run: fix the input."""


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()
FORMAT_ERROR = "[Format Error]"


# ---------------------------------------------------------------------------
# Config access, with the C++ defaults for keys the .dist might not carry
# ---------------------------------------------------------------------------


class Config:
    def __init__(self, values: Mapping[str, str]):
        self.values = dict(values)

    def raw(self, key: str, default: str | None = None) -> str:
        v = self.values.get("OllamaChat." + key)
        if v is None:
            if default is None:
                raise CaseError(f"config key OllamaChat.{key} is missing from the .dist")
            return default
        return v

    def text(self, key: str, default: str | None = None) -> str:
        return self.raw(key, default)

    def boolean(self, key: str, default: bool) -> bool:
        v = self.values.get("OllamaChat." + key)
        if v is None or v == "":
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    def integer(self, key: str, default: int) -> int:
        v = self.values.get("OllamaChat." + key)
        if v is None or v.strip() == "":
            return default
        try:
            return int(float(v))
        except ValueError:
            return default

    def number(self, key: str, default: float) -> float:
        v = self.values.get("OllamaChat." + key)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def pipe_list(self, key: str) -> list[str]:
        """std::getline split on '|', empty tokens dropped, NOT trimmed (config.cpp variations)."""
        return [t for t in self.values.get("OllamaChat." + key, "").split("|") if t]

    def env_list(self, key: str) -> list[str]:
        """LoadEnvCommentVector: split on '|', trimmed, empties dropped."""
        return [t.strip(" \t\r\n") for t in self.values.get("OllamaChat." + key, "").split("|")
                if t.strip(" \t\r\n")]


# C++ fallbacks (config.cpp) used when the config value is empty.
DEFAULT_MEMORY_TEMPLATE = " Things you remember:\n{memories}"
DEFAULT_RELATIONSHIP_TEMPLATE = " How you feel about people you know:\n{relationships}"
DEFAULT_SENTIMENT_TEMPLATE = ("Your relationship sentiment with {player_name} is {sentiment_value} "
                              "(0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone and response.")


# ---------------------------------------------------------------------------
# fmt::vformat equivalent
# ---------------------------------------------------------------------------


def f32_repr(x: float) -> str:
    """How fmt prints a float: shortest text that round-trips as float32; integral values without '.0'."""
    packed = struct.unpack("f", struct.pack("f", float(x)))[0]
    for digits in range(1, 10):
        s = f"{packed:.{digits}g}"
        if struct.unpack("f", struct.pack("f", float(s)))[0] == packed:
            break
    if "e" in s:
        return s
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _stringify(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f32_repr(v)
    return str(v)


def safe_format(template: str, args: Mapping[str, Any], where: str) -> str:
    """SafeFormat. ``args`` holds every name the C++ call site passes; MISSING values are case gaps."""
    tokens, errors = parse_template(template)
    if errors:
        return FORMAT_ERROR
    out: list[str] = []
    for tok in tokens:
        if tok.kind == "lit":
            out.append(tok.text)
            continue
        if tok.text not in args:
            return FORMAT_ERROR           # unknown {name}: fmt throws, SafeFormat returns this
        value = args[tok.text]
        if value is MISSING:
            raise CaseError(f"{where}: placeholder {{{tok.text}}} needs a value the case does not provide")
        if tok.spec:
            try:
                out.append(format(value, tok.spec))
            except (ValueError, TypeError):
                return FORMAT_ERROR
        else:
            out.append(_stringify(value))
    return "".join(out)


# ---------------------------------------------------------------------------
# Case access
# ---------------------------------------------------------------------------


class CaseView:
    def __init__(self, case: Mapping[str, Any], cid: str):
        self.case = case
        self.id = cid

    def section(self, name: str) -> Mapping[str, Any]:
        sec = self.case.get(name)
        if not isinstance(sec, dict):
            raise CaseError(f"case '{self.id}': needs a [{name}] table")
        return sec

    def args(self, section: str, names: list[str], prefix: str, *, optional: bool = False) -> dict[str, Any]:
        sec = self.case.get(section)
        sec = sec if isinstance(sec, dict) else {}
        out: dict[str, Any] = {}
        for n in names:
            out[prefix + n] = sec.get(n, MISSING)
        return out


def rng_for(case_id: str, sample: int, purpose: str) -> random.Random:
    """Seeded by case id + sample index (+ what it is used for), identical for baseline and candidate."""
    digest = hashlib.sha256(f"{case_id}|{sample}|{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# Optional blocks
# ---------------------------------------------------------------------------

_GESTURE_SAMPLES = [
    ["wave", "bow", "salute", "greet", "thank"],
    ["nod", "no", "shrug", "disagree", "doubt"],
    ["laugh", "chuckle", "giggle", "cheer", "applaud"],
    ["cry", "mourn", "plead", "cower", "blush"],
    ["angry", "growl", "roar", "taunt", "rude"],
    ["ponder", "confused", "curious", "point", "gasp"],
    ["pray", "drink", "flex", "victory", "toast"],
]


def build_gesture_prompt(cfg: Config, rng: random.Random) -> str:
    """Expression_BuildGesturePrompt. Same logic; the dice come from a seeded RNG, not urand."""
    if not cfg.boolean("EnableBotEmotes", True) or cfg.integer("GestureChance", 40) <= 0:
        return ""
    if rng.randint(1, 100) > cfg.integer("GestureChance", 40):
        return ""
    order = list(range(len(_GESTURE_SAMPLES)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    take = min(4, len(order))
    tags = ""
    for i in range(take):
        bucket = _GESTURE_SAMPLES[order[i]]
        name = bucket[rng.randint(0, len(bucket) - 1)]
        if tags:
            tags += " or " if i + 1 == take else ", "
        tags += f"[emote:{name}]"
    return (" You may end your reply with a single gesture tag such as " + tags +
            " when one genuinely fits -- any common emote name works. Omit it otherwise.")


def estimate_tokens(text: str) -> int:
    return (len(text.encode("utf-8")) + 3) // 4


def build_memory_section(cfg: Config, view: CaseView, *, about_name: str | None) -> str:
    """Memory_BuildPromptSection. The case supplies ``memories`` and ``relationships`` (both optional)."""
    if not cfg.boolean("Memory.Enable", True):
        return ""
    memories = view.case.get("memories") or []
    relationships = view.case.get("relationships") or []
    if not memories and not relationships:
        return ""          # no per-bot state: the C++ returns ""
    out = ""

    if cfg.boolean("Relationship.Enable", True) and relationships:
        rels = [r for r in relationships if r.get("description")]
        # whoever we are talking to first; the rest keep the order given (newest first by convention)
        rels.sort(key=lambda r: 0 if about_name and r.get("name") == about_name else 1)
        max_rel = cfg.integer("Relationship.MaxPerPrompt", 3)
        lines = ""
        taken = 0
        for r in rels:
            if max_rel > 0 and taken >= max_rel:
                break
            lines += f" - {r['name']}: {r['description']}\n"
            taken += 1
        tpl = cfg.text("Relationship.PromptTemplate", "") or DEFAULT_RELATIONSHIP_TEMPLATE
        if lines:
            out += safe_format(tpl, {"relationships": lines}, f"case '{view.id}' Relationship.PromptTemplate")

    tpl = cfg.text("Memory.PromptTemplate", "") or DEFAULT_MEMORY_TEMPLATE
    if memories:
        ordered = sorted(memories, key=lambda m: -int(m.get("importance", 0)))
        budget = cfg.integer("Memory.PromptTokenBudget", 400)
        used = 0
        lines = ""
        for m in ordered:
            cost = estimate_tokens(m["text"]) + 4
            if budget > 0 and used + cost > budget:
                continue
            lines += f" - {m['text']}\n"
            used += cost
        if lines:
            out += safe_format(tpl, {"memories": lines}, f"case '{view.id}' Memory.PromptTemplate")
    return out


def build_roleplay_voice(cfg: Config, view: CaseView) -> str:
    if not cfg.boolean("Roleplay.Enable", False):
        return ""
    voice = view.case.get("roleplay_voice")
    if not isinstance(voice, str):
        raise CaseError(f"case '{view.id}': Roleplay.Enable is on, so the case must give the text the "
                        f"module would append as roleplay_voice (not re-implemented here)")
    return voice


def personality_for(cfg: Config, view: CaseView, personalities: Mapping[str, str]) -> tuple[str, str]:
    """(personality name, personality prompt) as GetBotPersonality + GetPersonalityPromptAddition give them."""
    default_prompt = cfg.text("DefaultPersonalityPrompt", "")
    if not cfg.boolean("EnableRPPersonalities", False):
        name = "default"
    else:
        bot = view.case.get("bot") or {}
        name = bot.get("personality", "default")
        if name != "default" and name not in personalities:
            raise CaseError(f"case '{view.id}': personality '{name}' is not in the personalities file")
    return name, personalities.get(name, default_prompt)


def sentiment_info(cfg: Config, view: CaseView, player_name: str | None) -> str:
    if not cfg.boolean("EnableSentimentTracking", True) or not player_name:
        return ""
    tpl = cfg.text("SentimentPromptTemplate", DEFAULT_SENTIMENT_TEMPLATE) or ""
    if not tpl:
        return ""
    value = view.case.get("sentiment", cfg.number("SentimentDefaultValue", 0.5))
    return safe_format(tpl, {"player_name": player_name, "sentiment_value": float(value)},
                       f"case '{view.id}' SentimentPromptTemplate")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class Rendered:
    prompt: str
    system_prompt: str
    path: str
    rag: list[dict] = field(default_factory=list)       # [{"id":..., "score":...}]
    summary: str = ""
    format_error: bool = False


_BOT_FIELDS = ["name", "level", "class", "race", "gender", "role", "faction", "area", "zone", "map"]


def _bot_args(view: CaseView, personality: tuple[str, str]) -> dict[str, Any]:
    args = view.args("bot", _BOT_FIELDS, "bot_")
    args["bot_personality"] = personality[1]
    args["bot_personality_name"] = personality[0]
    return args


def _summary(view: CaseView) -> str:
    bot = view.case.get("bot") or {}
    where = bot.get("area") or bot.get("zone") or "?"
    return (f"{bot.get('name', '?')}, level {bot.get('level', '?')} {bot.get('race', '')} {bot.get('class', '')}"
            f" ({bot.get('faction', '?')}) in {where}").replace("  ", " ")


def render_reply(cfg: Config, view: CaseView, personalities: Mapping[str, str],
                 rag: RagSystem | None, sample: int) -> Rendered:
    where = f"case '{view.id}'"
    bot = view.section("bot")
    player = view.section("player")
    message = view.case.get("message")
    if not isinstance(message, str) or not message:
        raise CaseError(f"{where}: reply cases need a 'message'")

    personality = personality_for(cfg, view, personalities)
    player_name = player.get("name", MISSING)
    if player_name is MISSING:
        raise CaseError(f"{where}: [player] needs a name")

    # chat history (GetBotHistoryPrompt): nothing at all, footer included, without prior exchanges
    chat_history = ""
    history = view.case.get("history") or []
    if cfg.boolean("EnableChatHistory", True) and history:
        history = history[-max(1, cfg.integer("MaxConversationHistory", 5)):]
        chat_history = safe_format(cfg.text("ChatHistoryHeaderTemplate"), {"player_name": player_name},
                                   f"{where} ChatHistoryHeaderTemplate")
        for h in history:
            chat_history += safe_format(
                cfg.text("ChatHistoryLineTemplate"),
                {"player_name": player_name, "player_message": h.get("player", MISSING),
                 "bot_reply": h.get("bot", MISSING)}, f"{where} ChatHistoryLineTemplate")
        chat_history += safe_format(cfg.text("ChatHistoryFooterTemplate"),
                                    {"player_name": player_name, "player_message": message},
                                    f"{where} ChatHistoryFooterTemplate")

    sent = sentiment_info(cfg, view, player_name)

    rag_info = ""
    rag_hits: list[dict] = []
    if cfg.boolean("EnableRAG", False):
        if rag is None:
            raise CaseError("EnableRAG is 1 but no RAG data could be loaded (use --rag-dir or fix RAGDataPath)")
        results = rag.retrieve(message, cfg.integer("RAGMaxRetrievedItems", 3),
                               cfg.number("RAGSimilarityThreshold", 0.3))
        rag_hits = [{"id": r.entry.id, "score": round(r.similarity, 4)} for r in results]
        content = rag.format_info(results)
        if content:
            rag_info = safe_format(cfg.text("RAGPromptTemplate"), {"rag_info": content},
                                   f"{where} RAGPromptTemplate")

    extra_names = ["race", "gender", "role", "faction", "guild", "group_status", "gold"]
    extra_args = view.args("bot", extra_names, "bot_")
    extra_args.update(view.args("player", extra_names + ["distance"], "player_"))
    extra_args.update(view.args("bot", ["area", "zone", "map"], "bot_"))
    extra_info = safe_format(cfg.text("ChatExtraInfoTemplate"), extra_args, f"{where} ChatExtraInfoTemplate")

    chat_args = view.args("bot", ["name", "level", "class"], "bot_")
    chat_args.update({
        "bot_personality": personality[1], "bot_personality_name": personality[0],
        "player_level": player.get("level", MISSING), "player_class": player.get("class", MISSING),
        "player_name": player_name, "player_message": message,
        "extra_info": extra_info, "chat_history": chat_history, "sentiment_info": sent,
    })
    prompt = safe_format(cfg.text("ChatPromptTemplate"), chat_args, f"{where} ChatPromptTemplate")

    if rag_info:
        prompt += rag_info + "\n"
    if cfg.boolean("EnableChatBotSnapshotTemplate", False):
        snap = view.case.get("snapshot")
        if not isinstance(snap, str):
            raise CaseError(f"{where}: EnableChatBotSnapshotTemplate is on, so the case needs a 'snapshot' text")
        prompt += snap
    prompt += build_memory_section(cfg, view, about_name=player_name)
    prompt += build_roleplay_voice(cfg, view)
    prompt += build_gesture_prompt(cfg, rng_for(view.id, sample, "gesture"))

    return Rendered(prompt, cfg.text("SystemPrompt", ""), "reply", rag_hits,
                    f"{_summary(view)}; {player_name} says: {message!r}", prompt.startswith(FORMAT_ERROR))


def pick_env_text(cfg: Config, view: CaseView, sample: int) -> str:
    rnd = view.section("random")
    direct = rnd.get("environment_info")
    if isinstance(direct, str):
        return direct
    tpl_key = rnd.get("env_template")
    if not isinstance(tpl_key, str):
        raise CaseError(f"case '{view.id}': [random] needs environment_info or env_template")
    short = tpl_key.removeprefix("OllamaChat.")
    options = cfg.env_list(short)
    if not options:
        raise CaseError(f"case '{view.id}': {tpl_key} is empty or unknown in the config")
    chosen = options[rng_for(view.id, sample, "env").randrange(len(options))]
    env_vars = rnd.get("env_vars") or {}
    tokens, errors = parse_template(chosen)
    if errors:
        return FORMAT_ERROR
    args = {t.text: env_vars.get(t.text, MISSING) for t in tokens if t.kind == "field"}
    return safe_format(chosen, args, f"case '{view.id}' {tpl_key}")


def pick_variation(cfg: Config, view: CaseView, sample: int) -> str:
    rnd = view.section("random")
    explicit = rnd.get("variation")
    if isinstance(explicit, str):
        return explicit
    roleplay_lists = (cfg.boolean("Roleplay.Enable", False) and cfg.integer("Roleplay.Strictness", 1) >= 2
                      and bool(cfg.pipe_list("Roleplay.PromptVariations")))
    statements = cfg.pipe_list("Roleplay.PromptVariations" if roleplay_lists else "RandomChatterPromptVariations")
    questions = cfg.pipe_list("Roleplay.QuestionVariations" if roleplay_lists else "RandomChatterQuestionVariations")
    rng = rng_for(view.id, sample, "variation")
    roll, pick = rng.randint(0, 99), rng.random()
    if statements and questions:
        chosen = questions if roll < cfg.integer("RandomChatterQuestionChance", 35) else statements
    else:
        chosen = statements or questions
    if not chosen:
        return ""
    return chosen[min(int(pick * len(chosen)), len(chosen) - 1)]


def render_random(cfg: Config, view: CaseView, personalities: Mapping[str, str],
                  rag: RagSystem | None, sample: int) -> Rendered:
    where = f"case '{view.id}'"
    view.section("bot")
    personality = personality_for(cfg, view, personalities)
    args = _bot_args(view, personality)
    args["environment_info"] = pick_env_text(cfg, view, sample)
    prompt = safe_format(cfg.text("RandomChatterPromptTemplate"), args, f"{where} RandomChatterPromptTemplate")

    variation = pick_variation(cfg, view, sample)
    if variation:
        prompt += " " + variation

    prompt += build_memory_section(cfg, view, about_name=None)
    prompt += build_roleplay_voice(cfg, view)
    prompt += build_gesture_prompt(cfg, rng_for(view.id, sample, "gesture"))
    return Rendered(prompt, cfg.text("SystemPrompt", ""), "random", [],
                    f"{_summary(view)}; topic: {args['environment_info']}", prompt.startswith(FORMAT_ERROR))


def render_event(cfg: Config, view: CaseView, personalities: Mapping[str, str],
                 rag: RagSystem | None, sample: int) -> Rendered:
    where = f"case '{view.id}'"
    view.section("bot")
    ev = view.section("event")
    personality = personality_for(cfg, view, personalities)
    actor = ev.get("actor_name", MISSING)
    args = _bot_args(view, personality)
    args.update({
        "event_type": ev.get("event_type", MISSING),
        "event_detail": ev.get("event_detail", MISSING),
        "actor_name": actor,
        "sentiment_info": sentiment_info(cfg, view, actor if isinstance(actor, str) else None),
    })
    prompt = safe_format(cfg.text("EventChatterPromptTemplate"), args, f"{where} EventChatterPromptTemplate")
    prompt += build_memory_section(cfg, view, about_name=None)
    prompt += build_roleplay_voice(cfg, view)
    prompt += build_gesture_prompt(cfg, rng_for(view.id, sample, "gesture"))
    return Rendered(prompt, cfg.text("SystemPrompt", ""), "event", [],
                    f"{_summary(view)}; event: {ev.get('actor_name')} {ev.get('event_type')} {ev.get('event_detail')}",
                    prompt.startswith(FORMAT_ERROR))


def render_case(cfg: Config, case: Mapping[str, Any], personalities: Mapping[str, str],
                rag: RagSystem | None, sample: int) -> Rendered:
    cid = case.get("id")
    if not isinstance(cid, str) or not cid:
        raise CaseError("a case file has no 'id'")
    view = CaseView(case, cid)
    path = case.get("path")
    if path == "reply":
        return render_reply(cfg, view, personalities, rag, sample)
    if path == "random":
        return render_random(cfg, view, personalities, rag, sample)
    if path == "event":
        return render_event(cfg, view, personalities, rag, sample)
    raise CaseError(f"case '{cid}': path must be reply, random or event (got {path!r})")


def load_rag(cfg: Config, rag_dir: Path | None, module_root: Path) -> RagSystem | None:
    if rag_dir is not None:
        return RagSystem.from_directory(rag_dir)
    if not cfg.boolean("EnableRAG", False):
        return None
    path = Path(cfg.text("RAGDataPath", "rag/"))
    if not path.is_absolute():
        path = (module_root / path)
    if not path.is_dir():
        # the stock value walks up out of an install tree; fall back to the module's own data/rag
        fallback = module_root / "data" / "rag"
        if fallback.is_dir():
            path = fallback
    return RagSystem.from_directory(path)
