# Prompt eval harness

`run_eval.py` renders each case through the same templates, placeholder names, separators and composition order as
the C++, calls the configured provider with the module's request parameters, applies the module's reply clean-up and
reports baseline against candidate. Overview and quick start: [../README.md](../README.md).

```
run_eval.py --dist <conf.dist> --candidate <overrides.toml> [--baseline <overrides.toml>]
            --cases <dir> [--personalities <file.toml>] [--rag-dir <dir>] [--banned-terms <file>]
            [--samples 3] [--max-calls 300] [--concurrency 4] [--out <dir>] [--api-key-env NAME]
            [--openers lol,lmao,...] [--dry-run] [--yes]
```

Both variants are `.dist` + overrides, built with `build_conf.py`'s own code (same folding and validation). Without
`--baseline` the baseline is the bare `.dist`. The API key is read from `--api-key-env` (default
`OLLAMA_CHAT_API_KEY`, falling back to `ANTHROPIC_API_KEY` for the anthropic provider) and is never printed or
written to any output.

## Cases

One `*.toml` per situation in `--cases`. `tools/eval/cases-example/` has four.

```toml
id = "reply-lead-the-way"       # unique
path = "reply"                  # reply | random | event
notes = "free text, shown in the report"
message = "whats the command to have you lead the way?"          # reply only
history = [ { player = "...", bot = "..." } ]                    # reply only, optional

[bot]      # name, level, class, race, gender, role, faction, guild, group_status, gold, area, zone, map, personality
[player]   # reply only: name, level, class, race, gender, role, faction, guild, group_status, gold, distance
[event]    # event only: actor_name, kind, event_detail
[random]   # random only: environment_info (text), or env_template + env_vars; optional variation
```

* Every value a placeholder needs comes from the case; a missing one is an error naming the case and the
  placeholder. Only the fields the effective templates actually use are required.
* `role` and `class`/`race` are the strings the module would print ("holy priest", "Night Elf" - race is Title Case, as the server sends it).
* `personality` is a key in `--personalities` (default `personalities-stock.toml`, generated from the module's SQL with
  `sql_to_personalities.py`). When `EnableRPPersonalities = 0` the default personality prompt is used instead, as in
  the module.
* `[random]`: give the topic text as `environment_info`, or name an `EnvComment*` key and its variables:
  `env_template = "OllamaChat.EnvCommentZoneLandmark"`, `env_vars = { area_name = "...", zone_name = "..." }`.
  One `|`-separated variation of that key is picked. `variation` pins the statement/question line; otherwise one is
  picked with the module's statement/question split.
* `[event] kind` names the kind of event; the text for `{event_type}` is read from the effective config
  (`OllamaChat.EventType*`), as the module does, so a candidate that renames one shows up in the prompt.
  Valid kinds: `defeated`, `defeated_player`, `pet_defeated`, `got_item`, `died`, `completed_quest`,
  `learned_spell`, `requested_duel`, `started_dueling`, `won_duel`, `leveled_up`, `achievement`, `used_object`
  (the `EventType*` keys) and the guild ones `guild_epic_gear`, `guild_rare_gear`, `guild_join`, `guild_login`,
  `guild_leave`, `guild_promotion`, `guild_demotion`, `guild_achievement`, `guild_level_up`,
  `guild_dungeon_complete` (the `GuildEventType*` keys). Aliases: `loot` = `got_item`, `level_up` =
  `leveled_up`, `quest` = `completed_quest`, `pvp_kill` = `defeated_player`, `death` = `died`.
* Optional per-case blocks, standing in for state the C++ reads from the game: `snapshot` (text, required when
  `EnableChatBotSnapshotTemplate = 1`), `memories = [{ text, importance }]`, `relationships = [{ name, description }]`,
  `sentiment` (float), `roleplay_voice` (text, required when `Roleplay.Enable = 1`).
* Anything random is drawn from an RNG seeded by case id + sample index (+ purpose), so baseline and candidate get the
  same picks and a rerun gives the same prompts.

## What is mirrored

| Piece | C++ |
|---|---|
| reply prompt, chat history, extra info | `GenerateBotPrompt`, `GetBotHistoryPrompt` in `src/mod-ollama-chat_handler.cpp` |
| sentiment text | `GetSentimentPromptAddition` |
| random-chatter prompt, variations | `BuildRandomChatterPrompt` in `src/mod-ollama-chat_random.cpp` |
| topic text (`EnvComment*`) | `LoadEnvCommentVector` in `config.cpp`, the gatherers in `src/mod-ollama-chat_topics.cpp` |
| event prompt | `OllamaBotEventChatter::BuildPrompt` in `src/mod-ollama-chat_events.cpp` |
| memory and relationship blocks | `Memory_BuildPromptSection` |
| gesture hint | `Expression_BuildGesturePrompt` |
| retrieval | `src/mod-ollama-chat_rag.cpp`, reply path only |
| request and response | `BuildOllamaRequest`/`ParseGenerateBody`, `BuildOpenAIRequest`/`ParseOpenAIBody`, `BuildAnthropicRequest`/`ParseAnthropicBody`, `OllamaBuildRequestHeaders`, the 429/503/529 back-off in `QueryOllama` |
| reply clean-up | `ProcessLlmResponse` (think tags, gesture tags, markdown, quotes, speaker prefix, unicode, length) |

Provider status: **anthropic** was run against the live API; **ollama** and **openai** follow the C++ but are untested here.
Think mode is always off, as in the module's defaults.

`raw.jsonl` has one line per sample: case, variant, sample, prompt hash (not the prompt), system-prompt hash, raw and
processed reply, token usage, latency, status, error and - for the reply path - the retrieved RAG ids and scores
(use them to tune `RAGSimilarityThreshold`).

## Known differences from the C++

* **Live game state** (area and zone names, roles, gold, distance, nearby people, snapshot, memories,
  relationships, sentiment, the topic the picker would choose) comes from the case, not a `Player`. `snapshot` text
  is used verbatim; the snapshot builder is not reimplemented.
* **Dice** (statement/question choice, variation and topic pick, gesture hint) use a seeded Python RNG, not `urand`.
  The distributions match; the individual picks cannot.
* **Float printing** of `player_distance` and `sentiment_value` imitates `{fmt}` for a `float` (shortest float32
  round trip, `4.0` prints as `4`, fixed notation for decimal exponents -4 to 15 as {fmt} does).
* **Roleplay mode** is not reimplemented: with `Roleplay.Enable = 1` the case must supply `roleplay_voice`, and the
  roleplay meta-term reply filter (`Roleplay_FilterMetaTerms`) is not applied.
* **`MaxConversationHistory = 0`**: the C++ keeps no entries but still emits the history header and footer;
  this keeps at least one. **`Roleplay.PromptVariations` / `QuestionVariations`** are split without trimming and
  without the built-in fallbacks the C++ applies when the list is empty (only matters with roleplay on).
* **Guild chatter** templates (`GuildEnvComment*`, guild events) are not rendered.
* **RAG**: file order is sorted by name and the vocabulary sorted, where the C++ takes directory order and hash order;
  results do not depend on either, apart from the order of exactly tied scores. The C++ uses `float` accumulators,
  this uses doubles (differences around 1e-7, only visible against a threshold to that precision).
  `RAGDataPath` is resolved against the module root, falling back to the module's `data/rag`.
* **Reply clean-up** after the model is ported on bytes. Not applied: the governor's repetition and rate checks,
  `Roleplay_FilterMetaTerms`, and delivery.
* **Request parameters** are sent as plain JSON numbers; the C++ serialises the config's `float` values as doubles
  (`0.8` goes out as `0.800000011920929`). The OpenAI parameter self-heals (`max_completion_tokens`, rejected
  sampling parameters) and the capability probe are not implemented. Streaming is not used.

## Tests

`test_build_conf.py` (folding, quote rejection, unknown keys, placeholder validation, in-place replacement,
duplicate keys, CLI behaviour, a drift check of the placeholder table against the shipped `.dist`) and `test_eval.py`
(RAG scores worked out by hand from the C++, the `ProcessLlmResponse` port, request builders and parsers, retry
logic, metrics, rendering). `python3 -m unittest discover -s . -p 'test_*.py'` from this directory; no network.
