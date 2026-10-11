# custom/: our server's overlay

Everything here is specific to our realm. The generic tools live in `tools/` (branch `feat/prompt-tooling`, offered
upstream); this directory exists only on `server/overlay`.

| path | what |
|---|---|
| `overrides.toml` | the keys we change from `conf/mod_ollama_chat.conf.dist`, one reason comment each. The live config is built from it |
| `eval/cases/` | eval cases: hand-made situations (`event-*`, `random-*`, `trap-reply-*`) and 16 real replies from the 2026-10-10/11 playtest (`real-*`, with their history) |
| `eval/personalities.toml` | our editable copy of the personality prompts (identical to the live table on 2026-10-11; Q2 changes it) |
| `eval/banned-terms.txt` | content that does not exist on this realm; counted as `banned_hits` |
| `eval/reports/` | eval output (gitignored) |

All paths below are relative to the module root, `modules/mod-ollama-chat/`. `<LIVE_CONF>` is the live config,
`env/dist/etc/modules/mod_ollama_chat.conf` in the main repo; it is given as an argument and never stored here.

## Build and check the live config

```sh
# validate the overrides, write nothing
python3 tools/build_conf.py --dist conf/mod_ollama_chat.conf.dist --overrides custom/overrides.toml \
    --out <LIVE_CONF> --check

# what would change in the live file
python3 tools/build_conf.py --dist conf/mod_ollama_chat.conf.dist --overrides custom/overrides.toml \
    --out <LIVE_CONF> --diff

# write it (backs up the old file next to it); then reload or restart the worldserver
python3 tools/build_conf.py --dist conf/mod_ollama_chat.conf.dist --overrides custom/overrides.toml \
    --out <LIVE_CONF>
```

`--diff` against the live file is empty apart from one hand-written `#` comment line after the `ApiKey` line;
the tool never writes comments. Writing the file drops that comment.

## Dry run (no network, no cost)

Writes every rendered prompt to `tools/eval/reports/<timestamp>/prompts/`.

```sh
python3 tools/eval/run_eval.py --dist conf/mod_ollama_chat.conf.dist --candidate custom/overrides.toml \
    --cases custom/eval/cases --personalities custom/eval/personalities.toml \
    --banned-terms custom/eval/banned-terms.txt --out custom/eval/reports --samples 1 --dry-run
```

## Eval: our overrides against the stock `.dist`

Without `--baseline` the baseline is the bare `.dist` (personalities off, stock prompts). The key comes from
`OLLAMA_CHAT_API_KEY` or, for the anthropic provider, `ANTHROPIC_API_KEY`. The tool prints the call count and asks
before sending; `--max-calls` is a hard stop.

```sh
python3 tools/eval/run_eval.py --dist conf/mod_ollama_chat.conf.dist --candidate custom/overrides.toml \
    --cases custom/eval/cases --personalities custom/eval/personalities.toml \
    --banned-terms custom/eval/banned-terms.txt --out custom/eval/reports --samples 3 --max-calls 300
```

To compare two override sets, add `--baseline <older.toml>`. The case format and the event kinds are in
`tools/eval/README.md`.
