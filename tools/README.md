# Prompt tooling

Two standard-library Python scripts (Python 3.11+, no pip installs) for people who tune this module's prompts.

* `build_conf.py` builds your live config from the shipped `conf/mod_ollama_chat.conf.dist` plus a small TOML
  file holding only the settings you changed.
* `eval/run_eval.py` renders prompts exactly the way the C++ does, sends them to the model, and writes a
  side-by-side report of two override sets, so a prompt change can be judged before it goes live.

## Why overrides instead of editing the config

AzerothCore reads one file per module, one line per key; every `"` in a value is deleted; and when a key appears
twice the **first** one wins, so appending an override at the end of the file silently does nothing. A long prompt
on one line is also hard to read and impossible to review in a diff. `build_conf.py` keeps your changes in a short
TOML file (real multi-line strings, a comment per key), rewrites only those lines of the stock `.dist` in place, and
refuses to write anything that would turn a prompt into `[Format Error]`.

## build_conf.py

```toml
# my-overrides.toml  - keys are quoted because they contain dots
"OllamaChat.Url" = "http://localhost:11434/api/generate"   # the default; shown for the shape
"OllamaChat.Model" = "llama3.2:3b"
"OllamaChat.WorkerThreads" = 2
"OllamaChat.SystemPrompt" = """
You are one player in a busy game chat. Keep replies short and in plain English.
Never invent commands.
"""
```

```sh
python3 tools/build_conf.py --dist conf/mod_ollama_chat.conf.dist \
    --overrides my-overrides.toml --out /path/to/etc/modules/mod_ollama_chat.conf
```

* Strings are folded to one line: a run of whitespace that contains a newline becomes one space. At the two
  ends only such a run is dropped, so a deliberate leading or trailing space survives (the value is then written
  in quotes; the config parser trims the line before it deletes the quotes). Some stock templates use the two
  characters backslash and `n` as a separator and the module passes them through. **In TOML, `\n` inside `"..."`
  or `"""..."""` is a real newline** and folds to a space. To keep the two characters, write the value as a
  literal string (`'''...'''`) or as `\\n`. When the stock value has a literal `\n` and your value has none,
  `build_conf.py` prints a warning.
* `--check` validates and writes nothing; it also lists `.dist` keys that are new since the last build
  (it keeps the previous list next to the output as `<out>.keys`). Run it after updating the module.
* `--diff` prints a unified diff between the current `--out` and what would be written, and writes nothing.
* The old `--out` is copied to `<out>.bak-<timestamp>` before it is replaced, and the new file is written atomically.
* Quoting follows the `.dist` line being replaced (the parser deletes the quotes either way).

It stops with a non-zero exit and writes nothing if:

* an override key is not a real line in the `.dist` (renamed or removed upstream; a key that exists only as a
  commented-out example is reported as such), or appears more than once there;
* a string contains `"`, or a newline survives folding;
* a template uses a `{placeholder}` the C++ does not pass for that key, or has a stray or unbalanced brace.
  The allowed names come from the `SafeFormat(...)` call sites and live in `ALLOWED_PLACEHOLDERS`; a test checks
  that the shipped `.dist` stays inside that table. `OllamaChat.SystemPrompt` is never formatted, so a `{name}` in it
  would reach the model literally and is rejected.

## eval/run_eval.py

```sh
# Ollama (default): compare a candidate against the stock prompts, 3 samples per case
python3 tools/eval/run_eval.py --dist conf/mod_ollama_chat.conf.dist \
    --candidate my-overrides.toml --cases tools/eval/cases-example

# Anthropic as an example: set Provider/Url/Model in the overrides, key from the environment
export OLLAMA_CHAT_API_KEY=...        # or ANTHROPIC_API_KEY for the anthropic provider
python3 tools/eval/run_eval.py --dist conf/mod_ollama_chat.conf.dist \
    --baseline last-known-good.toml --candidate my-overrides.toml --cases my-cases --samples 3
```

It prints how many calls it will make (and an estimated cost where it knows the model's price) and asks
`Proceed? [y/N]`. `--max-calls` is a hard stop, `--yes` skips the question, and `--dry-run` writes every rendered
prompt to `<out>/<timestamp>/prompts/` and makes no network call - the quickest way to check what the model will see.

Each run writes `report.md` (metrics table, then every case with baseline and candidate samples side by side,
failures listed per case) and `raw.jsonl` to `tools/eval/reports/<timestamp>/` (gitignored). The metrics only catch
regressions - the opener share, opener variety, length, banned terms, near-duplicates and failures. You are the judge.

Full details, the case format and the list of known differences from the C++ are in [eval/README.md](eval/README.md).

## Tests

```sh
cd tools/eval && python3 -m unittest discover -s . -p 'test_*.py'
```

No network is used.
