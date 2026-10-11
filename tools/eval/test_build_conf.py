"""Tests for tools/build_conf.py. Standard library only, no network."""

import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import build_conf as bc  # noqa: E402

DIST = """\
# header comment
[worldserver]

# OllamaChat.Provider
OllamaChat.Provider = ollama

OllamaChat.WorkerThreads = 4
OllamaChat.Enable = 1
OllamaChat.SystemPrompt =
OllamaChat.ChatPromptTemplate = "Hi {bot_name}, level {bot_level}. {player_message}"
OllamaChat.Plain = no placeholders here
OllamaChat.Twice = a
OllamaChat.Twice = b
"""


def build(overrides, dist=DIST):
    return bc.build(dist, overrides)


class FoldTests(unittest.TestCase):
    def test_newline_runs_become_one_space(self):
        self.assertEqual(bc.fold("one\n   two\n\n\nthree  four\n"), "one two three  four")

    def test_spaces_without_newline_are_kept(self):
        self.assertEqual(bc.fold("a  b"), "a  b")

    def test_crlf(self):
        self.assertEqual(bc.fold("a\r\nb"), "a b")

    def test_backslash_n_is_untouched(self):
        self.assertEqual(bc.fold(r"line\nnext"), r"line\nnext")

    def test_edge_space_without_newline_is_kept(self):
        self.assertEqual(bc.fold(" lead and trail "), " lead and trail ")

    def test_edge_runs_with_newline_are_dropped(self):
        self.assertEqual(bc.fold("\n  text\n"), "text")
        self.assertEqual(bc.fold(" \n text \n "), "text")

    def test_values(self):
        self.assertEqual(bc.render_value(True), "1")
        self.assertEqual(bc.render_value(False), "0")
        self.assertEqual(bc.render_value(2), "2")
        self.assertEqual(bc.render_value(0.5), "0.5")


class TomlFoldTests(unittest.TestCase):
    """Folding as the user meets it: TOML text -> tomllib -> build -> value the module loads."""

    DIST2 = ('OllamaChat.Line = "{a}\\n{b}\\n"\n'
             'OllamaChat.Mem = " Things you remember:\\n{memories}"\n'
             'OllamaChat.Memory.PromptTemplate = " Things you remember:\\n{memories}"\n')

    def loaded(self, toml_text, key="OllamaChat.Line", dist=None):
        data = tomllib.loads(toml_text)
        r = bc.build(dist or self.DIST2, data)
        self.assertFalse(r.errors, r.errors)
        return bc.parse_conf_text(r.text)[key], r

    def test_basic_multiline_backslash_n_becomes_a_space(self):
        # In a TOML """ string, \n is a real newline, so it folds to a space (documented behaviour).
        v, _ = self.loaded('"OllamaChat.Line" = """{a}\\n{b}"""\n')
        self.assertEqual(v, "{a} {b}")

    def test_literal_multiline_keeps_the_two_characters(self):
        v, r = self.loaded("\"OllamaChat.Line\" = '''{a}\\n{b}\\n'''\n")
        self.assertEqual(v, "{a}\\n{b}\\n")
        self.assertFalse(r.warnings)

    def test_escaped_backslash_n_keeps_the_two_characters(self):
        v, _ = self.loaded('"OllamaChat.Line" = "{a}\\\\n{b}\\\\n"\n')
        self.assertEqual(v, "{a}\\n{b}\\n")

    def test_leading_space_is_kept_and_quoted(self):
        v, r = self.loaded('"OllamaChat.Mem" = " Block:{memories}"\n', "OllamaChat.Mem")
        self.assertEqual(v, " Block:{memories}")
        self.assertIn('OllamaChat.Mem = " Block:{memories}"', r.text)

    def test_delimiter_newlines_vanish_but_edge_space_stays(self):
        v, _ = self.loaded('"OllamaChat.Mem" = """\n Block\n"""\n', "OllamaChat.Mem")
        self.assertEqual(v, " Block")

    def test_warns_when_stock_separator_is_lost(self):
        _, r = self.loaded('"OllamaChat.Line" = """{a}\\n{b}"""\n')
        self.assertTrue(any("literal \\n separator" in w and "did TOML turn it into a newline" in w
                            for w in r.warnings), r.warnings)

    def test_no_warning_when_stock_has_no_separator(self):
        _, r = self.loaded('"OllamaChat.Plain" = "a\\nb"\n', "OllamaChat.Plain",
                           dist="OllamaChat.Plain = plain\n")
        self.assertFalse(r.warnings)


class FormatSpecTests(unittest.TestCase):
    def test_format_spec_rejected(self):
        r = build({"OllamaChat.ChatPromptTemplate": "level {bot_level:d}"})
        self.assertTrue(any("format spec" in e and "bot_level" in e for e in r.errors), r.errors)

    def test_conversion_rejected(self):
        r = build({"OllamaChat.ChatPromptTemplate": "name {bot_name!r}"})
        self.assertTrue(any("conversion" in e for e in r.errors), r.errors)

    def test_plain_placeholder_still_fine(self):
        r = build({"OllamaChat.ChatPromptTemplate": "name {bot_name} {{literal}}"})
        self.assertFalse(r.errors, r.errors)


class ReplaceTests(unittest.TestCase):
    def test_replaced_in_place_keeping_comments_and_order(self):
        r = build({"OllamaChat.Provider": "anthropic", "OllamaChat.WorkerThreads": 2})
        self.assertFalse(r.errors)
        out = r.text.split("\n")
        self.assertEqual(out[0], "# header comment")
        self.assertEqual(out[3], "# OllamaChat.Provider")           # the commented example is untouched
        self.assertEqual(out[4], "OllamaChat.Provider = anthropic")  # unquoted in the .dist -> unquoted
        self.assertEqual(out[6], "OllamaChat.WorkerThreads = 2")
        self.assertEqual(out[7], "OllamaChat.Enable = 1")
        self.assertEqual(len(out), len(DIST.split("\n")))

    def test_quoted_and_empty_values_get_quotes(self):
        r = build({"OllamaChat.ChatPromptTemplate": "x {bot_name}", "OllamaChat.SystemPrompt": "Be brief.\nBe kind."})
        self.assertFalse(r.errors)
        self.assertIn('OllamaChat.ChatPromptTemplate = "x {bot_name}"', r.text)
        self.assertIn('OllamaChat.SystemPrompt = "Be brief. Be kind."', r.text)

    def test_text_roundtrips_through_the_config_parser(self):
        r = build({"OllamaChat.SystemPrompt": "A\n\nB"})
        self.assertEqual(bc.parse_conf_text(r.text)["OllamaChat.SystemPrompt"], "A B")

    def test_trailing_newline_preserved(self):
        r = build({"OllamaChat.WorkerThreads": 1})
        self.assertTrue(r.text.endswith("\n"))
        self.assertNotIn("\r", r.text)

    def test_double_quote_rejected(self):
        r = build({"OllamaChat.SystemPrompt": 'say "hi"'})
        self.assertTrue(any("double quote" in e for e in r.errors))

    def test_unknown_key_rejected(self):
        r = build({"OllamaChat.Nope": 1})
        self.assertTrue(any("renamed or removed upstream" in e for e in r.errors))

    def test_commented_only_key_gets_a_hint(self):
        r = bc.build("# OllamaChat.Foo = 1\nOllamaChat.Bar = 2\n", {"OllamaChat.Foo": 3})
        self.assertTrue(any("commented-out" in e for e in r.errors))

    def test_duplicate_key_in_dist_is_an_error(self):
        r = build({"OllamaChat.Twice": "z"})
        self.assertTrue(any("2 times" in e for e in r.errors))

    def test_duplicate_key_not_overridden_only_warns(self):
        r = build({"OllamaChat.WorkerThreads": 1})
        self.assertFalse(r.errors)
        self.assertTrue(any("OllamaChat.Twice" in w for w in r.warnings))


class PlaceholderTests(unittest.TestCase):
    def test_known_placeholders_ok(self):
        r = build({"OllamaChat.ChatPromptTemplate": "{bot_name} {player_name} {chat_history}"})
        self.assertFalse(r.errors, r.errors)      # chat_history is passed by the C++ though the stock text omits it

    def test_unknown_placeholder(self):
        r = build({"OllamaChat.ChatPromptTemplate": "{bot_name} {bot_nmae}"})
        self.assertTrue(any("bot_nmae" in e for e in r.errors))

    def test_stray_braces(self):
        for bad in ("{bot_name", "bot_name}", "{}", "{bot_name:{x}}"):
            r = build({"OllamaChat.ChatPromptTemplate": bad})
            self.assertTrue(r.errors, bad)

    def test_escaped_braces_ok(self):
        r = build({"OllamaChat.ChatPromptTemplate": "{{literal}} {bot_name}"})
        self.assertFalse(r.errors)

    def test_system_prompt_is_not_formatted(self):
        r = build({"OllamaChat.SystemPrompt": "You are {bot_name}"})
        self.assertTrue(any("not run through the template formatter" in e for e in r.errors))
        r = build({"OllamaChat.SystemPrompt": 'JSON like { a: 1 } is fine'})
        self.assertFalse(r.errors)

    def test_key_without_placeholders_cannot_gain_them(self):
        r = build({"OllamaChat.Plain": "now {bot_name}"})
        self.assertTrue(r.errors)

    def test_parse_template_tokens(self):
        tokens, errors = bc.parse_template("a {x} b {{c}} {y:>4}")
        self.assertFalse(errors)
        self.assertEqual([t.text for t in tokens if t.kind == "field"], ["x", "y"])
        self.assertEqual("".join(t.text for t in tokens if t.kind == "lit"), "a  b {c} ")


class CliTests(unittest.TestCase):
    def run_cli(self, tmp, overrides_toml, *extra):
        dist = Path(tmp) / "d.dist"
        dist.write_text(DIST, encoding="utf-8")
        ov = Path(tmp) / "o.toml"
        ov.write_text(overrides_toml, encoding="utf-8")
        out = Path(tmp) / "live.conf"
        proc = subprocess.run([sys.executable, str(HERE.parent / "build_conf.py"), "--dist", str(dist),
                               "--overrides", str(ov), "--out", str(out), *extra],
                              capture_output=True, text=True)
        return proc, out

    def test_write_backup_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, out = self.run_cli(tmp, '"OllamaChat.WorkerThreads" = 2\n')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("OllamaChat.WorkerThreads = 2", out.read_text())
            self.assertTrue((Path(tmp) / "live.conf.keys").exists())
            proc, out = self.run_cli(tmp, '"OllamaChat.WorkerThreads" = 3\n')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(len(list(Path(tmp).glob("live.conf.bak-*"))), 1)

    def test_error_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, out = self.run_cli(tmp, '"OllamaChat.Nope" = 2\n')
            self.assertEqual(proc.returncode, 1)
            self.assertFalse(out.exists())

    def test_check_writes_nothing_and_lists_new_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "live.conf.keys").write_text("OllamaChat.Enable\n")
            proc, out = self.run_cli(tmp, '"OllamaChat.WorkerThreads" = 2\n', "--check")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(out.exists())
            self.assertIn("+ OllamaChat.WorkerThreads", proc.stdout)

    def test_toml_dotted_key_unquoted_is_explained(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, _ = self.run_cli(tmp, "OllamaChat.WorkerThreads = 2\n")
            self.assertEqual(proc.returncode, 2)
            self.assertIn("quoted", proc.stderr)

    def test_diff_prints_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, out = self.run_cli(tmp, '"OllamaChat.WorkerThreads" = 2\n', "--diff")
            self.assertEqual(proc.returncode, 0)
            self.assertIn("+OllamaChat.WorkerThreads = 2", proc.stdout)
            self.assertFalse(out.exists())


class ShippedDistTests(unittest.TestCase):
    """Drift guard: every placeholder in the real .dist must be covered by the call-site table."""

    def test_table_covers_the_dist(self):
        dist = HERE.parent.parent / "conf" / "mod_ollama_chat.conf.dist"
        if not dist.exists():
            self.skipTest("conf/mod_ollama_chat.conf.dist not found")
        text = dist.read_text(encoding="utf-8")
        _, keys = bc.scan_dist(text)
        problems = []
        for key, occ in keys.items():
            placeholders, errors = bc.placeholders_of(bc.conf_value(occ[0].raw_value))
            if errors:
                problems.append(f"{key}: {errors}")
            if placeholders:
                allowed = bc.ALLOWED_PLACEHOLDERS.get(key)
                if allowed is None or placeholders - allowed:
                    problems.append(f"{key}: {sorted(placeholders - (allowed or set()))} not in ALLOWED_PLACEHOLDERS")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
