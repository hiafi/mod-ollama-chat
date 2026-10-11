"""Tests for the personality tooling: loader, per-variant files, sweep mode, report grouping, SQL generator."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import api                      # noqa: E402
import personalities as pers    # noqa: E402
import personalities_to_sql as p2s   # noqa: E402
import render                   # noqa: E402
import run_eval                 # noqa: E402
import sql_to_personalities as s2p   # noqa: E402

DIST = HERE.parents[1] / "conf" / "mod_ollama_chat.conf.dist"


class LoaderTests(unittest.TestCase):
    def test_both_forms(self):
        out = pers.parse_personalities({"A": "plain", "B": {"prompt": "tbl", "manual_only": True},
                                        "C": {"prompt": "tbl2"}})
        self.assertEqual(out, {"A": ("plain", False), "B": ("tbl", True), "C": ("tbl2", False)})

    def test_errors_name_the_key(self):
        for bad, text in [({"X": {"manual_only": True}}, "without 'prompt'"),
                          ({"X": {"prompt": "p", "extra": 1}}, "unknown field"),
                          ({"X": {"prompt": 5}}, "must be a string"),
                          ({"X": 5}, "must be a string or a table"),
                          ({"X": {"prompt": "p", "manual_only": "yes"}}, "true or false")]:
            with self.assertRaises(ValueError) as cm:
                pers.parse_personalities(bad)
            self.assertIn("'X'", str(cm.exception))
            self.assertIn(text, str(cm.exception))

    def test_load_file_ignores_comment_marker(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "p.toml"
            f.write_text('A = "x"  # manual_only\nB = { prompt = "y", manual_only = true }\n', encoding="utf-8")
            self.assertEqual(pers.load_personalities(f), {"A": ("x", False), "B": ("y", True)})

    def test_stock_file_loads_all_non_manual(self):
        stock = pers.load_personalities(HERE / "personalities-stock.toml")
        self.assertGreater(len(stock), 3)
        self.assertFalse(any(m for _p, m in stock.values()))


class SqlToTomlTests(unittest.TestCase):
    def test_manual_row_written_as_inline_table_and_roundtrips(self):
        rows = [("A", "it's", False), ("B", "back\\slash \"q\"", True)]
        text = s2p.to_toml(rows)
        self.assertIn('B = { prompt = ', text)
        import tomllib
        got = pers.parse_personalities(tomllib.loads(text))
        self.assertEqual(got, {"A": ("it's", False), "B": ('back\\slash "q"', True)})


class SweepHelperTests(unittest.TestCase):
    P = {"A": ("pa", False), "B": ("pb", True), "C": ("pc", False)}

    def test_all_skips_manual(self):
        self.assertEqual(run_eval.expand_sweep("all", self.P), ["A", "C"])

    def test_list_includes_manual_and_rejects_unknown(self):
        self.assertEqual(run_eval.expand_sweep("B, A", self.P), ["B", "A"])
        with self.assertRaises(ValueError) as cm:
            run_eval.expand_sweep("A,ZZZ", self.P)
        self.assertIn("ZZZ", str(cm.exception))

    def test_baseline_fallback(self):
        self.assertEqual(run_eval.baseline_key_for("A", {"A": "x"}), ("A", ""))
        name, note = run_eval.baseline_key_for("NEW", {"A": "x"})
        self.assertEqual(name, "default")
        self.assertIn("DefaultPersonalityPrompt", note)

    def test_seed_includes_key_and_matches_across_variants(self):
        case = {"id": "c", "path": "random", "bot": {"personality": "X"}}
        a = run_eval.sweep_case(case, "K1")
        b = run_eval.sweep_case(case, "K2")
        self.assertEqual(render.CaseView(a, "c").seed, "c|K1")
        self.assertNotEqual(render.CaseView(a, "c").seed, render.CaseView(b, "c").seed)
        self.assertEqual(render.CaseView(case, "c").seed, "c")
        self.assertEqual(a["bot"]["personality"], "K1")
        self.assertEqual(case["bot"]["personality"], "X")     # original untouched


class SweepRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cases = self.tmp / "cases"
        self.cases.mkdir()
        for name in ("reply-greeting.toml", "event-levelup.toml"):
            shutil.copy(HERE / "cases-example" / name, self.cases / name)
        (self.tmp / "cand.toml").write_text('"OllamaChat.EnableRPPersonalities" = true\n', encoding="utf-8")
        (self.tmp / "cand-off.toml").write_text("", encoding="utf-8")
        (self.tmp / "cand-p.toml").write_text(
            'CASUAL = "casual"\nALPHA = "alpha cand"\nBETA = "beta cand"\nNEWONE = "brand new"\n'
            'HIDDEN = { prompt = "hidden cand", manual_only = true }\n', encoding="utf-8")
        (self.tmp / "base-p.toml").write_text('ALPHA = "alpha base"\nBETA = "beta cand"\n', encoding="utf-8")
        self.prompts = []
        orig = run_eval.api.query
        self.addCleanup(setattr, run_eval.api, "query", orig)

        def fake(settings, prompt):
            self.prompts.append((settings.system_prompt, prompt))
            return api.ApiResult(ok=True, text="hello there friend", status=200, input_tokens=1, output_tokens=1)
        run_eval.api.query = fake

    def run_main(self, *extra, cand="cand.toml"):
        out = self.tmp / "out"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = run_eval.main(["--dist", str(DIST), "--candidate", str(self.tmp / cand), "--cases", str(self.cases),
                                "--personalities", str(self.tmp / "cand-p.toml"), "--samples", "2",
                                "--concurrency", "1", "--yes", "--out", str(out), *extra])
        return rc, out, buf.getvalue()

    def run_dir(self, out):
        return next(out.iterdir())

    def test_call_count_and_raw_personality_field(self):
        rc, out, text = self.run_main("--baseline-personalities", str(self.tmp / "base-p.toml"),
                                      "--sweep-personalities", "all")
        self.assertEqual(rc, 0, text)
        # 2 cases x 4 non-manual keys x 2 samples x 2 variants
        self.assertEqual(len(self.prompts), 32)
        rows = [json.loads(x) for x in (self.run_dir(out) / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 32)
        self.assertEqual({r["personality"] for r in rows}, {"CASUAL", "ALPHA", "BETA", "NEWONE"})

    def test_max_calls_applies_to_total(self):
        rc, _out, text = self.run_main("--sweep-personalities", "all", "--max-calls", "31")
        self.assertEqual(rc, 2)
        self.assertEqual(self.prompts, [])
        self.assertIn("32 calls exceeds", text)

    def test_list_includes_manual_and_unknown_is_error(self):
        rc, out, _ = self.run_main("--sweep-personalities", "HIDDEN")
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.prompts), 2 * 1 * 2 * 2)
        self.prompts.clear()
        rc, _out, text = self.run_main("--sweep-personalities", "NOPE")
        self.assertEqual(rc, 1)
        self.assertIn("NOPE", text)
        self.assertEqual(self.prompts, [])

    def test_sweep_forces_rp_personalities_on(self):
        rc, _out, text = self.run_main("--sweep-personalities", "ALPHA", cand="cand-off.toml")
        self.assertEqual(rc, 0, text)
        self.assertIn("forced on", text)
        self.assertTrue(any("alpha cand" in s + p for s, p in self.prompts))

    def test_report_grouped_by_personality_with_fallback_and_per_variant_files(self):
        rc, out, text = self.run_main("--baseline-personalities", str(self.tmp / "base-p.toml"),
                                      "--sweep-personalities", "ALPHA,NEWONE")
        self.assertEqual(rc, 0, text)
        report = (self.run_dir(out) / "report.md").read_text(encoding="utf-8")
        self.assertLess(report.index("## Metrics"), report.index("## Personalities"))
        self.assertIn("### ALPHA", report)
        self.assertIn("### NEWONE", report)
        self.assertLess(report.index("### ALPHA"), report.index("### NEWONE"))
        self.assertIn("_Prompt, baseline:_ alpha base", report)
        self.assertIn("_Prompt, candidate:_ alpha cand", report)
        self.assertIn("uses DefaultPersonalityPrompt", report)
        self.assertEqual(report.count("#### example-reply-greeting"), 2)       # once per personality
        self.assertIn("slang_open_pct", report.split("### ALPHA")[1])
        self.assertIn("base-p.toml", report)
        self.assertIn("cand-p.toml", report)
        # the baseline got alpha base for ALPHA, the candidate alpha cand
        joined = [s + p for s, p in self.prompts]
        self.assertTrue(any("alpha base" in x for x in joined))
        self.assertTrue(any("alpha cand" in x for x in joined))
        self.assertTrue(any("brand new" in x for x in joined))

    def test_baseline_and_candidate_get_the_same_dice(self):
        rc, out, _ = self.run_main("--sweep-personalities", "ALPHA", "--dry-run")
        self.assertEqual(rc, 0)
        pdir = self.run_dir(out) / "prompts"
        names = sorted(p.name for p in pdir.iterdir())
        self.assertIn("example-event-levelup-ALPHA-baseline-0.txt", names)
        for case in ("example-event-levelup", "example-reply-greeting"):
            for sample in (0, 1):
                a = (pdir / f"{case}-ALPHA-baseline-{sample}.txt").read_text(encoding="utf-8")
                b = (pdir / f"{case}-ALPHA-candidate-{sample}.txt").read_text(encoding="utf-8")
                self.assertEqual(a, b)          # same file for both, so identical prompts incl. the dice
        self.assertEqual(self.prompts, [])      # dry run: no calls

    def test_non_sweep_unchanged_shape(self):
        rc, out, text = self.run_main()
        self.assertEqual(rc, 0, text)
        report = (self.run_dir(out) / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Cases", report)
        self.assertNotIn("## Personalities", report)
        rows = [json.loads(x) for x in (self.run_dir(out) / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIsNone(rows[0]["personality"])


class ToSqlTests(unittest.TestCase):
    def test_escaping_ordering_and_roundtrip(self):
        src = {"ZED": ("it's a \\ test, {braces}", True), "ALPHA": ("plain", False), "MID_1": ("o'k ''", False)}
        sql = p2s.to_sql(src, "p.toml", "note line")
        self.assertIn("REPLACE INTO `mod_ollama_chat_personality_templates` (`key`, `prompt`, `manual_only`) VALUES", sql)
        self.assertNotIn("DELETE FROM", sql)
        self.assertIn("-- note line", sql)
        self.assertIn("it''s a \\\\ test", sql)
        self.assertLess(sql.index("'ALPHA'"), sql.index("'MID_1'"))
        self.assertLess(sql.index("'MID_1'"), sql.index("'ZED'"))
        self.assertEqual(sql.count("REPLACE INTO"), 1)
        rows = s2p.parse_rows(sql)
        self.assertEqual({k: (p, m) for k, p, m in rows}, src)

    def test_validation(self):
        for bad, text in [({"lower": ("p", False)}, "key must match"),
                          ({"A" * 65: ("p", False)}, "key must match"),
                          ({"K": ("", False)}, "empty"),
                          ({"K": ("   ", False)}, "empty"),
                          ({"K": ('say "hi"', False)}, "'\"'"),
                          ({"K": ("a\x00b", False)}, "control character"),
                          ({"K": ("a\nb", False)}, "control character")]:
            with self.assertRaises(ValueError) as cm:
                p2s.to_sql(bad, "p.toml")
            self.assertIn(text, str(cm.exception))
            self.assertIn(next(iter(bad))[:10], str(cm.exception))

    def test_cli_writes_file_and_roundtrips_stock(self):
        stock = pers.load_personalities(HERE / "personalities-stock.toml")
        stock["EXTRA"] = ("manual one", True)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "p.toml"
            src.write_text(s2p.to_toml([(k, p, m) for k, (p, m) in stock.items()]), encoding="utf-8")
            out = Path(d) / "o.sql"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(p2s.main([str(src), "--out", str(out)]), 0)
            back = {k: (p, m) for k, p, m in s2p.parse_rows(out.read_text(encoding="utf-8"))}
            self.assertEqual(back, stock)
            src.write_text('bad = "x"\n', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(p2s.main([str(src), "--out", str(out)]), 1)


if __name__ == "__main__":
    unittest.main()
