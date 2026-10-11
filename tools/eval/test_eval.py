"""Tests for the eval harness: RAG scoring, the ProcessLlmResponse port, request building, metrics, rendering.

Expected values for the RAG and ProcessLlmResponse cases were worked out by hand from the C++ source
(src/mod-ollama-chat_rag.cpp, src/mod-ollama-chat_response.cpp, src/mod-ollama-chat_expression.cpp).
"""

import math
import sys
import unittest
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import api            # noqa: E402
import build_conf     # noqa: E402
import metrics        # noqa: E402
import postprocess as pp  # noqa: E402
import rag            # noqa: E402
import render         # noqa: E402


class RagTests(unittest.TestCase):
    def test_preprocess_deletes_punctuation_and_lowercases(self):
        self.assertEqual(rag.preprocess_text("It's A Test, ok?"), "its a test ok")

    def test_tokenize_splits_on_whitespace(self):
        self.assertEqual(rag.tokenize_text(" a\tb\n c  "), ["a", "b", "c"])

    def test_cosine_one_matching_term(self):
        # vocabulary {cat, sat, dog, ran}; query "Cat!" -> cat=1; entry "Cat cat sat" -> cat=2, sat=1
        # cos = (1*2) / (1 * sqrt(5))
        s = rag.RagSystem([rag.RagEntry("e1", "Cat", "cat sat"), rag.RagEntry("e2", "Dog", "dog ran")])
        self.assertAlmostEqual(s.similarity("Cat!", s.entries[0]), 2 / math.sqrt(5), places=6)
        self.assertEqual(s.similarity("Cat!", s.entries[1]), 0.0)
        hits = s.retrieve("Cat!", 3, 0.3)
        self.assertEqual([h.entry.id for h in hits], ["e1"])

    def test_apostrophe_is_deleted_not_split(self):
        # vocabulary {t, its, fine}; query "it's" -> its=1; entry "T its fine" -> 1,1,1; cos = 1/sqrt(3)
        s = rag.RagSystem([rag.RagEntry("e", "T", "its fine")])
        self.assertAlmostEqual(s.similarity("it's", s.entries[0]), 1 / math.sqrt(3), places=6)

    def test_keywords_count_toward_the_entry_vector(self):
        # vocabulary {a, x, y, z}; entry "A x y z z" -> (1,1,1,2); query "z" -> (0,0,0,1); cos = 2/sqrt(7)
        s = rag.RagSystem([rag.RagEntry("e", "A", "x y", ["z z"])])
        self.assertAlmostEqual(s.similarity("z", s.entries[0]), 2 / math.sqrt(7), places=6)

    def test_threshold_and_limit_and_order(self):
        s = rag.RagSystem([rag.RagEntry("a", "A", "cat"), rag.RagEntry("b", "B", "cat cat dog"),
                           rag.RagEntry("c", "C", "dog")])
        hits = s.retrieve("cat", 1, 0.3)
        self.assertEqual(len(hits), 1)
        # cos("cat", "A cat") = 1/sqrt(2) = 0.707; cos("cat", "B cat cat dog") = 2/sqrt(6) = 0.816
        self.assertEqual(hits[0].entry.id, "b")
        self.assertEqual(s.retrieve("cat", 5, 0.99), [])

    def test_format_info(self):
        s = rag.RagSystem([rag.RagEntry("a", "A", "one"), rag.RagEntry("b", "B", "two")])
        hits = [rag.RagResult(s.entries[0], 1.0), rag.RagResult(s.entries[1], 0.5)]
        self.assertEqual(s.format_info(hits), "- A: one\n- B: two")

    def test_shipped_data_loads(self):
        data = HERE.parent.parent / "data" / "rag"
        if not data.is_dir():
            self.skipTest("no data/rag")
        s = rag.RagSystem.from_directory(data)
        self.assertGreater(len(s.entries), 10)


class ProcessResponseTests(unittest.TestCase):
    def p(self, raw, bot="Bob", **kw):
        return pp.process_llm_response(raw, bot, **kw)[0]

    def test_quote_unwrap(self):
        self.assertEqual(self.p('"Hello there"'), "Hello there")
        self.assertEqual(self.p("'Hello there'"), "Hello there")

    def test_inner_quote_blocks_unwrap(self):
        self.assertEqual(self.p('"Hi" he said "yo"'), '"Hi" he said "yo"')

    def test_think_tags_and_speaker_prefix(self):
        self.assertEqual(self.p("<think>hmm</think>Bob: yo what's up"), "yo what's up")
        self.assertEqual(self.p("<think>cut off mid thought"), "")
        self.assertEqual(self.p("stray</think>kept"), "kept")

    def test_prefix_forms(self):
        self.assertEqual(self.p("[Bob] hi"), "hi")
        self.assertEqual(self.p("<bob> hi"), "hi")
        self.assertEqual(self.p("Warning: lava"), "Warning: lava")      # not this bot's name
        self.assertEqual(self.p("Bobby: hi"), "Bobby: hi")               # name must be followed by : or -

    def test_gesture_star_form_is_removed_and_reported(self):
        text, emote = pp.process_llm_response("*waves* hello", "Bob")
        self.assertEqual((text, emote), ("hello", "waves"))

    def test_bracket_and_slash_forms(self):
        self.assertEqual(self.p("hi there [emote:wave]"), "hi there")
        self.assertEqual(self.p("hi /wave there"), "hi there")
        self.assertEqual(self.p("and/or maybe"), "and/or maybe")         # not an emote: left alone

    def test_emphasis_keeps_the_word(self):
        self.assertEqual(self.p("that was *really* bad"), "that was really bad")

    def test_whole_line_stage_direction_is_dropped(self):
        self.assertEqual(self.p("*nods slowly at the guard*"), "")

    def test_one_word_at_the_edge_is_a_gesture_even_if_unknown(self):
        text, emote = pp.process_llm_response("*grunts* fine.", "Bob")
        self.assertEqual(text, "fine.")

    def test_markdown(self):
        self.assertEqual(self.p("**bold** and `code`"), "bold and code")
        self.assertEqual(self.p("- item"), "item")
        self.assertEqual(self.p("# Heading"), "Heading")
        self.assertEqual(self.p("a ```x = 1``` b"), "a b")
        # the emote extractor removes asterisk spans before StripMarkdown, so the markers go either way
        self.assertEqual(self.p("**bold** x", strip_markdown_on=False), "bold x")
        self.assertEqual(self.p("__u__ ~~s~~", strip_markdown_on=False), "__u__ ~~s~~")

    def test_decorative_unicode(self):
        self.assertEqual(self.p("It’s — ok… \U0001F600"), "It's - ok...")
        self.assertEqual(self.p("café"), "café")
        self.assertEqual(self.p("—", strip_decorative_on=False), "—")

    def test_whitespace_collapse(self):
        self.assertEqual(self.p("  a \n\n b\tc  "), "a b c")

    def test_clamp_prefers_a_sentence(self):
        self.assertEqual(pp.clamp_reply_length(b"aaaa. " + b"b" * 50, 30), b"aaaa.")

    def test_clamp_falls_back_to_word_boundary(self):
        # head = "one two three four f"; last space at index 18 > 10
        self.assertEqual(pp.clamp_reply_length(b"one two three four five six seven", 20), b"one two three four")

    def test_clamp_never_cuts_inside_a_utf8_sequence(self):
        # 'e acute' is 2 bytes; max_len lands between them, so the cut backs off one byte
        text = ("a" * 9 + "é" + "b" * 10).encode()
        self.assertEqual(pp.clamp_reply_length(text, 10), b"a" * 9)

    def test_clamp_zero_means_unlimited(self):
        self.assertEqual(pp.clamp_reply_length(b"x" * 500, 0), b"x" * 500)

    def test_max_length_applies_in_bytes(self):
        out = self.p("word " * 100, max_reply_length=40)
        self.assertLessEqual(len(out.encode()), 40)
        self.assertTrue(out.endswith("word"))

    def test_empty(self):
        self.assertEqual(self.p(""), "")


class ApiTests(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(provider="anthropic", url="https://example.invalid/v1/messages", model="m",
                    system_prompt="SYS", num_predict=100, temperature=0.8, api_key="sekrit")
        base.update(kw)
        return api.Settings(**base)

    def test_anthropic_request(self):
        r = api.build_anthropic_request(self.cfg(stop="a, b"), "hello")
        self.assertEqual(r["max_tokens"], 100)
        self.assertEqual(r["system"], "SYS")
        self.assertEqual(r["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(r["stop_sequences"], ["a", "b"])
        self.assertEqual(r["temperature"], 0.8)
        self.assertNotIn("thinking", r)
        self.assertNotIn("top_k", r)
        self.assertNotIn("top_p", r)

    def test_anthropic_defaults_and_omissions(self):
        r = api.build_anthropic_request(self.cfg(system_prompt="", num_predict=0, top_k=40), "x")
        self.assertEqual(r["max_tokens"], 1024)
        self.assertNotIn("system", r)
        self.assertEqual(r["top_k"], 40)

    def test_ollama_request_omits_default_sampling(self):
        r = api.build_ollama_request(self.cfg(provider="ollama"), "hello")
        self.assertEqual(r["options"], {"num_predict": 100})
        self.assertIs(r["think"], False)
        self.assertEqual(r["system"], "SYS")
        r = api.build_ollama_request(self.cfg(provider="ollama", temperature=0.5, num_ctx=2048), "hello")
        self.assertEqual(r["options"], {"num_predict": 100, "temperature": 0.5, "num_ctx": 2048})

    def test_openai_request(self):
        r = api.build_openai_request(self.cfg(provider="openai"), "hello")
        self.assertEqual([m["role"] for m in r["messages"]], ["system", "user"])
        self.assertEqual(r["max_tokens"], 100)
        self.assertIn("top_p", r)

    def test_headers(self):
        h = dict(api.build_request_headers(self.cfg()))
        self.assertEqual(h, {"anthropic-version": "2023-06-01", "x-api-key": "sekrit"})
        h = dict(api.build_request_headers(self.cfg(provider="openai")))
        self.assertEqual(h, {"Authorization": "Bearer sekrit"})
        h = dict(api.build_request_headers(self.cfg(provider="openai", api_key_header="authorization")))
        self.assertEqual(h, {"authorization": "Bearer sekrit"})

    def test_key_is_not_in_repr(self):
        self.assertNotIn("sekrit", repr(self.cfg()))

    def test_parse_anthropic(self):
        body = '{"content":[{"type":"text","text":"hi"},{"type":"thinking","thinking":"..."}],"usage":{"input_tokens":10,"output_tokens":3}}'
        p = api.parse_anthropic_body(body)
        self.assertEqual((p.text, p.thinking, p.input_tokens, p.output_tokens), ("hi", "...", 10, 3))
        self.assertIn("bad", api.parse_anthropic_body('{"type":"error","error":{"message":"bad"}}').error)

    def test_parse_ollama_and_openai(self):
        p = api.parse_ollama_body('{"response":"hi","eval_count":4,"prompt_eval_count":9}')
        self.assertEqual((p.text, p.input_tokens, p.output_tokens), ("hi", 9, 4))
        p = api.parse_openai_body('{"choices":[{"message":{"content":"yo"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}')
        self.assertEqual((p.text, p.input_tokens, p.output_tokens), ("yo", 5, 2))

    def test_resolve_key(self):
        self.assertEqual(api.resolve_api_key("A", "anthropic", {"ANTHROPIC_API_KEY": " k "}), "k")
        self.assertEqual(api.resolve_api_key("A", "ollama", {"ANTHROPIC_API_KEY": "k"}), "")
        self.assertEqual(api.resolve_api_key("A", "anthropic", {"A": "x", "ANTHROPIC_API_KEY": "k"}), "x")

    def test_retry_on_429_then_success(self):
        calls, sleeps = [], []

        class Resp:
            status = 200

            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"content":[{"type":"text","text":"ok"}]}'

        def opener(req, timeout):
            calls.append(req)
            if len(calls) == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "slow", {"Retry-After": "3"}, None)
            return Resp()

        res = api.query(self.cfg(rate_limit_retries=2), "x", opener=opener, sleep=sleeps.append)
        self.assertTrue(res.ok)
        self.assertEqual((res.text, len(calls), sleeps), ("ok", 2, [3]))

    def test_retries_are_bounded(self):
        calls, sleeps = [], []

        def opener(req, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 529, "busy", {}, None)

        res = api.query(self.cfg(rate_limit_retries=2, rate_limit_max_wait=15), "x", opener=opener,
                        sleep=sleeps.append)
        self.assertFalse(res.ok)
        self.assertEqual((len(calls), sleeps), (3, [2, 4]))

    def test_non_retryable_status(self):
        calls = []

        def opener(req, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

        res = api.query(self.cfg(), "x", opener=opener, sleep=lambda s: None)
        self.assertEqual((res.ok, len(calls)), (False, 1))


class MetricsTests(unittest.TestCase):
    def test_metrics(self):
        replies = ["lol nice", "Lol again", "yo what", "Fine weather today", "Fine weather today!"]
        m = metrics.compute(replies, 1, metrics.DEFAULT_OPENERS, ["weather"])
        self.assertEqual(m["slang_open_pct"], 60.0)
        self.assertEqual(m["distinct_opener_ratio"], 0.6)                  # lol, lol, yo, fine, fine -> 3 of 5
        self.assertEqual(m["failures"], 1)

    def test_distinct_opener_counts_words(self):
        m = metrics.compute(["lol a", "Lol b", "yo c", "fine d"], 0, ["lol"], [])
        self.assertEqual(m["distinct_opener_ratio"], 0.75)                  # lol, yo, fine over 4

    def test_banned_whole_word_and_phrase(self):
        m = metrics.compute(["Lich King wins", "the lichking", "I like Arena. arena!"], 0, [],
                            ["lich king", "arena"])
        self.assertEqual(m["banned_hits"], 3)
        self.assertEqual(dict(m["banned_top"]), {"arena": 2, "lich king": 1})

    def test_near_duplicates(self):
        m = metrics.compute(["see you in the tavern", "see you in the tavern!", "completely different"], 0, [], [])
        self.assertAlmostEqual(m["near_dup_pct"], round(100 / 3, 1))

    def test_word_stats(self):
        m = metrics.compute(["a b", "a b c d"], 0, [], [])
        self.assertEqual((m["words_mean"], m["words_stdev"]), (3.0, 1.0))


DIST = HERE.parent.parent / "conf" / "mod_ollama_chat.conf.dist"


@unittest.skipUnless(DIST.exists(), "needs conf/mod_ollama_chat.conf.dist")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conf = build_conf.parse_conf_text(DIST.read_text(encoding="utf-8"))
        cls.cfg = render.Config(cls.conf)

    BOT = dict(name="Thalrin", level=12, **{"class": "warrior"}, race="dwarf", gender="Male", role="arms warrior",
               faction="Alliance", guild="No Guild", group_status="Solo", gold=3, area="Kharanos",
               zone="Dun Morogh", map="Eastern Kingdoms", personality="CASUAL")
    PLAYER = dict(name="Aveline", level=12, **{"class": "mage"}, race="human", gender="Female", role="frost mage",
                  faction="Alliance", guild="No Guild", group_status="Solo", gold=5, distance=8.5)

    def reply_case(self, **kw):
        case = dict(id="t", path="reply", message="hi", bot=self.BOT, player=self.PLAYER)
        case.update(kw)
        return case

    def test_reply_matches_the_template_by_hand(self):
        cfg = render.Config({**self.conf, "OllamaChat.ChatPromptTemplate": "{bot_name}|{player_message}|{extra_info}|{chat_history}|{sentiment_info}|{bot_personality_name}",
                             "OllamaChat.ChatExtraInfoTemplate": "{bot_race}/{player_distance}/{bot_zone}",
                             "OllamaChat.GestureChance": "0"})
        r = render.render_case(cfg, self.reply_case(), {}, None, 0)
        self.assertEqual(r.prompt, "Thalrin|hi|dwarf/8.5/Dun Morogh|||default")

    def test_history_block_and_missing_footer_without_history(self):
        cfg = render.Config({**self.conf, "OllamaChat.ChatPromptTemplate": "[{chat_history}]",
                             "OllamaChat.GestureChance": "0"})
        none = render.render_case(cfg, self.reply_case(), {}, None, 0).prompt
        self.assertEqual(none, "[]")
        hist = render.render_case(cfg, self.reply_case(history=[{"player": "a", "bot": "b"}]), {}, None, 0).prompt
        self.assertEqual(hist, "[Recent chats with Aveline. Use only for context. Reply to the new message."
                               "Aveline said: a\\nYou said: b\\nNEW MESSAGE from Aveline: hi]")

    def test_missing_case_value_names_case_and_placeholder(self):
        bot = {k: v for k, v in self.BOT.items() if k != "race"}
        with self.assertRaises(render.CaseError) as ctx:
            render.render_case(self.cfg, self.reply_case(bot=bot), {}, None, 0)
        self.assertIn("'t'", str(ctx.exception))
        self.assertIn("bot_race", str(ctx.exception))

    def test_unknown_placeholder_in_config_gives_format_error(self):
        cfg = render.Config({**self.conf, "OllamaChat.ChatPromptTemplate": "{nope}"})
        r = render.render_case(cfg, self.reply_case(), {}, None, 0)
        self.assertTrue(r.format_error)

    def test_personality_only_when_enabled(self):
        pers = {"CASUAL": "Chill."}
        cfg_on = render.Config({**self.conf, "OllamaChat.EnableRPPersonalities": "1",
                                "OllamaChat.ChatPromptTemplate": "{bot_personality}:{bot_personality_name}",
                                "OllamaChat.GestureChance": "0"})
        self.assertEqual(render.render_case(cfg_on, self.reply_case(), pers, None, 0).prompt, "Chill.:CASUAL")
        cfg_off = render.Config({**cfg_on.values, "OllamaChat.EnableRPPersonalities": "0"})
        out = render.render_case(cfg_off, self.reply_case(), pers, None, 0).prompt
        self.assertEqual(out, self.conf["OllamaChat.DefaultPersonalityPrompt"] + ":default")

    def test_random_is_deterministic_per_case_and_sample(self):
        case = dict(id="r", path="random", bot=self.BOT,
                    random=dict(env_template="OllamaChat.EnvCommentZoneLandmark",
                                env_vars=dict(area_name="Kharanos", zone_name="Dun Morogh")))
        a = render.render_case(self.cfg, case, {}, None, 1).prompt
        b = render.render_case(self.cfg, case, {}, None, 1).prompt
        self.assertEqual(a, b)
        self.assertNotEqual({render.render_case(self.cfg, case, {}, None, s).prompt for s in range(8)}, {a})

    def test_random_explicit_variation(self):
        cfg = render.Config({**self.conf, "OllamaChat.RandomChatterPromptTemplate": "T {environment_info}",
                             "OllamaChat.GestureChance": "0"})
        case = dict(id="r", path="random", bot=self.BOT,
                    random=dict(environment_info="You are bored.", variation="Say hi."))
        self.assertEqual(render.render_case(cfg, case, {}, None, 0).prompt, "T You are bored. Say hi.")

    def test_event_path(self):
        cfg = render.Config({**self.conf, "OllamaChat.EventChatterPromptTemplate": "{actor_name} {event_type} {event_detail} {sentiment_info}",
                             "OllamaChat.GestureChance": "0", "OllamaChat.EnableSentimentTracking": "0"})
        case = dict(id="e", path="event", bot=self.BOT,
                    event=dict(actor_name="Grogg", event_type="leveled up", event_detail="29"))
        self.assertEqual(render.render_case(cfg, case, {}, None, 0).prompt, "Grogg leveled up 29 ")

    def test_rag_block_appended_with_newline(self):
        s = rag.RagSystem([rag.RagEntry("a", "Mage", "frost mages slow enemies")])
        cfg = render.Config({**self.conf, "OllamaChat.EnableRAG": "1", "OllamaChat.ChatPromptTemplate": "P",
                             "OllamaChat.RAGPromptTemplate": "INFO {rag_info}", "OllamaChat.GestureChance": "0"})
        r = render.render_case(cfg, self.reply_case(message="tell me about mage"), {}, s, 0)
        self.assertEqual(r.prompt, "PINFO - Mage: frost mages slow enemies\n")
        self.assertEqual(r.rag[0]["id"], "a")

    def test_float_formatting_matches_fmt(self):
        self.assertEqual(render.f32_repr(8.5), "8.5")
        self.assertEqual(render.f32_repr(4.0), "4")
        self.assertEqual(render.f32_repr(0.05), "0.05")
        self.assertEqual(render.f32_repr(-1.0), "-1")


if __name__ == "__main__":
    unittest.main()
