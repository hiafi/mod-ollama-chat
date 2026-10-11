#!/usr/bin/env python3
"""Render prompts exactly as the module does, send them to the model, compare two override sets.

    run_eval.py --dist conf/mod_ollama_chat.conf.dist --candidate new.toml [--baseline old.toml] \\
                --cases tools/eval/cases-example [--samples 3] [--dry-run]

The baseline defaults to the bare .dist. Both variants get the same cases, the
same samples and the same seeded dice (variation picks, gesture hint), so any
difference in the replies comes from the config. See tools/eval/README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import api                                   # noqa: E402
import build_conf                            # noqa: E402
import metrics                               # noqa: E402
import postprocess                           # noqa: E402
import render                                # noqa: E402

MODULE_ROOT = HERE.parent.parent

# USD per million tokens (input, output). Matched by model-name prefix; check the provider's
# current price list. An unknown model simply gets no estimate.
PRICES = [
    ("claude-haiku-4-5", (1.0, 5.0)),
    ("claude-sonnet-4-5", (3.0, 15.0)),
    ("claude-opus-4-5", (5.0, 25.0)),
]


def price_for(model: str):
    for prefix, price in PRICES:
        if model.startswith(prefix):
            return price
    return None


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Variant:
    name: str
    overrides: dict
    cfg: render.Config
    settings: api.Settings


@dataclass
class Job:
    case: dict
    variant: Variant
    sample: int
    rendered: render.Rendered
    result: api.ApiResult | None = None
    processed: str = ""
    emote: str = ""
    skipped: str = ""


def load_cases(directory: Path) -> list[dict]:
    files = sorted(directory.glob("*.toml"))
    if not files:
        raise SystemExit(f"error: no *.toml cases in {directory}")
    cases = []
    seen: set[str] = set()
    for f in files:
        with open(f, "rb") as fh:
            case = tomllib.load(fh)
        cid = case.get("id")
        if not isinstance(cid, str) or not cid:
            raise SystemExit(f"error: {f.name}: missing 'id'")
        if cid in seen:
            raise SystemExit(f"error: duplicate case id '{cid}' ({f.name})")
        seen.add(cid)
        case["_file"] = f.name
        cases.append(case)
    return cases


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip() or "_(empty)_"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path, help="overrides TOML under test")
    ap.add_argument("--baseline", type=Path, help="overrides TOML to compare against (default: bare .dist)")
    ap.add_argument("--cases", required=True, type=Path, help="directory of case *.toml files")
    ap.add_argument("--personalities", type=Path, default=HERE / "personalities-stock.toml")
    ap.add_argument("--rag-dir", type=Path, help="directory of RAG *.json (default: RAGDataPath from the config)")
    ap.add_argument("--banned-terms", type=Path)
    ap.add_argument("--openers", default=",".join(metrics.DEFAULT_OPENERS),
                    help="comma-separated openers counted by slang_open_pct")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "reports")
    ap.add_argument("--api-key-env", default="OLLAMA_CHAT_API_KEY",
                    help="environment variable holding the API key (anthropic also falls back to ANTHROPIC_API_KEY)")
    ap.add_argument("--dry-run", action="store_true", help="render prompts to files, make no network call")
    ap.add_argument("--yes", action="store_true", help="skip the Proceed? prompt")
    args = ap.parse_args(argv)

    if args.samples < 1:
        raise SystemExit("error: --samples must be at least 1")

    dist_text = args.dist.read_text(encoding="utf-8")
    try:
        cand_over = build_conf.load_overrides(args.candidate)
        base_over = build_conf.load_overrides(args.baseline) if args.baseline else {}
        cand_conf = build_conf.effective_config(dist_text, cand_over)
        base_conf = build_conf.effective_config(dist_text, base_over)
    except (ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with open(args.personalities, "rb") as fh:
        personalities = tomllib.load(fh)
    cases = load_cases(args.cases)
    banned = metrics.load_banned_terms(args.banned_terms)
    openers = [o.strip() for o in args.openers.split(",") if o.strip()]

    key_cache: dict[str, str] = {}

    def make_variant(name: str, overrides: dict, conf: dict) -> Variant:
        cfg = render.Config(conf)
        provider = api.parse_provider(conf.get("OllamaChat.Provider", "ollama"))
        key = key_cache.setdefault(provider, api.resolve_api_key(args.api_key_env, provider))
        return Variant(name, overrides, cfg, api.settings_from_config(conf, key))

    variants = [make_variant("baseline", base_over, base_conf), make_variant("candidate", cand_over, cand_conf)]

    # ---- render everything up front: a bad case should fail before any spend ----
    jobs: list[Job] = []
    try:
        rag_by_variant = {v.name: render.load_rag(v.cfg, args.rag_dir, MODULE_ROOT) for v in variants}
        for case in cases:
            for sample in range(args.samples):
                for v in variants:
                    rendered = render.render_case(v.cfg, case, personalities, rag_by_variant[v.name], sample)
                    jobs.append(Job(case, v, sample, rendered))
    except (render.CaseError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # ---- spend guard ----
    calls = sum(1 for j in jobs if not j.rendered.format_error)
    in_chars = sum(len(j.rendered.prompt) + len(j.rendered.system_prompt) for j in jobs if not j.rendered.format_error)
    est_in = in_chars / 4
    est_out = sum(j.variant.settings.num_predict or 1024 for j in jobs if not j.rendered.format_error)
    by_path: dict[str, int] = {}
    for j in jobs:
        by_path[j.rendered.path] = by_path.get(j.rendered.path, 0) + 1
    print(f"{len(cases)} cases x {args.samples} samples x {len(variants)} variants = {len(jobs)} prompts "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(by_path.items()))})")
    skipped_fmt = len(jobs) - calls
    if skipped_fmt:
        print(f"note: {skipped_fmt} prompt(s) render to '[Format Error]' and will be reported as failures, not sent")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / stamp

    if args.dry_run:
        pdir = run_dir / "prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        for j in jobs:
            (pdir / f"{j.case['id']}-{j.variant.name}-{j.sample}.txt").write_text(
                (f"[system]\n{j.rendered.system_prompt}\n\n" if j.rendered.system_prompt else "") +
                f"[user]\n{j.rendered.prompt}\n", encoding="utf-8")
        print(f"dry run: wrote {len(jobs)} prompts to {pdir}; no network calls made")
        return 0

    print(f"{calls} API call(s) planned (limit {args.max_calls})")
    models = {v.settings.model for v in variants}
    for model in sorted(models):
        price = price_for(model)
        if price:
            cost = est_in / 1e6 * price[0] + est_out / 1e6 * price[1]
            print(f"  estimated cost for {model}: <= ${cost:.3f} "
                  f"(~{int(est_in)} input tokens at chars/4, up to {est_out} output tokens)")
        else:
            print(f"  no price on file for model '{model}'; ~{int(est_in)} input tokens (chars/4), "
                  f"up to {est_out} output tokens")
    if calls > args.max_calls:
        print(f"error: {calls} calls exceeds --max-calls {args.max_calls}; hard stop, nothing sent", file=sys.stderr)
        return 2
    for v in variants:
        if v.settings.provider != "ollama" and not v.settings.api_key:
            print(f"error: no API key for the {v.settings.provider} provider "
                  f"(set {args.api_key_env}); nothing sent", file=sys.stderr)
            return 2
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted; nothing sent")
            return 2

    # ---- run ----
    def run(job: Job) -> Job:
        if job.rendered.format_error:
            job.skipped = "prompt rendered to [Format Error]"
            return job
        s = job.variant.settings
        s_local = api.Settings(**{**s.__dict__, "system_prompt": job.rendered.system_prompt})
        job.result = api.query(s_local, job.rendered.prompt)
        if job.result.ok:
            cfg = job.variant.cfg
            job.processed, job.emote = postprocess.process_llm_response(
                job.result.text, str(job.case["bot"]["name"]),
                strip_markdown_on=cfg.boolean("StripMarkdown", True),
                strip_decorative_on=cfg.boolean("StripDecorativeUnicode", True),
                max_reply_length=cfg.integer("MaxReplyLength", 240))
        return job

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for fut in cf.as_completed(futures):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}", flush=True)

    # ---- outputs ----
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "raw.jsonl", "w", encoding="utf-8") as fh:
        for j in jobs:
            r = j.result
            fh.write(json.dumps({
                "case": j.case["id"], "variant": j.variant.name, "sample": j.sample,
                "prompt_hash": sha(j.rendered.prompt), "system_hash": sha(j.rendered.system_prompt),
                "raw": r.text if r else None, "processed": j.processed or None, "emote": j.emote or None,
                "input_tokens": r.input_tokens if r else None, "output_tokens": r.output_tokens if r else None,
                "latency_ms": r.latency_ms if r else None, "status": r.status if r else None,
                "error": (r.error if r and not r.ok else None) or j.skipped or None,
                "rag": j.rendered.rag,
            }, ensure_ascii=False) + "\n")

    def failure_of(j: Job) -> str:
        if j.skipped:
            return j.skipped
        if j.result is None:
            return "no result"
        if not j.result.ok:
            return j.result.error or "request failed"
        if not j.processed:
            return "empty reply after clean-up"
        return ""

    stats = {}
    for v in variants:
        vj = [j for j in jobs if j.variant is v]
        replies = [j.processed for j in vj if not failure_of(j)]
        stats[v.name] = metrics.compute(replies, sum(1 for j in vj if failure_of(j)), openers, banned)

    ok_results = [j.result for j in jobs if j.result and j.result.ok]
    in_toks = [r.input_tokens for r in ok_results if r.input_tokens is not None]
    out_toks = [r.output_tokens for r in ok_results if r.output_tokens is not None]

    md: list[str] = []
    md.append(f"# Prompt eval {stamp}\n")
    md.append(f"* candidate: `{args.candidate}`  baseline: `{args.baseline or 'bare .dist'}`")
    md.append(f"* cases: {len(cases)}, samples: {args.samples}, calls made: {sum(1 for j in jobs if j.result)}")
    md.append("* models: " + ", ".join(f"{v.name}={v.settings.provider}/{v.settings.model}" for v in variants))
    if in_toks and out_toks:
        md.append(f"* mean tokens per call: {sum(in_toks) / len(in_toks):.0f} in / {sum(out_toks) / len(out_toks):.0f} out")
    md.append("")
    md.append("## Metrics\n")
    md.append("| metric | baseline | candidate |")
    md.append("|---|---|---|")
    for m in ["replies", "slang_open_pct", "distinct_opener_ratio", "words_mean", "words_stdev",
              "near_dup_pct", "banned_hits", "failures"]:
        md.append(f"| {m} | {stats['baseline'][m]} | {stats['candidate'][m]} |")
    for v in ("baseline", "candidate"):
        top = ", ".join(f"{t} x{n}" for t, n in stats[v]["banned_top"]) or "none"
        md.append(f"| banned top ({v}) | {top if v == 'baseline' else ''} | {top if v == 'candidate' else ''} |")
    md.append("")
    md.append("## Cases\n")
    for case in cases:
        cjobs = [j for j in jobs if j.case is case]
        first = cjobs[0].rendered
        md.append(f"### {case['id']}  ({case['path']})\n")
        if case.get("notes"):
            md.append(f"_Notes:_ {case['notes']}\n")
        md.append(f"_Situation:_ {first.summary}\n")
        md.append("| # | baseline | candidate |")
        md.append("|---|---|---|")
        for s in range(args.samples):
            cells = []
            for name in ("baseline", "candidate"):
                j = next(x for x in cjobs if x.sample == s and x.variant.name == name)
                f = failure_of(j)
                cells.append(f"**FAILED:** {md_cell(f)}" if f else md_cell(j.processed))
            md.append(f"| {s + 1} | {cells[0]} | {cells[1]} |")
        rag_hits = {tuple((h['id'], h['score']) for h in x.rendered.rag) for x in cjobs if x.rendered.rag}
        for hits in sorted(rag_hits):
            md.append(f"\nRAG retrieved: {', '.join(f'{i} ({sc})' for i, sc in hits)}")
        md.append("")
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"report: {run_dir / 'report.md'}")
    print(f"raw:    {run_dir / 'raw.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
