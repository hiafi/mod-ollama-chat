#!/usr/bin/env python3
"""Retrieval-only check of a RAG directory: no API calls.

    python3 custom/eval/rag_probe.py [--rag-dir custom/rag] [--probes custom/eval/rag-probes.toml]
                                      [--cases custom/eval/cases] [--top 5] [--threshold 0.3 --max-items 3]
                                      [--sweep] [--ignore-common-words 0|1] [--boost 0.5]

For every probe message prints the top entries with their raw scores and where the expected entry ranked. With
--sweep prints, for a grid of thresholds (and, with --boosts, location boosts), how many chit-chat messages retrieve nothing, how many expected entries land in
the top 3 / top max-items, and how many unrelated entries are injected on average. Scores are those of the module's
cosine (tools/eval/rag.py, sparse form).
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools" / "eval"))
import rag  # noqa: E402


def load_probes(probes_path: Path, cases_dir: Path):
    probes = tomllib.loads(probes_path.read_text(encoding="utf-8"))["probe"]
    out = []
    for p in probes:
        if p["group"] == "case":
            case = tomllib.loads((cases_dir / f"{p['case']}.toml").read_text(encoding="utf-8"))
            bot = case.get("bot", {})
            p = {"zone": bot.get("zone", ""), "area": bot.get("area", ""), **p, "message": case["message"]}
        out.append({"group": p["group"], "message": p["message"], "expect": p.get("expect", []),
                    "note": p.get("note", ""), "label": p.get("case", ""),
                    "locations": [x for x in (p.get("area", ""), p.get("zone", "")) if x]})
    # every other reply case (the q3-* ones repeat probes above): retrieval is not expected, and is reported as 'case (no expectation)'
    listed = {p["label"] for p in out if p["label"]}
    for f in sorted(cases_dir.glob("*.toml")):
        case = tomllib.loads(f.read_text(encoding="utf-8"))
        if case.get("path") == "reply" and case["id"] not in listed and not case["id"].startswith("q3-"):
            bot = case.get("bot", {})
            out.append({"group": "case-nothing", "message": case["message"], "expect": [], "note": "", "label": case["id"],
                        "locations": [x for x in (bot.get("area", ""), bot.get("zone", "")) if x]})
    return out


def rank_of(results, expect):
    ids = [r.entry.id for r in results]
    best = [ids.index(e) + 1 for e in expect if e in ids]
    return min(best) if best else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-dir", type=Path, default=HERE.parent / "rag")
    ap.add_argument("--probes", type=Path, default=HERE / "rag-probes.toml")
    ap.add_argument("--cases", type=Path, default=HERE / "cases")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max-items", type=int, default=3)
    ap.add_argument("--ignore-common-words", type=int, choices=(0, 1), default=1,
                    help="OllamaChat.RAGIgnoreCommonWords (default 1)")
    ap.add_argument("--boost", type=float, default=0.0, help="OllamaChat.RAGLocationBoost for the listing")
    ap.add_argument("--boosts", default="0,0.25,0.5,1.0", help="boosts for --sweep")
    ap.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5", help="thresholds for --sweep")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="skip the per-message listing")
    args = ap.parse_args(argv)

    system = rag.RagSystem.from_directory(args.rag_dir, bool(args.ignore_common_words))
    probes = load_probes(args.probes, args.cases)
    print(f"{len(system.entries)} entries, {len(system.vocabulary)} terms, {len(probes)} probes, ignore common words={args.ignore_common_words}\n")

    scored = []  # (probe, {boost: ranked results above no threshold})
    boosts = sorted({args.boost, *(float(b) for b in args.boosts.split(","))}) if args.sweep else [args.boost]
    for p in probes:
        # threshold 0 lists everything, ranked by the boost; a threshold applies later on the raw similarity
        by_boost = {b: system.retrieve(p["message"], len(system.entries), 0.0, p["locations"], b) for b in boosts}
        top = by_boost[args.boost][:max(args.top, 10)]
        scored.append((p, by_boost))
        if args.quiet:
            continue
        print(f"[{p['group']}] {p['message']!r}  expect={p['expect'] or 'nothing'}  rank={rank_of(top, p['expect'])}"
              + (f"  ({p['note']})" if p["note"] else ""))
        for r in top[:args.top]:
            mark = "*" if r.entry.id in p["expect"] else " "
            print(f"   {mark} {r.similarity:.3f}  {r.entry.id:<16} {r.entry.title[:70]}")
    if not args.sweep:
        return 0

    print("\nthreshold x location boost sweep (max items = top-N injected)")
    print("| threshold | boost | chit-chat empty | no-expect cases empty | where top1 | where top3 | command top1 | "
          "rework top1 | avg injected (answerable) |")
    print("|---|---|---|---|---|---|---|---|---|")
    groups = lambda g: [(p, t) for p, t in scored if p["group"] == g]
    for thr in (float(x) for x in args.thresholds.split(",")):
        for b in boosts:
            def kept(t, n=args.max_items):
                return [r for r in t[b] if r.similarity >= thr][:n]
            chit, nothing = groups("chitchat"), groups("case-nothing")
            where, cmd, rew = groups("where"), groups("command"), groups("rework")
            c_empty = sum(1 for _p, t in chit if not kept(t))
            n_empty = sum(1 for _p, t in nothing if not kept(t))
            top1 = lambda grp: sum(1 for p, t in grp if rank_of(kept(t, 1), p["expect"]))
            w3 = sum(1 for p, t in where if rank_of(kept(t, 3), p["expect"]))
            answerable = where + cmd + rew
            inj = sum(len(kept(t)) for _p, t in answerable) / max(len(answerable), 1)
            print(f"| {thr:.2f} | {b:g} | {c_empty}/{len(chit)} | {n_empty}/{len(nothing)} | {top1(where)}/{len(where)} | "
                  f"{w3}/{len(where)} | {top1(cmd)}/{len(cmd)} | {top1(rew)}/{len(rew)} | {inj:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
