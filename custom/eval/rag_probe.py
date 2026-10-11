#!/usr/bin/env python3
"""Retrieval-only check of a RAG directory: no API calls.

    python3 custom/eval/rag_probe.py [--rag-dir custom/rag] [--probes custom/eval/rag-probes.toml]
                                      [--cases custom/eval/cases] [--top 5] [--threshold 0.3 --max-items 3]
                                      [--sweep] [--ignore-common-words 0|1]

For every probe message prints the top entries with their raw scores and where the expected entry ranked. With
--sweep prints, for a grid of thresholds, how many chit-chat messages retrieve nothing, how many expected entries land in
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
            p = {**p, "message": case["message"]}
        out.append({"group": p["group"], "message": p["message"], "expect": p.get("expect", []),
                    "note": p.get("note", ""), "label": p.get("case", "")})
    # every other reply case (the q3-* ones repeat probes above): retrieval is not expected, and is reported as 'case (no expectation)'
    listed = {p["label"] for p in out if p["label"]}
    for f in sorted(cases_dir.glob("*.toml")):
        case = tomllib.loads(f.read_text(encoding="utf-8"))
        if case.get("path") == "reply" and case["id"] not in listed and not case["id"].startswith("q3-"):
            out.append({"group": "case-nothing", "message": case["message"], "expect": [], "note": "", "label": case["id"]})
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
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="skip the per-message listing")
    args = ap.parse_args(argv)

    system = rag.RagSystem.from_directory(args.rag_dir, bool(args.ignore_common_words))
    probes = load_probes(args.probes, args.cases)
    print(f"{len(system.entries)} entries, {len(system.vocabulary)} terms, {len(probes)} probes, ignore common words={args.ignore_common_words}\n")

    scored = []
    for p in probes:
        top = system.retrieve(p["message"], max(args.top, 10), 0.0)
        scored.append((p, top))
        if args.quiet:
            continue
        print(f"[{p['group']}] {p['message']!r}  expect={p['expect'] or 'nothing'}  rank={rank_of(top, p['expect'])}"
              + (f"  ({p['note']})" if p["note"] else ""))
        for r in top[:args.top]:
            mark = "*" if r.entry.id in p["expect"] else " "
            print(f"   {mark} {r.similarity:.3f}  {r.entry.id:<16} {r.entry.title[:70]}")
    if not args.sweep:
        return 0

    print("\nthreshold sweep (max items = top-N injected)")
    print("| threshold | chit-chat empty | no-expect cases empty | where top3 | where top1 | command top3 | "
          "avg injected (where) | avg injected (all non-chitchat) |")
    print("|---|---|---|---|---|---|---|---|")
    groups = lambda g: [(p, t) for p, t in scored if p["group"] == g]
    for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        def kept(t, n=args.max_items):
            return [r for r in t if r.similarity >= thr][:n]
        chit = groups("chitchat")
        nothing = groups("case-nothing")
        where, cmd = groups("where"), groups("command")
        c_empty = sum(1 for _p, t in chit if not kept(t))
        n_empty = sum(1 for _p, t in nothing if not kept(t))
        w3 = sum(1 for p, t in where if rank_of(kept(t, 3), p["expect"]))
        w1 = sum(1 for p, t in where if rank_of(kept(t, 1), p["expect"]))
        m3 = sum(1 for p, t in cmd if rank_of(kept(t, 3), p["expect"]))
        inj_w = sum(len(kept(t)) for _p, t in where) / max(len(where), 1)
        non_chit = [(p, t) for p, t in scored if p["group"] != "chitchat"]
        inj_all = sum(len(kept(t)) for _p, t in non_chit) / max(len(non_chit), 1)
        print(f"| {thr:.2f} | {c_empty}/{len(chit)} | {n_empty}/{len(nothing)} | {w3}/{len(where)} | {w1}/{len(where)} | "
              f"{m3}/{len(cmd)} | {inj_w:.1f} | {inj_all:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
