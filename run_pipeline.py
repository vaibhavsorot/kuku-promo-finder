"""Command-line entry point.

  python run_pipeline.py --input "data/input/My Show"                 # all episodes, provider from .env
  python run_pipeline.py --input show.zip --episodes 1-5 --no-cross   # quick test
  python run_pipeline.py --input ... --provider replay                # re-run offline from cache
  python run_pipeline.py --input ... --inspect                        # parse only, no LLM
"""
from __future__ import annotations

import argparse
import json
import sys

import config
from src.ingestion import load_show
from src.pipeline import run_pipeline


def parse_range(text: str | None) -> list[int] | None:
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find and rank microdrama promos in a screenplay pack.")
    ap.add_argument("--input", default=str(config.DEFAULT_INPUT_DIR), help="folder or .zip of episode scripts")
    ap.add_argument("--provider", default=None, help=f"override LLM_PROVIDER (now: {config.LLM_PROVIDER})")
    ap.add_argument("--model", default=None, help="override LLM_MODEL")
    ap.add_argument("--episodes", default=None, help="e.g. 1-5,9")
    ap.add_argument("--no-cross", action="store_true", help="skip cross-episode montages")
    ap.add_argument("--blind-top", type=int, default=-1, help="blind-viewer test: -1 = all candidates (default), N = top N, 0 = off")
    ap.add_argument("--inspect", action="store_true", help="only parse + report input structure")
    a = ap.parse_args(argv)

    if a.inspect:
        rep = load_show(a.input)
        print(json.dumps({"show": rep.show_title, "stats": rep.stats, "errors": rep.errors,
                          "warnings": rep.warnings[:50]}, indent=2))
        return 0 if rep.episodes else 1

    def progress(stage, done, total, msg):
        print(f"[{stage} {done}/{total}] {msg}", flush=True)

    res = run_pipeline(a.input, a.provider, a.model, parse_range(a.episodes), not a.no_cross, a.blind_top, progress)
    print(json.dumps(res["meta"]["counts"], indent=2))
    print("outputs:", res["dirs"]["base"])
    top = [p for p in res["promos"] if p.rank][:10]
    for p in top:
        segs = " + ".join(f"E{s.episode} {s.start:.1f}-{s.end:.1f}s" for s in p.segments)
        print(f"#{p.rank:>2} {p.final_score:5.1f} {p.promo_id:<8} {p.duration:4.0f}s  {segs}  | {p.hook_text[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
