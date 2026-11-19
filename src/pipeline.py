"""End-to-end orchestration: ingest -> generate (LLM) -> build/validate/score (code) -> evaluate -> export.

Output layout (one folder per show so weekly shows never collide):

outputs/<show-slug>/
  processed/       ingest_report.json, episodes.json (parsed screenplay)
  raw_outputs/     verbatim LLM responses, one file per call (unedited)
  candidates/      per-episode candidate promos + moments, cross_episode.json
  final_promos/    promos.json, promos.csv, ranked_promos.md
  evaluation/      quality_report.json/.md, manual_review_sheet.csv
  run_manifest.json, run.log
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import config
from .candidate_generator import EpisodeResult, generate_cross_episode, generate_for_episode
from .evaluation import automated_report, blind_test, report_markdown, write_review_sheet
from .exporter import promo_record, write_all
from .ingestion import IngestReport, load_show, slugify
from .llm_provider import LLMClient, LLMError, get_provider
from .models import Episode, Issue, Promo, Segment
from .scoring import rank_promos
from .validation import validate_promo

Progress = Callable[[str, int, int, str], None]
log = logging.getLogger("promo_finder")


def _noop(stage, done, total, msg):
    pass


def show_dirs(slug: str, root: Path = config.OUTPUT_ROOT) -> dict[str, Path]:
    base = root / slug
    d = {k: base / k for k in ("processed", "raw_outputs", "candidates", "final_promos", "evaluation")}
    d["base"] = base
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _setup_log(path: Path):
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            log.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    if not any(type(h) is logging.StreamHandler for h in log.handlers):
        log.addHandler(logging.StreamHandler())


def run_pipeline(input_path: str | Path, provider_name: str | None = None, model: str | None = None,
                 episodes: list[int] | None = None, cross_episode: bool = True, blind_top_n: int = -1,
                 progress: Progress = _noop, output_root: Path = config.OUTPUT_ROOT) -> dict:
    t0 = time.time()
    ingest: IngestReport = load_show(input_path, extract_root=config.ROOT / "data" / "uploads")
    if not ingest.episodes:
        raise ValueError("no episodes could be parsed: " + "; ".join(ingest.errors))
    slug = slugify(ingest.show_title)
    selected = sorted(e.number for e in ingest.episodes if not episodes or e.number in episodes)
    if episodes and len(selected) < len(ingest.episodes):
        # a partial run must never overwrite the full-show results
        slug += f"__episodes-{selected[0]}-{selected[-1]}" if selected else "__episodes-none"
    dirs = show_dirs(slug, output_root)
    _setup_log(dirs["base"] / "run.log")
    eps_all = ingest.episodes
    eps = [e for e in eps_all if not episodes or e.number in episodes]
    log.info("ingested %d episodes (%d selected) from %s", len(eps_all), len(eps), ingest.input_path)
    (dirs["processed"] / "ingest_report.json").write_text(json.dumps(
        {"show": ingest.show_title, "input": ingest.input_path, "stats": ingest.stats,
         "errors": ingest.errors, "warnings": ingest.warnings}, indent=2), encoding="utf-8")
    (dirs["processed"] / "episodes.json").write_text(json.dumps(
        [asdict(e) for e in eps_all], indent=1, ensure_ascii=False), encoding="utf-8")
    progress("ingest", 1, 1, f"parsed {len(eps_all)} episodes, {ingest.stats['blocks']} blocks")

    provider = get_provider(provider_name, model)
    client = LLMClient(provider, raw_dir=dirs["raw_outputs"])
    log.info("provider %s", provider.describe())

    # ---- pass 1: per-episode (parallel; each result persisted as it lands -> resumable)
    results: list[EpisodeResult] = []
    errors: dict[int, str] = {}
    by_no = {e.number: e for e in eps_all}
    with ThreadPoolExecutor(max_workers=max(1, config.LLM_MAX_WORKERS)) as pool:
        futs = {pool.submit(generate_for_episode, client, e, ingest.show_title): e for e in eps}
        for i, fut in enumerate(as_completed(futs), start=1):
            e = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                (dirs["candidates"] / f"E{e.number:03d}.json").write_text(json.dumps({
                    "episode": e.number, "summary": r.summary, "character_notes": r.character_notes,
                    "moments": r.moments, "promos": [promo_record(p) for p in r.promos]},
                    indent=1, ensure_ascii=False), encoding="utf-8")
                msg = f"E{e.number}: {len(r.promos)} candidates"
            except (LLMError, ValueError, KeyError, TypeError) as exc:
                errors[e.number] = str(exc)[:400]
                msg = f"E{e.number}: FAILED - {str(exc)[:160]}"
                log.error(msg)
            progress("episodes", i, len(eps), msg)
    results.sort(key=lambda r: r.episode)

    promos: list[Promo] = [p for r in results for p in r.promos]

    # ---- pass 2: cross-episode montages
    if cross_episode:
        progress("cross_episode", 0, 1, "composing cross-episode montages")
        try:
            cross = generate_cross_episode(client, eps, results, ingest.show_title)
            promos += cross
            (dirs["candidates"] / "cross_episode.json").write_text(json.dumps(
                [promo_record(p) for p in cross], indent=1, ensure_ascii=False), encoding="utf-8")
            progress("cross_episode", 1, 1, f"{len(cross)} montage candidates")
        except LLMError as exc:
            errors[0] = f"cross-episode pass failed: {exc}"[:400]
            log.error(errors[0])
            progress("cross_episode", 1, 1, f"FAILED - {str(exc)[:160]}")

    # ---- deterministic validation + scoring + ranking
    for p in promos:
        p.issues += validate_promo(p, by_no)
    promos = rank_promos(promos)
    progress("rank", 1, 1, f"{sum(p.rank is not None for p in promos)} valid promos ranked")

    # ---- blind-viewer check (independent judge), then re-rank with both judges.
    # blind_top_n = -1 -> every non-rejected candidate (default); N -> only the current top N.
    pool_ = [p for p in promos if p.status != "rejected"]
    top = pool_ if blind_top_n < 0 else [p for p in promos if p.rank][:blind_top_n]
    if blind_top_n and top:
        with ThreadPoolExecutor(max_workers=max(1, config.LLM_MAX_WORKERS)) as pool:
            futs = {pool.submit(blind_test, client, p, by_no): p for p in top}
            for i, fut in enumerate(as_completed(futs), start=1):
                try:
                    fut.result()
                except LLMError as exc:
                    futs[fut].issues.append(Issue("warning", "blind_test_failed", str(exc)[:200]))
                progress("blind_test", i, len(top), futs[fut].promo_id)
        promos = rank_promos(promos)
        progress("rank", 1, 1, f"re-ranked with blind-viewer scores: {sum(p.rank is not None for p in promos)} ranked")

    report = automated_report(promos, eps, errors, client.calls)
    calls = client.calls
    cost = sum(float(u.get("total_cost_usd") or 0) for c in calls if not c.cached for u in c.usage)
    meta = {"show": ingest.show_title, "slug": slug, "input": ingest.input_path,
            "provider": provider.describe(),
            "models_reported": sorted({c.model for c in calls}),
            "prompt_version": config.PROMPT_VERSION, "weights": config.WEIGHTS, "penalties": config.PENALTIES,
            "blind_blend": {"weight": config.BLIND_BLEND, "map": config.BLIND_MAP,
                            "blind_tested": sum(p.blind_test is not None for p in promos)},
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "episodes_processed": [e.number for e in eps], "episode_errors": errors,
            "elapsed_s": round(time.time() - t0, 1), "counts": report["counts"],
            "llm_calls": len(calls), "llm_calls_from_cache": sum(c.cached for c in calls),
            "reported_cost_usd_this_run": round(cost, 4)}
    paths = write_all(promos, by_no, meta, dirs["final_promos"])
    (dirs["evaluation"] / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
    (dirs["evaluation"] / "quality_report.md").write_text(report_markdown(report, ingest.show_title,
                                                                          provider.describe()), encoding="utf-8")
    write_review_sheet(promos, dirs["evaluation"] / "manual_review_sheet.csv")
    (dirs["base"] / "run_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("done in %.1fs: %s", time.time() - t0, report["counts"])
    progress("done", 1, 1, "finished")
    return {"meta": meta, "promos": promos, "episodes": by_no, "ingest": ingest, "report": report,
            "paths": {k: str(v) for k, v in paths.items()}, "dirs": {k: str(v) for k, v in dirs.items()}}


# --------------------------------------------------------------------------- reload saved results
def promo_from_record(r: dict) -> Promo:
    segs = [Segment(s["episode"], s["start_s"], s["end_s"], s["blocks"][0], s["blocks"][1]) for s in r["segments"]]
    p = Promo(promo_id=r["promo_id"], source=r["type"], title=r.get("title", ""), segments=segs,
              hook_text=r["hook"], hook_ref=None, cliffhanger_text=r["cliffhanger"], cliffhanger_ref=None,
              open_question=r.get("open_question", ""), why_it_works=r.get("why_it_works", ""),
              context_gap=r.get("context_gap", ""), llm_scores=r.get("llm_scores", {}), tags=r.get("tags", []),
              issues=[Issue(**i) for i in r.get("issues", [])], penalties=r.get("penalties", {}),
              effective_scores=r.get("effective_scores", {}),
              generator_only_score=r.get("generator_only_score", 0.0),
              weighted_score=r.get("weighted_llm_score", 0), final_score=r.get("final_score", 0),
              status=r.get("status", ""), duplicate_of=r.get("duplicate_of"), rank=r.get("rank"),
              rank_reason=r.get("rank_reason", ""), hook_offset_s=r.get("hook_at_s"),
              cliff_end_gap_s=r.get("cliffhanger_gap_to_cut_s"), blind_test=r.get("blind_test"))
    return p


def load_saved(slug: str, output_root: Path = config.OUTPUT_ROOT) -> dict | None:
    base = output_root / slug
    f = base / "final_promos" / "promos.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    ep_file = base / "processed" / "episodes.json"
    eps: dict[int, Episode] = {}
    if ep_file.exists():
        from .models import Block, Scene
        for e in json.loads(ep_file.read_text(encoding="utf-8")):
            ep = Episode(e["number"], e["title"], e["logline"], e["continuity"], e["source_file"],
                         [Scene(**s) for s in e["scenes"]], [Block(**b) for b in e["blocks"]], e["warnings"])
            eps[ep.number] = ep
    rep_f = base / "evaluation" / "quality_report.json"
    ing_f = base / "processed" / "ingest_report.json"
    return {"meta": data["meta"], "promos": [promo_from_record(r) for r in data["promos"]], "episodes": eps,
            "report": json.loads(rep_f.read_text(encoding="utf-8")) if rep_f.exists() else {},
            "ingest": json.loads(ing_f.read_text(encoding="utf-8")) if ing_f.exists() else {},
            "dirs": {"base": str(base)}}


def list_saved(output_root: Path = config.OUTPUT_ROOT) -> list[str]:
    if not output_root.exists():
        return []
    return sorted(p.name for p in output_root.iterdir() if (p / "final_promos" / "promos.json").exists())
