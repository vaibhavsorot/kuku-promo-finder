"""JSON / CSV / Markdown export of ranked promos."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import config
from .models import Episode, Promo
from .screenplay_parser import fmt_time


def segments_str(p: Promo) -> str:
    return " + ".join(f"E{s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}" for s in p.segments)


def promo_record(p: Promo) -> dict:
    """Flat, human-facing record (the assignment's required fields first)."""
    return {
        "rank": p.rank,
        "promo_id": p.promo_id,
        "status": p.status,
        "duplicate_of": p.duplicate_of,
        "type": p.source,
        "episodes": p.episodes,
        "start": fmt_time(p.segments[0].start) if p.segments else None,
        "end": fmt_time(p.segments[-1].end) if p.segments else None,
        "duration_s": p.duration,
        "segments": [{"order": i + 1, "episode": s.episode, "start": fmt_time(s.start), "end": fmt_time(s.end),
                      "start_s": s.start, "end_s": s.end, "duration_s": s.duration,
                      "blocks": [s.first_block, s.last_block]} for i, s in enumerate(p.segments)],
        "hook": p.hook_text,
        "hook_at_s": p.hook_offset_s,
        "cliffhanger": p.cliffhanger_text,
        "open_question": p.open_question,
        "cliffhanger_gap_to_cut_s": p.cliff_end_gap_s,
        "why_it_works": p.why_it_works,
        "final_score": p.final_score,
        "weighted_llm_score": p.weighted_score,
        "generator_only_score": p.generator_only_score,
        "llm_scores": p.llm_scores,
        "effective_scores": p.effective_scores,
        "penalties": p.penalties,
        "rank_reason": p.rank_reason,
        "context_gap": p.context_gap,
        "title": p.title,
        "tags": p.tags,
        "issues": [{"severity": i.severity, "code": i.code, "message": i.message} for i in p.issues],
        "blind_test": p.blind_test,
    }


CSV_COLUMNS = ["rank", "promo_id", "status", "type", "episodes", "segments", "start", "end", "duration_s",
               "final_score", "generator_only_score", "blind_understood", "blind_stop_scroll", "blind_want_next",
               "hook", "cliffhanger", "open_question", "why_it_works", "rank_reason",
               *[f"score_{k}" for k in config.WEIGHTS], "penalties", "errors", "warnings", "duplicate_of"]


def to_csv(promos: list[Promo]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for p in promos:
        r = promo_record(p)
        w.writerow({
            "rank": p.rank or "", "promo_id": p.promo_id, "status": p.status, "type": p.source,
            "episodes": " ".join(f"E{e}" for e in p.episodes), "segments": segments_str(p),
            "start": r["start"], "end": r["end"], "duration_s": p.duration, "final_score": p.final_score,
            "generator_only_score": p.generator_only_score,
            "blind_understood": (p.blind_test or {}).get("understood_score", ""),
            "blind_stop_scroll": (p.blind_test or {}).get("first_3s_stop_scroll", ""),
            "blind_want_next": (p.blind_test or {}).get("want_next_episode", ""),
            "hook": p.hook_text, "cliffhanger": p.cliffhanger_text, "open_question": p.open_question,
            "why_it_works": p.why_it_works, "rank_reason": p.rank_reason,
            **{f"score_{k}": p.llm_scores.get(k, "") for k in config.WEIGHTS},
            "penalties": "; ".join(f"{k} -{v:g}" for k, v in p.penalties.items()),
            "errors": " | ".join(i.message for i in p.issues if i.severity == "error"),
            "warnings": " | ".join(i.code for i in p.issues if i.severity == "warning"),
            "duplicate_of": p.duplicate_of or "",
        })
    return buf.getvalue()


def to_json(promos: list[Promo], meta: dict) -> str:
    return json.dumps({"meta": meta, "promos": [promo_record(p) for p in promos]}, indent=2, ensure_ascii=False)


def to_markdown(promos: list[Promo], eps: dict[int, Episode], show: str, top_n: int = 15) -> str:
    from .promo_builder import transcript
    L = [f"# Ranked promos — {show}", ""]
    for p in [p for p in promos if p.rank][:top_n]:
        L += [f"## #{p.rank} · {p.promo_id} · {p.final_score:.1f}/100 · {p.duration:.0f}s — {p.title}",
              f"- **Clip locations:** {segments_str(p)}",
              f"- **Opening hook ({p.hook_offset_s}s into the clip):** {p.hook_text}",
              f"- **Ending moment:** {p.cliffhanger_text}",
              f"- **Question it leaves:** {p.open_question}",
              f"- **Why this clip may work:** {p.why_it_works}",
              f"- **Why it is ranked here:** {p.rank_reason}"]
        if p.context_gap:
            L.append(f"- **What a new viewer may not know (AI estimate):** {p.context_gap}")
        L += ["", "```", transcript(p, eps, with_camera=False), "```", ""]
    return "\n".join(L)


def write_all(promos: list[Promo], eps: dict[int, Episode], meta: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "promos.json", "csv": out_dir / "promos.csv", "md": out_dir / "ranked_promos.md"}
    paths["json"].write_text(to_json(promos, meta), encoding="utf-8")
    paths["csv"].write_text(to_csv(promos), encoding="utf-8-sig")  # BOM so Excel reads UTF-8
    paths["md"].write_text(to_markdown(promos, eps, meta.get("show", "")), encoding="utf-8")
    return paths
