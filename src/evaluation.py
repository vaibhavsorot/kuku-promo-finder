"""Quality evaluation.

1. Automated, deterministic checks over the whole run (rejection reasons, verified hook/cliffhanger
   placement, quote support, score spread to spot inflation, episode coverage, LLM self-corrections).
2. Blind-viewer test (LLM, optional): a *separate* call sees only the promo transcript - no show
   context, no generator explanation - and reports what it understood. This is an independent check of
   the generator's own "standalone_clarity" claim. It is still a model, not an audience: treat it as a
   cheap smoke test, not ground truth.
3. A manual-review sheet for human editors (filled in by people; never auto-filled).

There are no ground-truth labels, so no precision/recall is reported.
"""
from __future__ import annotations

import csv
import statistics as st
from collections import Counter
from pathlib import Path

import config
from . import prompts
from .llm_provider import LLMClient
from .models import Episode, Promo
from .promo_builder import transcript


def _validate_blind(d) -> list[str]:
    if not isinstance(d, dict):
        return ["top level must be an object"]
    probs = []
    for k in ("understood_score", "first_3s_stop_scroll", "want_next_episode"):
        try:
            v = float(d.get(k))
            if not 1 <= v <= 10:
                probs.append(f"{k} must be 1-10")
        except (TypeError, ValueError):
            probs.append(f"{k} must be a number 1-10")
    return probs


def blind_test(client: LLMClient, promo: Promo, eps: dict[int, Episode]) -> dict:
    t = transcript(promo, eps)
    d = client.call_json(f"{promo.promo_id}_blind", prompts.BLIND_SYSTEM, prompts.blind_user_prompt(t),
                         _validate_blind)
    promo.blind_test = d
    return d


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {"n": len(vals), "mean": round(st.mean(vals), 2), "stdev": round(st.pstdev(vals), 2),
            "min": round(min(vals), 2), "max": round(max(vals), 2), "median": round(st.median(vals), 2)}


def automated_report(promos: list[Promo], episodes: list[Episode], episode_errors: dict[int, str],
                     llm_calls: list) -> dict:
    valid = [p for p in promos if p.status == "valid"]
    rejected = [p for p in promos if p.status == "rejected"]
    dups = [p for p in promos if p.status == "duplicate"]
    rej_reasons = Counter(i.code for p in rejected for i in p.issues if i.severity == "error")
    warn_counts = Counter(i.code for p in promos for i in p.issues if i.severity == "warning")
    tol = config.HOOK_WINDOW_S + config.HOOK_TOLERANCE_S
    all_built = [p for p in promos if p.segments]
    hook_ok = [p for p in all_built if p.hook_offset_s is not None and p.hook_offset_s <= tol]
    cliff_ok = [p for p in all_built if p.cliff_end_gap_s is not None and p.cliff_end_gap_s <= config.CLIFF_WINDOW_S]
    unsupported = [p for p in all_built if any(i.code.endswith("quote_unsupported") for i in p.issues)]
    covered = {e for p in valid + dups for e in p.episodes}
    dims = {k: _stats([p.llm_scores.get(k, 0) for p in all_built if k in p.llm_scores]) for k in config.WEIGHTS}
    blind = [p for p in valid if p.blind_test]
    agreement = []
    for p in blind:
        gen = p.llm_scores.get("standalone_clarity", 0)
        bl = float(p.blind_test.get("understood_score", 0))
        agreement.append({"promo_id": p.promo_id, "rank": p.rank, "generator_standalone": gen,
                          "blind_understood": bl, "gap": round(gen - bl, 1),
                          "blind_first_3s": p.blind_test.get("first_3s_stop_scroll"),
                          "blind_want_next": p.blind_test.get("want_next_episode"),
                          "blind_confusions": p.blind_test.get("confusing_points", [])})
    agreement.sort(key=lambda a: a["rank"] or 10**6)
    # how much did the independent judge change the ranking?
    ranked = [p for p in valid if p.rank]
    gen_order = sorted(ranked, key=lambda p: -p.generator_only_score)
    gen_rank = {p.promo_id: i + 1 for i, p in enumerate(gen_order)}
    n = len(ranked)
    spearman = None
    if n > 2:
        d2 = sum((p.rank - gen_rank[p.promo_id]) ** 2 for p in ranked)
        spearman = round(1 - 6 * d2 / (n * (n * n - 1)), 3)
    movers = sorted(ranked, key=lambda p: -abs(p.rank - gen_rank[p.promo_id]))[:8]
    rank_shift = {"spearman_generator_vs_final": spearman,
                  "top10_overlap": len({p.promo_id for p in ranked[:10]} & {p.promo_id for p in gen_order[:10]}),
                  "biggest_moves": [{"promo_id": p.promo_id, "generator_rank": gen_rank[p.promo_id],
                                     "final_rank": p.rank} for p in movers]}
    retries = [c for c in llm_calls if not c.cached and c.attempts > 1]
    return {
        "counts": {"episodes_loaded": len(episodes), "episodes_failed_llm": len(episode_errors),
                   "candidates": len(promos), "valid_ranked": len(valid), "rejected": len(rejected),
                   "duplicates": len(dups),
                   "episode_candidates": sum(p.source == "episode" for p in promos),
                   "cross_episode_candidates": sum(p.source == "cross_episode" for p in promos)},
        "episode_errors": episode_errors,
        "rejection_reasons": dict(rej_reasons.most_common()),
        "warnings": dict(warn_counts.most_common()),
        "claim_checks": {
            "hook_verified_in_first_3s": f"{len(hook_ok)}/{len(all_built)}",
            "cliffhanger_verified_at_cut": f"{len(cliff_ok)}/{len(all_built)}",
            "promos_with_unsupported_quote": f"{len(unsupported)}/{len(all_built)}",
        },
        "episode_coverage": {"episodes_with_valid_promo": len(covered),
                             "episodes_without": sorted({e.number for e in episodes} - covered)},
        "duration_s": _stats([p.duration for p in valid]),
        "final_score": _stats([p.final_score for p in valid]),
        "weighted_llm_score_all": _stats([p.weighted_score for p in all_built]),
        "llm_score_by_dimension": dims,
        "llm_calls": {"total": len(llm_calls), "served_from_cache": sum(c.cached for c in llm_calls),
                      "needed_self_correction": len(retries),
                      "self_corrected_calls": [c.name for c in retries]},
        "rank_shift": rank_shift,
        "blind_test": {"n": len(agreement),
                       "mean_blind_first_3s": round(st.mean([float(a["blind_first_3s"]) for a in agreement]), 2) if agreement else None,
                       "mean_generator_hook": round(st.mean([p.llm_scores.get("hook", 0) for p in blind]), 2) if blind else None,
                       "generator_minus_blind_standalone": _stats([a["gap"] for a in agreement]),
                       "mean_generator_standalone": round(st.mean([a["generator_standalone"] for a in agreement]), 2) if agreement else None,
                       "mean_blind_understood": round(st.mean([a["blind_understood"] for a in agreement]), 2) if agreement else None,
                       "rows": agreement},
    }


def report_markdown(rep: dict, show: str, provider: str) -> str:
    c = rep["counts"]
    checks = rep["claim_checks"]
    coverage = rep["episode_coverage"]
    labels = {
        "episodes_loaded": "Episodes read",
        "episodes_failed_llm": "Episodes the AI could not process",
        "candidates": "Promo ideas found",
        "valid_ranked": "Promos ranked",
        "rejected": "Promos rejected by checks",
        "duplicates": "Promos marked as too similar",
        "episode_candidates": "Ideas from one episode",
        "cross_episode_candidates": "Ideas joining episodes",
    }
    L = [f"# Quality check: {show}", "",
         f"AI provider and model: `{provider}`.",
         "This report checks facts such as timestamps and quoted lines. AI scores are estimates, not real audience results.", "",
         "## Run summary"]
    L += [f"- {labels.get(k, k.replace('_', ' ').capitalize())}: **{v}**" for k, v in c.items()]
    L += [f"- Episodes with at least one usable promo: **{coverage['episodes_with_valid_promo']}**",
          f"- Episodes without a usable promo: **{', '.join(map(str, coverage['episodes_without'])) if coverage['episodes_without'] else 'none'}**", ""]

    L += ["## Checks against the scripts",
          f"- The AI's named hook starts within the first 3 seconds: **{checks['hook_verified_in_first_3s']}**",
          f"- The AI's named cliffhanger ends within 8 seconds of the cut: **{checks['cliffhanger_verified_at_cut']}**",
          f"- Promos with a quote missing from the script: **{checks['promos_with_unsupported_quote']}**",
          f"- Promos rejected: **{sum(rep['rejection_reasons'].values())}**"]
    if rep["rejection_reasons"]:
        L += ["", "Reasons for rejection:"] + [f"- {k.replace('_', ' ')}: {v}" for k, v in rep["rejection_reasons"].items()]
    L += ["", "## Things an editor should review"]
    warning_labels = {
        "cliffhanger_ref_corrected": "The cliffhanger reference needed correcting",
        "boundary_snapped": "A cut was adjusted to avoid splitting a line",
        "hook_ref_corrected": "The hook reference needed correcting",
        "no_content_in_first_3s": "The opening 3 seconds may have little spoken or visible action",
    }
    if rep["warnings"]:
        L += [f"- {warning_labels.get(k, k.replace('_', ' '))}: {v}" for k, v in rep["warnings"].items()]
    else:
        L.append("- No warnings were recorded.")

    score = rep["final_score"]
    duration = rep["duration_s"]
    if score and duration:
        L += ["", "## Scores and clip length",
              f"- Ranked promo scores: average **{score['mean']}**, from **{score['min']}** to **{score['max']}** out of 100.",
              f"- Clip length: average **{duration['mean']} seconds**; median **{duration['median']} seconds**.", "",
              "Average AI scores by category (1–10):", "",
              "| Category | Average | Range |", "|---|---:|---:|"]
        score_labels = {
            "hook": "Opening hook", "cliffhanger": "Unresolved ending", "standalone_clarity": "Easy for a new viewer to follow",
            "emotional_intensity": "Emotion", "curiosity": "Makes you curious", "coherence": "Flows as a short story",
            "premise_appeal": "Sells the show's premise",
        }
        for k, s in rep["llm_score_by_dimension"].items():
            if s:
                L.append(f"| {score_labels.get(k, k)} | {s['mean']} | {s['min']}–{s['max']} |")

    lc = rep["llm_calls"]
    L += ["", "## AI calls",
          f"- Calls made or read from cache: **{lc['total']}** ({lc['served_from_cache']} from cache).",
          f"- Calls that needed a correction: **{lc['needed_self_correction']}**."]
    b = rep["blind_test"]
    untested = c["valid_ranked"] - sum(1 for r in b["rows"] if r["rank"])
    if b["n"]:
        L += ["", "## New-viewer check",
              "A separate AI saw only each promo's transcript and guessed what a first-time viewer would understand. This is a useful warning signal, not a real viewer study.",
              f"- Promos checked: **{b['n']}**.",
              f"- Average understanding score: the promo-finder AI gave **{b['mean_generator_standalone']}**, while the new-viewer check gave **{b['mean_blind_understood']}**.",
              f"- Average opening-hook score: promo-finder AI **{b.get('mean_generator_hook', 'not recorded')}**; "
              f"new-viewer check **{b.get('mean_blind_first_3s', 'not recorded')}**."]
        if untested > 0:
            L.append(f"- **{untested}** ranked promos have no new-viewer result, so their score uses only the promo-finder AI.")
        rs = rep.get("rank_shift", {})
        if rs:
            L += [f"- The two rankings had **{rs['top10_overlap']} of the same 10 promos** in their top tens.",
                  "- Biggest rank changes: " + ", ".join(f"{m['promo_id']} moved from #{m['generator_rank']} to #{m['final_rank']}"
                                                           for m in rs["biggest_moves"][:5])]
        L += ["", "Top 10 and what confused the new-viewer check:", "",
              "| Rank | Promo | Understood (1–10) | Stop and watch (1–10) | Want next (1–10) | What was unclear |",
              "|---:|---|---:|---:|---:|---|"]
        for a in b["rows"][:10]:
            conf = "; ".join(a["blind_confusions"]) if a["blind_confusions"] else "Nothing listed"
            if len(conf) > 120:
                conf = conf[:117].rsplit(" ", 1)[0] + "…"
            L.append(f"| {a['rank']} | {a['promo_id']} | {a['blind_understood']:.0f} | {a['blind_first_3s']} | {a['blind_want_next']} | {conf} |")
    if rep["episode_errors"]:
        L += ["", "## Episodes the AI could not process"]
        L += [f"- Episode {k}: {v}" for k, v in rep["episode_errors"].items()]
    return "\n".join(L) + "\n"


REVIEW_COLUMNS = ["rank", "promo_id", "episodes", "segments", "duration_s", "final_score", "hook", "cliffhanger",
                  "REVIEW_hook_stops_scroll(y/n)", "REVIEW_ends_unresolved(y/n)", "REVIEW_understandable_cold(y/n)",
                  "REVIEW_timestamps_accurate(y/n)", "REVIEW_would_post(y/n)", "REVIEW_notes"]


def write_review_sheet(promos: list[Promo], path: Path, top_n: int = 20):
    from .exporter import segments_str
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(REVIEW_COLUMNS)
        for p in [p for p in promos if p.rank][:top_n]:
            w.writerow([p.rank, p.promo_id, " ".join(f"E{e}" for e in p.episodes), segments_str(p), p.duration,
                        p.final_score, p.hook_text, p.cliffhanger_text, "", "", "", "", "", ""])
