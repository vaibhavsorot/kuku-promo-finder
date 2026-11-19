"""Explainable scoring and ranking.

final = 10 * sum(weight_k * score_k)  -  deterministic penalties      (clamped to 0-100)

* score_k: 1-10 rubric scores from the generator LLM. For hook, standalone clarity and cliffhanger,
           when a blind-viewer test exists, score_k = 0.5*generator + 0.5*blind viewer
           (first-3s stop-scroll / understood / want-next) - two independent judges.
* penalties:   from validation warnings (code) - e.g. the LLM *claimed* a hook but the hook beat
               starts 7 s in, or a quote is not in the screenplay. The LLM cannot talk its way past these.
Ranking: valid promos by final score; a promo sharing >50% of its screen time with a better one is
marked as a duplicate so the top of the list is a diverse posting schedule, not five cuts of one scene.
"""
from __future__ import annotations

import config
from .models import Promo
from .validation import is_duplicate

ISSUE_TO_PENALTY = {
    "hook_not_in_first_3s": "hook_not_in_first_3s",
    "hook_quote_unsupported": "hook_quote_unsupported",
    "cliff_not_at_end": "cliff_not_at_end",
    "cliffhanger_quote_unsupported": "cliff_quote_unsupported",
    "no_content_in_first_3s": "no_content_in_first_3s",
}


def effective_scores(p: Promo) -> dict[str, float]:
    """Generator scores, with hook / standalone / cliffhanger averaged with the blind viewer's ratings."""
    eff = dict(p.llm_scores)
    bt = p.blind_test or {}
    for dim, blind_key in config.BLIND_MAP.items():
        try:
            b = float(bt[blind_key])
        except (KeyError, TypeError, ValueError):
            continue
        if dim in eff:
            eff[dim] = round((1 - config.BLIND_BLEND) * eff[dim] + config.BLIND_BLEND * b, 2)
    return eff


def score_promo(p: Promo) -> Promo:
    p.effective_scores = effective_scores(p)
    p.weighted_score = round(10 * sum(w * p.effective_scores.get(k, 0.0) for k, w in config.WEIGHTS.items()), 2)
    generator_weighted = 10 * sum(w * p.llm_scores.get(k, 0.0) for k, w in config.WEIGHTS.items())
    pen: dict[str, float] = {}
    codes = {i.code for i in p.issues}
    for code, key in ISSUE_TO_PENALTY.items():
        if code in codes:
            pen[key] = config.PENALTIES[key]
    if "boundary_snapped" in codes:
        pen["boundary_snapped"] = config.PENALTIES["boundary_snapped"]
    extra = max(0, len(p.segments) - 1)
    if extra:
        pen["extra_segments"] = extra * config.PENALTIES["per_extra_segment"]
    if len(p.episodes) > 1:
        pen["cross_episode"] = config.PENALTIES["cross_episode"]
    p.penalties = pen
    p.final_score = round(max(0.0, min(100.0, p.weighted_score - sum(pen.values()))), 2)
    p.generator_only_score = round(max(0.0, min(100.0, generator_weighted - sum(pen.values()))), 2)
    return p


def _reason(p: Promo, rank: int) -> str:
    s = p.effective_scores or p.llm_scores
    strengths = sorted(config.WEIGHTS, key=lambda k: -s.get(k, 0))[:2]
    labels = {
        "hook": "opening hook", "cliffhanger": "unresolved ending", "standalone_clarity": "clear to a new viewer",
        "emotional_intensity": "emotion", "curiosity": "curiosity", "coherence": "story flow",
        "premise_appeal": "sells the show's premise",
    }
    bits = [f"#{rank}: score {p.final_score:.1f}",
            "highest marks for " + " and ".join(f"{labels.get(k, k)} ({s.get(k, 0):.0f}/10)" for k in strengths)]
    if p.hook_offset_s is not None and p.hook_offset_s <= config.HOOK_WINDOW_S + config.HOOK_TOLERANCE_S:
        bits.append(f"the named hook starts {p.hook_offset_s:.1f}s in")
    if p.cliff_end_gap_s is not None and p.cliff_end_gap_s <= config.CLIFF_WINDOW_S:
        bits.append(f"the cut is {p.cliff_end_gap_s:.1f}s after the named cliffhanger")
    bits.append(f"{p.duration:.0f}s, {'1 continuous segment' if len(p.segments) == 1 else f'{len(p.segments)} segments'}")
    if p.blind_test:
        bt = p.blind_test
        bits.append(f"new-viewer check: understood {bt.get('understood_score')}/10, "
                    f"opening made them pause {bt.get('first_3s_stop_scroll')}/10, "
                    f"wanted the next episode {bt.get('want_next_episode')}/10")
    if p.penalties:
        penalty_names = {
            "hook_not_in_first_3s": "named hook starts too late", "hook_quote_unsupported": "hook quote not in script",
            "cliff_not_at_end": "cliffhanger is too far from the cut", "cliff_quote_unsupported": "cliffhanger quote not in script",
            "no_content_in_first_3s": "little content in opening 3 seconds", "boundary_snapped": "cut moved to avoid splitting a line",
            "per_extra_segment": "extra segments", "cross_episode": "joins episodes",
        }
        bits.append("score reductions: " + ", ".join(f"{penalty_names.get(k, k)} −{v:g}" for k, v in p.penalties.items()))
    return "; ".join(bits)


def rank_promos(promos: list[Promo]) -> list[Promo]:
    """Sets status/rank on every promo and returns them sorted (ranked first). Safe to call again
    after blind tests are added (it recomputes everything)."""
    for p in promos:
        p.rank, p.duplicate_of, p.rank_reason = None, None, ""
        score_promo(p)
        p.status = "rejected" if any(i.severity == "error" for i in p.issues) else "valid"
    valid = sorted((p for p in promos if p.status == "valid"),
                   key=lambda p: (-p.final_score, -p.llm_scores.get("hook", 0), p.duration))
    kept: list[Promo] = []
    for p in valid:
        dup = next((k for k in kept if is_duplicate(p, k)), None)
        if dup:
            p.status, p.duplicate_of = "duplicate", dup.promo_id
            continue
        kept.append(p)
    for r, p in enumerate(kept, start=1):
        p.rank = r
        p.rank_reason = _reason(p, r)
    rest = [p for p in promos if p.rank is None]
    rest.sort(key=lambda p: (p.status != "duplicate", -p.final_score))
    return kept + rest
