"""Deterministic validation. No LLM here: every rule is plain arithmetic on parsed timestamps.

Errors reject a promo. Warnings are kept for the editor and some carry score penalties
(see config.PENALTIES / scoring.py).
"""
from __future__ import annotations

import config
from .models import Episode, Issue, Promo
from .promo_builder import promo_timeline


def validate_promo(promo: Promo, eps: dict[int, Episode]) -> list[Issue]:
    out: list[Issue] = []
    add = lambda sev, code, msg: out.append(Issue(sev, code, msg))  # noqa: E731

    if not promo.segments:
        add("error", "no_segments", "promo has no usable segments")
        return out
    if len(promo.segments) > config.MAX_SEGMENTS:
        add("error", "too_many_segments", f"{len(promo.segments)} segments (max {config.MAX_SEGMENTS})")

    # --- timestamps & episode references
    for i, s in enumerate(promo.segments):
        ep = eps.get(s.episode)
        if ep is None:
            add("error", "bad_episode", f"segment {i + 1}: episode {s.episode} not loaded")
            continue
        if s.start < 0 or s.end <= s.start:
            add("error", "bad_timestamps", f"segment {i + 1}: start {s.start} / end {s.end} invalid")
        if s.end > ep.duration + 0.5:
            add("error", "beyond_episode", f"segment {i + 1}: ends at {s.end:.2f}s, episode is {ep.duration:.2f}s")
        if s.duration < config.MIN_SEGMENT_S:
            add("error", "segment_too_short", f"segment {i + 1}: {s.duration:.2f}s < {config.MIN_SEGMENT_S}s")

    # --- ordering / overlap (story order: episode asc, then time asc, no overlap)
    for i, (a, b) in enumerate(zip(promo.segments, promo.segments[1:]), start=1):
        if (b.episode, b.start) < (a.episode, a.start):
            add("error", "not_chronological", f"segment {i + 1} plays before segment {i} in story order")
        elif a.episode == b.episode and b.start < a.end - 0.05:
            add("error", "overlapping_segments", f"segments {i} and {i + 1} overlap by {a.end - b.start:.2f}s")

    # --- duration
    d = promo.duration
    if d < config.MIN_PROMO_S or d > config.MAX_PROMO_S:
        add("error", "duration_out_of_range",
            f"total {d:.1f}s outside {config.MIN_PROMO_S:.0f}-{config.MAX_PROMO_S:.0f}s")

    # --- required content
    if not promo.hook_text.strip():
        add("error", "missing_hook", "no hook given")
    if not promo.cliffhanger_text.strip():
        add("error", "missing_cliffhanger", "no cliffhanger given")
    if not promo.why_it_works.strip():
        add("warning", "missing_explanation", "no 'why it works' line")
    missing = [k for k in config.WEIGHTS if k not in promo.llm_scores]
    if missing:
        add("error", "incomplete_scores", f"LLM scores missing: {missing}")

    # --- hook really in the first 3 s?  (checks the LLM's claim against the timeline)
    limit = config.HOOK_WINDOW_S + config.HOOK_TOLERANCE_S
    if promo.hook_offset_s is None:
        add("warning", "hook_not_in_first_3s", "hook beat could not be located inside the promo")
    elif promo.hook_offset_s > limit:
        add("warning", "hook_not_in_first_3s",
            f"hook beat starts at {promo.hook_offset_s:.1f}s into the promo (must be <= {config.HOOK_WINDOW_S:.0f}s)")
    tl = promo_timeline(promo, eps)
    if not any(b.kind in ("dialogue", "action", "sfx") and ps <= config.HOOK_WINDOW_S for ps, _pe, _s, b in tl):
        add("warning", "no_content_in_first_3s", "first 3 s contain only camera directions (no line/action/sound)")

    # --- cliffhanger really at the cut?
    if promo.cliff_end_gap_s is None:
        add("warning", "cliff_not_at_end", "cliffhanger beat could not be located inside the promo")
    elif promo.cliff_end_gap_s > config.CLIFF_WINDOW_S:
        add("warning", "cliff_not_at_end",
            f"cliffhanger beat ends {promo.cliff_end_gap_s:.1f}s before the cut (max {config.CLIFF_WINDOW_S:.0f}s)")
    return out


def segment_overlap_s(a: Promo, b: Promo) -> float:
    total = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                total += max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
    return total


def is_duplicate(a: Promo, b: Promo) -> bool:
    shorter = min(a.duration, b.duration) or 1.0
    return segment_overlap_s(a, b) / shorter > config.DUPLICATE_OVERLAP
