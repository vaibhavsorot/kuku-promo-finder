"""LLM candidate generation.

Pass 1 (one call per episode, parallel, cached): the LLM reads the whole episode with block ids and
proposes promos + short standout "moments" + a label->character glossary.
Pass 2 (one call per show): the LLM sees only the compact moment index across all episodes
(~10k tokens, not the full 50 scripts) and composes cross-episode montages.

Each response is validated in code; problems are sent back to the model for self-correction
(see LLMClient.call_json). Prefix "SOFT:" marks rule violations the model should fix but which do not
make the response unusable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from . import prompts
from .llm_provider import LLMClient
from .models import Episode, Promo
from .promo_builder import build_segment, promo_from_llm
from .screenplay_parser import fmt_time
from .validation import validate_promo

SOFT_CODES = {"duration_out_of_range", "hook_not_in_first_3s", "cliff_not_at_end", "not_chronological",
              "overlapping_segments", "segment_too_short", "hook_quote_unsupported",
              "cliffhanger_quote_unsupported", "beyond_episode", "too_many_segments"}


@dataclass
class EpisodeResult:
    episode: int
    summary: str = ""
    character_notes: dict = field(default_factory=dict)
    promos: list[Promo] = field(default_factory=list)
    moments: list[dict] = field(default_factory=list)
    error: str | None = None


def _int(v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"expected an integer, got {v!r}")
    return int(v)


def _check_promo_items(items, eps, default_ep, label) -> list[str]:
    problems: list[str] = []
    if not isinstance(items, list):
        return [f"'{label}' must be a list"]
    for i, item in enumerate(items):
        where = f"promos[{i}]"
        if not isinstance(item, dict) or not item.get("segments"):
            problems.append(f"{where}: missing segments"); continue
        try:
            for s in item["segments"]:
                _int(s["start_block"]); _int(s["end_block"])
                if default_ep is None:
                    _int(s["episode"])
            for k in ("hook", "cliffhanger"):
                if not isinstance(item.get(k), dict) or "block" not in item[k] or "text" not in item[k]:
                    problems.append(f"{where}: '{k}' needs 'block' and 'text'")
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{where}: malformed segment/beat ({exc})"); continue
        if isinstance(item.get("cliffhanger"), dict) and not str(item["cliffhanger"].get("open_question", "")).strip():
            problems.append(f"SOFT: {where}: cliffhanger.open_question is empty")
        sc = item.get("scores")
        if not isinstance(sc, dict) or any(k not in sc for k in config.WEIGHTS):
            problems.append(f"{where}: 'scores' must contain {list(config.WEIGHTS)}")
        promo = promo_from_llm(item, "check", "check", eps, default_episode=default_ep)
        for issue in promo.issues + validate_promo(promo, eps):
            if issue.code in SOFT_CODES:
                problems.append(f"SOFT: {where} ({item.get('title', '')}): {issue.message}")
            elif issue.severity == "error" and issue.code not in ("incomplete_scores", "missing_hook",
                                                                   "missing_cliffhanger"):
                problems.append(f"{where}: {issue.message}")
    return problems


def make_episode_validator(ep: Episode):
    eps = {ep.number: ep}

    def validate(data) -> list[str]:
        if not isinstance(data, dict):
            return ["top level must be a JSON object"]
        problems = _check_promo_items(data.get("promos"), eps, ep.number, "promos")
        for j, m in enumerate(data.get("moments") or []):
            try:
                a, b = _int(m["start_block"]), _int(m["end_block"])
                if not (0 <= a <= b < len(ep.blocks)):
                    problems.append(f"moments[{j}]: block range [{a},{b}] invalid")
            except (KeyError, TypeError, ValueError):
                problems.append(f"moments[{j}]: needs integer start_block/end_block")
        return problems

    return validate


def generate_for_episode(client: LLMClient, ep: Episode, show_title: str) -> EpisodeResult:
    data = client.call_json(f"E{ep.number:03d}_generate", prompts.GENERATE_SYSTEM,
                            prompts.generate_user_prompt(ep, show_title), make_episode_validator(ep))
    eps = {ep.number: ep}
    res = EpisodeResult(ep.number, str(data.get("episode_summary", "")),
                        {str(k): str(v) for k, v in (data.get("character_notes") or {}).items()})
    for i, item in enumerate(data.get("promos") or [], start=1):
        res.promos.append(promo_from_llm(item, f"E{ep.number:02d}-P{i}", "episode", eps, default_episode=ep.number))
    for m in data.get("moments") or []:
        try:
            seg, _ = build_segment(ep, int(m["start_block"]), int(m["end_block"]))
        except (KeyError, TypeError, ValueError):
            continue
        if seg:
            res.moments.append({"episode": ep.number, "start_block": seg.first_block, "end_block": seg.last_block,
                                "start": seg.start, "end": seg.end, "duration": seg.duration,
                                "type": str(m.get("type", "")), "text": str(m.get("text", "")),
                                "intensity": m.get("intensity")})
    return res


def moment_index(episodes: list[Episode], results: list[EpisodeResult], limit: int = 220) -> str:
    moments = [m for r in results for m in r.moments if 2.0 <= m["duration"] <= 20.0]
    moments.sort(key=lambda m: -(float(m.get("intensity") or 0)))
    moments = sorted(moments[:limit], key=lambda m: (m["episode"], m["start"]))
    notes: dict[str, str] = {}
    for r in results:
        for k, v in r.character_notes.items():
            notes.setdefault(k, v)
    lines = ["EPISODE LOGLINES:"] + [f"E{e.number}: {e.logline}" for e in episodes]
    lines += ["", "SPEAKER LABEL GLOSSARY:"] + [f"{k} = {v}" for k, v in sorted(notes.items())]
    lines += ["", "MOMENT INDEX (episode [start_block-end_block] start-end (duration) type (intensity): text):"]
    for m in moments:
        lines.append(f"E{m['episode']} [{m['start_block']}-{m['end_block']}] {fmt_time(m['start'])}-"
                     f"{fmt_time(m['end'])} ({m['duration']:.1f}s) {m['type']} ({m.get('intensity')}): {m['text']}")
    return "\n".join(lines)


def make_cross_validator(eps: dict[int, Episode], allowed: set[tuple[int, int, int]]):
    def validate(data) -> list[str]:
        if not isinstance(data, dict):
            return ["top level must be a JSON object"]
        problems = _check_promo_items(data.get("promos"), eps, None, "promos")
        for i, item in enumerate(data.get("promos") or []):
            try:
                segs = item["segments"]
                if len({_int(s["episode"]) for s in segs}) < 2:
                    problems.append(f"SOFT: promos[{i}]: a montage must use at least two different episodes")
                for s in segs:
                    key = (_int(s["episode"]), _int(s["start_block"]), _int(s["end_block"]))
                    if key not in allowed:
                        problems.append(f"SOFT: promos[{i}]: segment {key} is not a range from the moment index")
            except (KeyError, TypeError, ValueError):
                pass  # already reported by _check_promo_items
        return problems

    return validate


def generate_cross_episode(client: LLMClient, episodes: list[Episode], results: list[EpisodeResult],
                           show_title: str) -> list[Promo]:
    ok = [r for r in results if not r.error and r.moments]
    if len({r.episode for r in ok}) < 2:
        return []
    eps = {e.number: e for e in episodes}
    index = moment_index(episodes, ok)
    allowed = {(m["episode"], m["start_block"], m["end_block"]) for r in ok for m in r.moments}
    user = f"SERIES: {show_title}\n\n{index}\n\nBuild the cross-episode montage promos. JSON only."
    data = client.call_json("CROSS_generate", prompts.CROSS_SYSTEM, user, make_cross_validator(eps, allowed))
    return [promo_from_llm(item, f"X-P{i}", "cross_episode", eps)
            for i, item in enumerate(data.get("promos") or [], start=1)]
