"""Deterministic construction of promos from LLM block-id selections.

The LLM never supplies timestamps. It names block ids; this module converts them to real
start/end times from the parsed screenplay, snaps cut points so no dialogue line is cut in half,
and locates the hook / cliffhanger beats in promo time so validation can check the LLM's claims.
"""
from __future__ import annotations

import difflib
import re

from .models import Block, Episode, Issue, Promo, Segment


# --------------------------------------------------------------------------- segments
def build_segment(ep: Episode, start_idx: int, end_idx: int) -> tuple[Segment | None, list[Issue]]:
    issues: list[Issue] = []
    n = len(ep.blocks)
    if not (0 <= start_idx < n and 0 <= end_idx < n):
        return None, [Issue("error", "bad_block_id", f"E{ep.number}: block range [{start_idx},{end_idx}] "
                                                     f"outside 0..{n - 1}")]
    if end_idx < start_idx:
        return None, [Issue("error", "reversed_range", f"E{ep.number}: end_block {end_idx} < start_block {start_idx}")]
    rng = [b for b in ep.blocks[start_idx:end_idx + 1] if b.timed]
    if not rng:
        return None, [Issue("error", "untimed_segment", f"E{ep.number}: no timestamped blocks in [{start_idx},{end_idx}]")]
    s = min(b.start for b in rng)
    e = max(b.end for b in rng)

    dialogue = [b for b in ep.blocks if b.kind == "dialogue" and b.timed]
    for _ in range(3):  # iterate until cut points are stable
        changed = False
        for d in dialogue:
            if d.start < s < d.end:          # cut-in lands mid-line -> include the whole line
                issues.append(Issue("warning", "boundary_snapped",
                                    f"E{ep.number}: start moved {s:.2f}->{d.start:.2f}s to include line [{d.idx}]"))
                s, changed = d.start, True
            if d.start < e < d.end:          # cut-out lands mid-line
                heard = (e - d.start) / max(d.end - d.start, 1e-6)
                new_e = d.end if heard >= 0.5 or start_idx <= d.idx <= end_idx else d.start
                if new_e <= s:
                    new_e = d.end
                issues.append(Issue("warning", "boundary_snapped",
                                    f"E{ep.number}: end moved {e:.2f}->{new_e:.2f}s so line [{d.idx}] is not cut"))
                e, changed = new_e, True
        if not changed:
            break
    return Segment(ep.number, round(s, 3), round(e, 3), start_idx, end_idx), issues


def segment_blocks(seg: Segment, ep: Episode) -> list[Block]:
    """Blocks a viewer actually sees/hears in the segment (time-based, >=50% inside)."""
    out = []
    for b in ep.blocks:
        if not b.timed:
            continue
        overlap = min(b.end, seg.end) - max(b.start, seg.start)
        dur = max(b.end - b.start, 1e-6)
        if overlap > 0 and (overlap / dur >= 0.5 or b.start >= seg.start and b.start < seg.end):
            out.append(b)
    return out


def promo_timeline(promo: Promo, eps: dict[int, Episode]) -> list[tuple[float, float, Segment, Block]]:
    """(promo_start, promo_end, segment, block) for every block in playback order."""
    out, offset = [], 0.0
    for seg in promo.segments:
        ep = eps[seg.episode]
        for b in segment_blocks(seg, ep):
            ps = offset + max(b.start, seg.start) - seg.start
            pe = offset + min(b.end, seg.end) - seg.start
            out.append((round(ps, 2), round(pe, 2), seg, b))
        offset += seg.duration
    return out


def transcript(promo: Promo, eps: dict[int, Episode], with_camera: bool = True) -> str:
    lines = []
    last_seg = None
    for ps, pe, seg, b in promo_timeline(promo, eps):
        if seg is not last_seg and len(promo.segments) > 1:
            lines.append("[CUT]")
        last_seg = seg
        if b.kind == "camera" and not with_camera:
            continue
        lines.append(f"{ps:5.1f}-{pe:5.1f}s {b.label}: {b.text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- quote checks
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def quote_matches(quote: str, block_text: str) -> bool:
    q, t = _norm(quote), _norm(block_text)
    if not q or not t:
        return False
    if q in t or (len(t) >= 6 and t in q):
        return True
    return difflib.SequenceMatcher(None, q, t).ratio() >= 0.75


def locate_beat(promo: Promo, eps: dict[int, Episode], ref: tuple[int, int] | None, quote: str,
                which: str) -> tuple[tuple[int, int] | None, float | None, float | None, list[Issue]]:
    """Find the hook/cliffhanger block inside the promo.

    Returns (ref, promo_start, promo_end, issues). If the named block does not contain the quote
    but another block in the promo does, the reference is corrected (warning). If the quote is
    found nowhere in the promo, it is flagged as unsupported."""
    issues: list[Issue] = []
    tl = promo_timeline(promo, eps)
    named = [(ps, pe, b) for ps, pe, seg, b in tl if ref and (seg.episode, b.idx) == tuple(ref)]
    if named and quote_matches(quote, named[0][2].text):
        ps, pe, b = named[0]
        return ref, ps, pe, issues
    # search the whole promo for the quoted text (prefer earliest for hook, latest for cliffhanger)
    hits = [(ps, pe, seg, b) for ps, pe, seg, b in tl if quote_matches(quote, b.text)]
    if hits:
        ps, pe, seg, b = hits[0] if which == "hook" else hits[-1]
        issues.append(Issue("warning", f"{which}_ref_corrected",
                            f"{which} quote found in block E{seg.episode}[{b.idx}], not the named block {ref}"))
        return (seg.episode, b.idx), ps, pe, issues
    issues.append(Issue("warning", f"{which}_quote_unsupported",
                        f"{which} text {quote[:60]!r} not found verbatim in the promo's screenplay blocks"))
    if named:  # fall back to the named block's timing
        ps, pe, _b = named[0]
        return ref, ps, pe, issues
    issues.append(Issue("warning", f"{which}_outside_promo", f"{which} block {ref} is not inside the promo"))
    return ref, None, None, issues


# --------------------------------------------------------------------------- from LLM JSON
def _scores(raw: dict) -> dict[str, float]:
    out = {}
    for k, v in (raw or {}).items():
        try:
            out[k] = max(1.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def promo_from_llm(item: dict, promo_id: str, source: str, eps: dict[int, Episode],
                   default_episode: int | None = None) -> Promo:
    segments: list[Segment] = []
    issues: list[Issue] = []
    for s in item.get("segments", []):
        ep_no = int(s.get("episode", default_episode or 0))
        if ep_no not in eps:
            issues.append(Issue("error", "bad_episode", f"segment references unknown episode {ep_no}"))
            continue
        seg, seg_issues = build_segment(eps[ep_no], int(s["start_block"]), int(s["end_block"]))
        issues += seg_issues
        if seg:
            segments.append(seg)
    hook, cliff = item.get("hook") or {}, item.get("cliffhanger") or {}

    def ref(d):
        if d.get("block") is None:
            return None
        return (int(d.get("episode", default_episode or 0)), int(d["block"]))

    promo = Promo(
        promo_id=promo_id, source=source, title=str(item.get("title", ""))[:120], segments=segments,
        hook_text=str(hook.get("text", "")), hook_ref=ref(hook),
        cliffhanger_text=str(cliff.get("text", "")), cliffhanger_ref=ref(cliff),
        open_question=str(cliff.get("open_question", "")), why_it_works=str(item.get("why_it_works", "")),
        context_gap=str(item.get("context_gap", "")), llm_scores=_scores(item.get("scores", {})),
        tags=[str(t) for t in item.get("tags", [])][:8], issues=issues,
    )
    if segments:
        promo.hook_ref, promo.hook_offset_s, _, hi = locate_beat(promo, eps, promo.hook_ref, promo.hook_text, "hook")
        promo.cliffhanger_ref, _, cliff_end, ci = locate_beat(promo, eps, promo.cliffhanger_ref,
                                                              promo.cliffhanger_text, "cliffhanger")
        promo.issues += hi + ci
        if cliff_end is not None:
            promo.cliff_end_gap_s = round(promo.duration - cliff_end, 2)
    return promo
