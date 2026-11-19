import config
from src.promo_builder import build_segment, promo_from_llm, quote_matches
from src.validation import validate_promo

SCORES = {k: 6 for k in config.WEIGHTS}


def item(segs, hook=(1, '"You slapped me!"'), cliff=(5, '"Then who is she?"'), **kw):
    return {"title": "t", "segments": [{"start_block": a, "end_block": b} for a, b in segs],
            "hook": {"block": hook[0], "text": hook[1]},
            "cliffhanger": {"block": cliff[0], "text": cliff[1], "open_question": "who?"},
            "why_it_works": "w", "scores": dict(SCORES), **kw}


def codes(issues):
    return {i.code for i in issues}


def test_segment_times_come_from_blocks(ep):
    seg, issues = build_segment(ep, 1, 5)
    assert (seg.start, seg.end) == (4.0, 24.0) and not issues


def test_snap_includes_whole_line_at_cut_in(ep):
    # camera block 4 runs 14-20; build a range that starts mid-way through a line
    ep.blocks[4].start = 11.0  # now block 4 starts inside MARK's line (9-14)
    seg, issues = build_segment(ep, 4, 5)
    assert seg.start == 9.0 and "boundary_snapped" in codes(issues)


def test_bad_ranges(ep):
    assert build_segment(ep, 5, 2)[0] is None
    assert build_segment(ep, 0, 99)[0] is None


def test_valid_promo_passes(eps):
    p = promo_from_llm(item([(1, 5)]), "x", "episode", eps, default_episode=7)
    iss = validate_promo(p, eps)
    assert not [i for i in iss if i.severity == "error"], iss
    assert p.duration == 20.0 and p.hook_offset_s == 0.0 and p.cliff_end_gap_s == 0.0


def test_too_short_is_rejected(eps):
    p = promo_from_llm(item([(1, 3)], cliff=(3, '"You were never going to be my wife."')), "x", "episode", eps, 7)
    assert "duration_out_of_range" in codes(validate_promo(p, eps))


def test_late_hook_and_early_cliff_flagged(eps):
    p = promo_from_llm(item([(0, 7)], hook=(3, '"You were never going to be my wife."'),
                            cliff=(5, '"Then who is she?"')), "x", "episode", eps, 7)
    c = codes(validate_promo(p, eps))
    assert "hook_not_in_first_3s" in c and "cliff_not_at_end" in c


def test_fabricated_quote_flagged(eps):
    p = promo_from_llm(item([(1, 5)], hook=(1, '"I will destroy your whole family"')), "x", "episode", eps, 7)
    assert "hook_quote_unsupported" in codes(p.issues)


def test_wrong_block_id_corrected_by_quote(eps):
    p = promo_from_llm(item([(1, 5)], hook=(2, '"You slapped me!"')), "x", "episode", eps, 7)
    assert p.hook_ref == (7, 1) and "hook_ref_corrected" in codes(p.issues)


def test_order_and_overlap(eps):
    p = promo_from_llm(item([(3, 5), (1, 2)]), "x", "episode", eps, 7)
    assert "not_chronological" in codes(validate_promo(p, eps))
    p2 = promo_from_llm(item([(1, 4), (3, 6)]), "x", "episode", eps, 7)
    assert "overlapping_segments" in codes(validate_promo(p2, eps))


def test_unknown_episode_and_missing_scores(eps):
    it = item([(1, 5)])
    it["segments"][0]["episode"] = 99
    p = promo_from_llm(it, "x", "episode", eps)
    assert "bad_episode" in codes(p.issues) or "no_segments" in codes(validate_promo(p, eps))
    it2 = item([(1, 5)]); it2["scores"] = {"hook": 5}
    p2 = promo_from_llm(it2, "x", "episode", eps, 7)
    assert "incomplete_scores" in codes(validate_promo(p2, eps))


def test_quote_matching():
    assert quote_matches("you slapped me", '"You slapped me!"')
    assert quote_matches('"Then who is she?"', '"Then who is she?"')
    assert not quote_matches("completely different words here", '"You slapped me!"')
