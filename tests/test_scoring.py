import json

import pytest

import config
from src.llm_provider import LLMClient, LLMError, Provider, LLMResult, extract_json
from src.promo_builder import promo_from_llm
from src.scoring import rank_promos, score_promo
from src.validation import validate_promo


def mk(eps, segs, scores, pid, hook=(1, '"You slapped me!"'), cliff=(5, '"Then who is she?"')):
    it = {"title": pid, "segments": [{"start_block": a, "end_block": b} for a, b in segs],
          "hook": {"block": hook[0], "text": hook[1]},
          "cliffhanger": {"block": cliff[0], "text": cliff[1], "open_question": "?"},
          "why_it_works": "w", "scores": scores}
    p = promo_from_llm(it, pid, "episode", eps, default_episode=7)
    p.issues += validate_promo(p, eps)
    return p


def test_weights_sum_to_one():
    assert abs(sum(config.WEIGHTS.values()) - 1) < 1e-9


def test_weighted_score_formula(eps):
    p = score_promo(mk(eps, [(1, 5)], {k: 8 for k in config.WEIGHTS}, "a"))
    assert p.weighted_score == 80.0 and p.final_score == 80.0 and p.penalties == {}


def test_penalties_apply(eps):
    p = score_promo(mk(eps, [(0, 7)], {k: 8 for k in config.WEIGHTS}, "b",
                       hook=(3, '"You were never going to be my wife."'), cliff=(5, '"Then who is she?"')))
    assert p.penalties["hook_not_in_first_3s"] == config.PENALTIES["hook_not_in_first_3s"]
    assert p.penalties["cliff_not_at_end"] == config.PENALTIES["cliff_not_at_end"]
    # the clip opens on a 4 s camera-only shot, so the "no content in first 3 s" penalty also applies
    assert p.penalties["no_content_in_first_3s"] == config.PENALTIES["no_content_in_first_3s"]
    assert p.final_score == 80 - sum(p.penalties.values())


def test_ranking_dedup_and_rejection(eps):
    hi = mk(eps, [(1, 5)], {k: 9 for k in config.WEIGHTS}, "hi")
    dup = mk(eps, [(1, 6)], {k: 7 for k in config.WEIGHTS}, "dup")
    bad = mk(eps, [(1, 2)], {k: 10 for k in config.WEIGHTS}, "bad", cliff=(2, "Anna staggers back"))
    out = rank_promos([dup, bad, hi])
    by = {p.promo_id: p for p in out}
    assert by["hi"].rank == 1 and by["hi"].status == "valid"
    assert by["dup"].status == "duplicate" and by["dup"].duplicate_of == "hi"
    assert by["bad"].status == "rejected" and by["bad"].rank is None
    assert "hook verified" in by["hi"].rank_reason


# ---------------------------------------------------------------- LLM client plumbing (test double)
class ScriptedProvider(Provider):
    """Test double that returns pre-scripted strings - used only in unit tests."""
    name = "scripted"

    def __init__(self, replies):
        super().__init__("test")
        self.replies = list(replies)
        self.seen = []

    def complete(self, system, messages):
        self.seen.append(messages)
        return LLMResult(self.replies.pop(0), "test", self.name)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json")


def test_call_json_retries_with_feedback_then_caches(tmp_path):
    prov = ScriptedProvider(["not json", '{"x": 0}', '{"x": 2}'])
    client = LLMClient(prov, cache_dir=tmp_path / "c", raw_dir=tmp_path / "raw")
    validate = lambda d: [] if d.get("x", 0) > 0 else ["x must be positive"]  # noqa: E731
    assert client.call_json("t", "sys", "user", validate) == {"x": 2}
    assert "x must be positive" in prov.seen[2][-1]["content"]          # feedback was sent
    assert (tmp_path / "raw" / "t.json").exists()
    # second call: served from cache, provider not called again
    client2 = LLMClient(ScriptedProvider([]), cache_dir=tmp_path / "c")
    client2.provider.name = "scripted"
    assert client2.call_json("t", "sys", "user", validate) == {"x": 2}
    assert client2.calls[0].cached


def test_soft_problems_accepted_on_last_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
    prov = ScriptedProvider(['{"x": 1}', '{"x": 1}'])
    client = LLMClient(prov, cache_dir=tmp_path)
    assert client.call_json("t", "s", "u", lambda d: ["SOFT: too long"]) == {"x": 1}


def test_hard_problems_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
    client = LLMClient(ScriptedProvider(["{}", "{}"]), cache_dir=tmp_path, raw_dir=tmp_path / "raw")
    with pytest.raises(LLMError):
        client.call_json("t", "s", "u", lambda d: ["missing promos"])
    assert (tmp_path / "raw" / "t.FAILED.json").exists()


def test_replay_mode_never_invents(tmp_path):
    from src.llm_provider import ReplayProvider
    with pytest.raises(LLMError):
        LLMClient(ReplayProvider(), cache_dir=tmp_path).call_json("t", "s", "u")


def test_blind_viewer_blending(eps):
    p = mk(eps, [(1, 5)], {k: 8 for k in config.WEIGHTS}, "a")
    p.blind_test = {"understood_score": 4, "first_3s_stop_scroll": 2, "want_next_episode": 8}
    score_promo(p)
    assert p.effective_scores["standalone_clarity"] == 6.0      # (8 + 4) / 2
    assert p.effective_scores["hook"] == 5.0                    # (8 + 2) / 2
    assert p.effective_scores["cliffhanger"] == 8.0
    assert p.generator_only_score == 80.0
    assert p.final_score == round(80 - 10 * (0.20 * 3 + 0.15 * 2), 2)
