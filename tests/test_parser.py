import zipfile

import pytest

from src.ingestion import load_show
from src.screenplay_parser import fmt_time, parse_screenplay, parse_time


def test_time_roundtrip():
    assert parse_time("01:05.60") == 65.6
    assert parse_time("1:02:03.5") == 3723.5
    assert fmt_time(65.6) == "01:05.60"
    with pytest.raises(ValueError):
        parse_time("abc")


def test_parses_header_scenes_and_blocks(ep):
    assert ep.number == 7
    assert ep.logline.startswith("A bride")
    assert len(ep.scenes) == 2 and ep.scenes[1].start == 24.0
    assert len(ep.blocks) == 9
    kinds = [b.kind for b in ep.blocks]
    assert kinds.count("dialogue") == 4 and kinds.count("camera") == 3 and "sfx" in kinds
    slap = ep.blocks[1]
    assert (slap.speaker, slap.delivery, slap.start, slap.end) == ("ANNA", "Shocked", 4.0, 6.0)
    assert slap.text == '"You slapped me!"'
    mark = ep.blocks[7]
    assert mark.speaker == "MARK" and mark.delivery == "Hard"   # (O.S.) stripped from delivery
    assert ep.duration == 40.0 and not ep.warnings


def test_uppercase_scene_headers_like_episode_6(sample_text):
    text = sample_text.replace("Scene start:", "SCENE START:").replace("Scene end:", "SCENE END:") \
                      .replace("Scene heading:", "SCENE HEADING:")
    e = parse_screenplay(text)
    assert [s.start for s in e.scenes] == [0.0, 24.0] and e.scenes[0].heading.startswith("INT.")


def test_missing_timestamp_is_flagged_not_fatal(sample_text):
    text = sample_text.replace("Start: (00:04.00) | End: (00:06.00)\n", "")
    e = parse_screenplay(text)
    assert not e.blocks[1].timed
    assert any("no timestamp" in w for w in e.warnings)


def test_reversed_timestamp_swapped(sample_text):
    e = parse_screenplay(sample_text.replace("Start: (00:04.00) | End: (00:06.00)", "Start: (00:06.00) | End: (00:04.00)"))
    assert (e.blocks[1].start, e.blocks[1].end) == (4.0, 6.0)
    assert any("swapped" in w for w in e.warnings)


def test_episode_number_from_filename(sample_text):
    e = parse_screenplay(sample_text.replace("EPISODE NUMBER: Episode 7\n", ""), "episode-012-x.txt")
    assert e.number == 12


def test_garbage_raises():
    with pytest.raises(ValueError):
        parse_screenplay("just some prose\nwith no blocks", "episode-001.txt")


def test_variant_timestamp_and_label_formats(sample_text):
    text = sample_text.replace("Start: (00:09.00) | End: (00:14.00)", "start: 00:09.00, end: 00:14.00")
    e = parse_screenplay(text)
    assert e.blocks[3].start == 9.0 and e.blocks[3].end == 14.0


def test_ingest_folder_zip_duplicates_and_bad_files(tmp_path, sample_text):
    show = tmp_path / "My Show"
    show.mkdir()
    (show / "episode-001-a.txt").write_text(sample_text.replace("Episode 7", "Episode 1"), encoding="utf-8")
    (show / "episode-002-a.txt").write_text(sample_text.replace("Episode 7", "Episode 2"), encoding="utf-8")
    (show / "episode-002-copy.txt").write_text(sample_text.replace("Episode 7", "Episode 2"), encoding="utf-8")
    (show / "episode-004-bad.txt").write_text("not a screenplay", encoding="utf-8")
    rep = load_show(show)
    assert [e.number for e in rep.episodes] == [1, 2]
    assert any("identical content" in w for w in rep.warnings)
    assert any("episode-004-bad" in e for e in rep.errors)
    z = tmp_path / "show.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in show.iterdir():
            zf.write(f, f"My Show/{f.name}")
    rep2 = load_show(z, extract_root=tmp_path / "x")
    assert [e.number for e in rep2.episodes] == [1, 2] and rep2.show_title == "My Show"


def test_real_input_pack_if_present():
    from pathlib import Path
    import config
    packs = [p for p in Path(config.DEFAULT_INPUT_DIR).glob("*") if p.is_dir()]
    if not packs:
        pytest.skip("input pack not present")
    rep = load_show(packs[0])
    assert len(rep.episodes) >= 1 and not rep.errors
    assert all(b.timed for e in rep.episodes for b in e.blocks)
