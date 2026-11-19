"""Headless smoke test of the Streamlit app against a saved run (skipped if no run exists)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_saved_run():
    from src.pipeline import list_saved
    if not list_saved():
        pytest.skip("no saved run in outputs/")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert any("Candidate promos" == m.label for m in at.metric)
    # open the detail view for a different promo to exercise the screenplay table
    sb = at.selectbox[-1]
    if len(sb.options) > 1:
        sb.select_index(1).run()
        assert not at.exception, at.exception
