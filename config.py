"""Central configuration. Everything tunable lives here or in environment variables (.env)."""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional; environment variables work without it
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).parent.resolve()
DEFAULT_INPUT_DIR = Path(os.getenv("PROMO_INPUT_DIR", ROOT / "data" / "input"))
OUTPUT_ROOT = Path(os.getenv("PROMO_OUTPUT_DIR", ROOT / "outputs"))
CACHE_DIR = Path(os.getenv("PROMO_CACHE_DIR", ROOT / "data" / "cache"))

# ---------------------------------------------------------------- LLM provider
# LLM_PROVIDER: gemini | openai_compatible | anthropic | claude_cli | replay
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty -> provider default
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # openai_compatible only
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "300"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # malformed-JSON retries
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))  # parallel episode calls
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "0"))  # free-tier rate limiting

# Bump when prompts change: it is part of the cache key, so old cached answers are not reused.
PROMPT_VERSION = "v4"

# ---------------------------------------------------------------- promo rules
MIN_PROMO_S = 20.0
MAX_PROMO_S = 60.0
HOOK_WINDOW_S = 3.0          # hook must START within the first 3 s of the promo
HOOK_TOLERANCE_S = 0.5       # small slack for timestamp rounding
CLIFF_WINDOW_S = 8.0         # cliffhanger beat must END within the last 8 s
MIN_SEGMENT_S = 2.0          # shorter segments are unusable as edits
MAX_SEGMENTS = 4
PROMOS_PER_EPISODE = 3
CROSS_EPISODE_PROMOS = 6
DUPLICATE_OVERLAP = 0.5      # >50% shared screen time with a better promo -> duplicate

# ---------------------------------------------------------------- scoring
# LLM rubric scores are 1-10. Weighted sum *10 -> 0-100, then deterministic penalties.
WEIGHTS = {
    "hook": 0.20,                 # does the first 3 s stop the scroll?
    "cliffhanger": 0.20,          # does the cut leave an open, urgent question?
    "standalone_clarity": 0.15,   # can a stranger follow it?
    "emotional_intensity": 0.15,  # slap / tears / rage / humiliation
    "curiosity": 0.10,            # mystery or secret beyond the final cut
    "coherence": 0.10,            # flows as one mini-story; no confusing jumps
    "premise_appeal": 0.10,       # sells the show's core trope / premise
}
PENALTIES = {
    "hook_not_in_first_3s": 12,      # LLM-claimed hook starts after the 3 s window
    "hook_quote_unsupported": 10,    # hook text not found in the source block
    "cliff_not_at_end": 8,           # LLM-claimed cliffhanger is not at the cut
    "cliff_quote_unsupported": 10,
    "no_content_in_first_3s": 5,     # only camera description, no line/action, in first 3 s
    "boundary_snapped": 2,           # a cut point fell inside a dialogue line (fixed)
    "per_extra_segment": 3,          # prefer one continuous stretch
    "cross_episode": 2,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# Two-judge blending. The generator knows the whole episode; the blind viewer sees only the clip
# transcript, like a stranger scrolling. Run 1 showed the generator over-rating clarity and hooks
# (docs/solution_document.md), so where a blind test exists these dimensions average both judges.
BLIND_BLEND = 0.5
BLIND_MAP = {"hook": "first_3s_stop_scroll",
             "standalone_clarity": "understood_score",
             "cliffhanger": "want_next_episode"}
