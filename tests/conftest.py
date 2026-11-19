import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.screenplay_parser import parse_screenplay  # noqa: E402

SAMPLE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A bride is jilted at the altar.
CONTINUITY NOTES: Test fixture.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CHURCH – DAY
Scene start: 00:00.00
Scene end: 00:40.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the church doors.

**ANNA (Shocked):**
Start: (00:04.00) | End: (00:06.00)
"You slapped me!"

**ACTION:**
Start: (00:06.00) | End: (00:09.00)
Anna staggers back, holding her cheek.

**MARK (Cold):**
Start: (00:09.00) | End: (00:14.00)
"You were never going to be my wife."

**CAMERA:**
Start: (00:14.00) | End: (00:20.00)
Close-up on Anna's tears.

**ANNA (Whispering):**
Start: (00:20.00) | End: (00:24.00)
"Then who is she?"

SCENE 02
Scene heading: EXT. CHURCH – CONTINUOUS
Scene start: 00:24.00
Scene end: 00:40.00

**SOUND FX:**
Start: (00:24.00) | End: (00:26.00)
Thunder.

**MARK (O.S.) (Hard):**
Start: (00:26.00) | End: (00:33.00)
"Your sister. And tomorrow you lose everything."

**CAMERA:**
Start: (00:33.00) | End: (00:40.00)
Anna's face hardens. Cut to black.

--- SCRIPT END ---
"""


@pytest.fixture
def sample_text():
    return SAMPLE


@pytest.fixture
def ep():
    return parse_screenplay(SAMPLE, "episode-007-test.txt")


@pytest.fixture
def eps(ep):
    return {ep.number: ep}
