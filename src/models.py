"""Plain dataclasses shared by every stage. Times are float seconds from episode start."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx",
                "MUSIC": "sfx", "TRANSITION": "camera", "INSERT": "camera", "TEXT": "action",
                "SUPER": "action", "TITLE": "action"}


@dataclass
class Block:
    """One timed screenplay unit: a camera note, action, sound cue or dialogue line."""
    idx: int                      # 0-based position in the episode (the ID the LLM sees)
    kind: str                     # camera | action | sfx | dialogue
    label: str                    # raw label, e.g. "LADY JULIANA (Confused, Breathless)"
    speaker: Optional[str]        # dialogue only, e.g. "LADY JULIANA"
    delivery: Optional[str]       # dialogue parenthetical, e.g. "Confused, Breathless"
    start: Optional[float]
    end: Optional[float]
    text: str
    scene: int = 0
    line_no: int = 0              # line in the source file (for traceability)

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass
class Scene:
    number: int
    heading: str
    start: Optional[float]
    end: Optional[float]


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    source_file: str
    scenes: list[Scene] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [b.end for b in self.blocks if b.end is not None]
        scene_ends = [s.end for s in self.scenes if s.end is not None]
        return max(ends + scene_ends, default=0.0)

    def block(self, idx: int) -> Block:
        return self.blocks[idx]


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    first_block: int
    last_block: int

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class Issue:
    severity: str   # "error" (promo rejected) | "warning" (penalised / informational)
    code: str
    message: str


@dataclass
class Promo:
    promo_id: str
    source: str                         # "episode" | "cross_episode"
    title: str
    segments: list[Segment]
    hook_text: str
    hook_ref: tuple[int, int] | None    # (episode, block idx)
    cliffhanger_text: str
    cliffhanger_ref: tuple[int, int] | None
    open_question: str
    why_it_works: str
    context_gap: str                    # what a new viewer may not get (LLM self-report)
    llm_scores: dict[str, float]
    tags: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    penalties: dict[str, float] = field(default_factory=dict)
    effective_scores: dict[str, float] = field(default_factory=dict)  # after blending with blind test
    generator_only_score: float = 0.0  # what the score would be from the generator's self-scores alone
    weighted_score: float = 0.0
    final_score: float = 0.0
    status: str = "pending"            # valid | rejected | duplicate
    duplicate_of: str | None = None
    rank: int | None = None
    rank_reason: str = ""
    hook_offset_s: float | None = None     # where the hook beat starts, in promo time
    cliff_end_gap_s: float | None = None   # seconds between cliffhanger beat end and the cut
    blind_test: dict | None = None

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        out: list[int] = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = self.duration
        d["episodes"] = self.episodes
        return d
