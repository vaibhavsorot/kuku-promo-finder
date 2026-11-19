"""Parse one screenplay text file into an Episode.

Expected format (observed in the Kuku input pack):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL – DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    <text, possibly several lines>
    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"

The parser is deliberately tolerant: header case varies between files (episode 6 uses
"SCENE START:"), labels may lack bold markers, timestamps may be MM:SS(.cc) or HH:MM:SS(.cc),
and blocks without timestamps are kept but flagged instead of crashing the run.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import NON_DIALOGUE, Block, Episode, Scene

TIME_RE = r"(\d{1,2}:)?\d{1,3}:\d{1,2}(?:\.\d+)?"
_HEADER_KEYS = {
    "EPISODE NUMBER": "number", "EPISODE TITLE": "title",
    "EPISODE LOGLINE": "logline", "LOGLINE": "logline",
    "CONTINUITY NOTES": "continuity",
}
_SCENE_NO = re.compile(r"^\s*SCENE\s+(\d+)\s*$", re.I)
_SCENE_FIELD = re.compile(r"^\s*SCENE\s+(HEADING|START|END)\s*:\s*(.*)$", re.I)
_LABEL = re.compile(r"^\s*\*\*(?P<label>[^*]+?):?\*\*\s*:?\s*$")
_LABEL_PLAIN = re.compile(r"^\s*(?P<label>[A-Z][A-Z0-9 .'\-&/]+(?:\([^)]*\))*)\s*:\s*$")
_TS = re.compile(
    rf"^\s*Start\s*:\s*\(?\s*(?P<s>{TIME_RE})\s*\)?\s*[|,;]?\s*End\s*:\s*\(?\s*(?P<e>{TIME_RE})\s*\)?\s*$",
    re.I,
)
_EP_FROM_NAME = re.compile(r"episode[-_ ]?0*(\d+)", re.I)


def parse_time(value: str) -> float:
    """'01:05.60' -> 65.6 ; '1:02:03.5' -> 3723.5"""
    parts = value.strip().split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp {value!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return round(secs, 3)


def fmt_time(seconds: float | None) -> str:
    """65.6 -> '01:05.60' (same style as the source scripts)."""
    if seconds is None:
        return "--:--.--"
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _split_label(label: str) -> tuple[str, str, str | None, str | None]:
    """Return (kind, base, speaker, delivery) for a block label."""
    label = label.strip()
    base = re.split(r"\s*\(", label, maxsplit=1)[0].strip().upper()
    if base in NON_DIALOGUE:
        return NON_DIALOGUE[base], base, None, None
    parens = re.findall(r"\(([^)]*)\)", label)
    # drop technical markers like (O.S.) / (V.O.) / (CONT'D) from the delivery note
    delivery_parts = [p for p in parens if p.strip().upper().replace("'", "") not in
                      {"O.S.", "V.O.", "OS", "VO", "CONTD", "CONT.", "O.C."}]
    delivery = ", ".join(delivery_parts) or None
    return "dialogue", base, base, delivery


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_screenplay(text: str, source_file: str = "<memory>", fallback_number: int | None = None) -> Episode:
    lines = text.splitlines()
    meta = {"number": "", "title": "", "logline": "", "continuity": ""}
    warnings: list[str] = []

    # ---- header (everything before the first SCENE / block)
    for line in lines[:40]:
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        k = key.strip().upper()
        if k in _HEADER_KEYS and not meta[_HEADER_KEYS[k]]:
            meta[_HEADER_KEYS[k]] = val.strip()

    number = None
    m = re.search(r"(\d+)", meta["number"])
    if m:
        number = int(m.group(1))
    else:
        m = _EP_FROM_NAME.search(Path(source_file).name)
        if m:
            number = int(m.group(1))
            warnings.append("episode number missing from header; taken from file name")
        elif fallback_number is not None:
            number = fallback_number
            warnings.append("episode number missing; using file order")
        else:
            raise ValueError(f"{source_file}: cannot determine episode number")

    ep = Episode(number=number, title=meta["title"] or f"Episode {number}", logline=meta["logline"],
                 continuity=meta["continuity"], source_file=source_file, warnings=warnings)

    # ---- body
    scene_no = 0
    cur_scene: Scene | None = None
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        body = " ".join(t.strip() for t in cur["text"] if t.strip())
        kind, _base, speaker, delivery = _split_label(cur["label"])
        if cur["start"] is None:
            warnings.append(f"line {cur['line_no']}: block '{cur['label']}' has no timestamp")
        if not body:
            warnings.append(f"line {cur['line_no']}: block '{cur['label']}' has no text")
        start, end = cur["start"], cur["end"]
        if start is not None and end is not None and end < start:
            warnings.append(f"line {cur['line_no']}: end {fmt_time(end)} < start {fmt_time(start)}; swapped")
            start, end = end, start
        ep.blocks.append(Block(idx=len(ep.blocks), kind=kind, label=cur["label"], speaker=speaker,
                               delivery=delivery, start=start, end=end, text=body,
                               scene=scene_no, line_no=cur["line_no"]))
        cur = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("---") and "SCRIPT" in stripped.upper():
            flush()
            continue
        if (m := _SCENE_NO.match(line)):
            flush()
            scene_no = int(m.group(1))
            cur_scene = Scene(number=scene_no, heading="", start=None, end=None)
            ep.scenes.append(cur_scene)
            continue
        if (m := _SCENE_FIELD.match(line)):
            flush()
            field_name, val = m.group(1).upper(), m.group(2).strip()
            if cur_scene is None:
                scene_no += 1
                cur_scene = Scene(number=scene_no, heading="", start=None, end=None)
                ep.scenes.append(cur_scene)
            try:
                if field_name == "HEADING":
                    cur_scene.heading = val
                elif field_name == "START":
                    cur_scene.start = parse_time(val)
                else:
                    cur_scene.end = parse_time(val)
            except ValueError:
                warnings.append(f"line {i}: unparseable scene {field_name.lower()} '{val}'")
            continue
        if stripped.lower().startswith("actions and dialogue"):
            continue
        lm = _LABEL.match(line) or (_LABEL_PLAIN.match(line) if cur is None or cur["text"] else None)
        if lm and not _TS.match(line):
            flush()
            cur = {"label": lm.group("label").strip().rstrip(":"), "start": None, "end": None,
                   "text": [], "line_no": i}
            continue
        if cur is not None and cur["start"] is None and not cur["text"] and (tm := _TS.match(line)):
            try:
                cur["start"], cur["end"] = parse_time(tm.group("s")), parse_time(tm.group("e"))
            except ValueError:
                warnings.append(f"line {i}: unparseable timestamp '{stripped}'")
            continue
        if cur is not None and stripped:
            cur["text"].append(stripped)
    flush()

    if not ep.blocks:
        raise ValueError(f"{source_file}: no screenplay blocks found (unsupported format?)")

    # ---- sanity checks on chronology
    timed = [b for b in ep.blocks if b.timed]
    backwards = sum(1 for a, b in zip(timed, timed[1:]) if b.start + 1.0 < a.start)
    if backwards:
        warnings.append(f"{backwards} block(s) start >1 s earlier than the previous block (non-chronological order)")
    seen = set()
    for b in ep.blocks:
        key = (b.kind, b.start, b.text)
        if key in seen:
            warnings.append(f"line {b.line_no}: duplicate block '{b.label}' at {fmt_time(b.start)}")
        seen.add(key)
    for s in ep.scenes:
        if s.start is None or s.end is None:
            warnings.append(f"scene {s.number}: missing start/end")
    return ep


def parse_file(path: str | Path, fallback_number: int | None = None) -> Episode:
    path = Path(path)
    return parse_screenplay(read_text(path), source_file=str(path), fallback_number=fallback_number)
