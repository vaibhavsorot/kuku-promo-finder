"""Load a whole show (folder or .zip of screenplay files) into parsed Episodes.

Source files are only ever read, never modified. A file that fails to parse is reported
and skipped so one bad episode cannot sink a 50-episode run.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .models import Episode
from .screenplay_parser import parse_file

TEXT_SUFFIXES = {".txt", ".md", ".fountain", ".text"}


@dataclass
class IngestReport:
    show_title: str
    input_path: str
    episodes: list[Episode] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        eps = self.episodes
        kinds: dict[str, int] = {}
        for e in eps:
            for b in e.blocks:
                kinds[b.kind] = kinds.get(b.kind, 0) + 1
        return {
            "episodes": len(eps),
            "blocks": sum(len(e.blocks) for e in eps),
            "block_kinds": kinds,
            "untimed_blocks": sum(1 for e in eps for b in e.blocks if not b.timed),
            "total_runtime_s": round(sum(e.duration for e in eps), 1),
            "min_episode_s": round(min((e.duration for e in eps), default=0), 1),
            "max_episode_s": round(max((e.duration for e in eps), default=0), 1),
            "files_with_warnings": sum(1 for e in eps if e.warnings),
            "errors": len(self.errors),
        }


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "show"


def extract_zip(zip_path: Path, dest_root: Path) -> Path:
    """Extract into a fresh folder named after the zip's hash; refuses path traversal."""
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()[:12]
    dest = dest_root / f"{slugify(zip_path.stem)}-{digest}"
    if not dest.exists():
        dest.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise ValueError(f"unsafe path in zip: {member.filename}")
            zf.extractall(dest)
    return dest


def _guess_show_title(folder: Path, files: list[Path]) -> str:
    parents = {f.parent for f in files}
    if len(parents) == 1:
        p = parents.pop()
        if p != folder or p.name:
            return p.name
    return folder.name


def load_show(input_path: str | Path, extract_root: Path | None = None) -> IngestReport:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if path.is_file() and path.suffix.lower() == ".zip":
        path = extract_zip(path, extract_root or path.parent / "_extracted")

    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        and not p.name.startswith((".", "_")) and "__MACOSX" not in p.parts)
    report = IngestReport(show_title=_guess_show_title(path, files) if files else path.name,
                          input_path=str(path))
    if not files:
        report.errors.append(f"no screenplay text files found under {path}")
        return report

    by_number: dict[int, Episode] = {}
    content_hashes: dict[str, str] = {}
    for order, f in enumerate(files, start=1):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if digest in content_hashes:
            report.warnings.append(f"{f.name}: identical content to {content_hashes[digest]}; skipped")
            continue
        content_hashes[digest] = f.name
        try:
            ep = parse_file(f, fallback_number=order)
        except Exception as exc:  # noqa: BLE001 - report and continue
            report.errors.append(f"{f.name}: {exc}")
            continue
        if ep.number in by_number:
            report.warnings.append(f"{f.name}: duplicate episode number {ep.number}; kept "
                                   f"{Path(by_number[ep.number].source_file).name}")
            continue
        by_number[ep.number] = ep
        for w in ep.warnings:
            report.warnings.append(f"E{ep.number:02d} ({f.name}): {w}")

    report.episodes = [by_number[k] for k in sorted(by_number)]
    nums = [e.number for e in report.episodes]
    if nums:
        missing = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
        if missing:
            report.warnings.append(f"episode numbers missing from sequence: {missing}")
    return report
