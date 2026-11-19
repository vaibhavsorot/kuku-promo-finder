"""Render docs/solution_document.md to docs/solution_document.pdf (reportlab, compact 2-page layout).

Supports the small Markdown subset the document uses: # / ## headings, paragraphs, '- ' bullets,
**bold**, `code`, and pipe tables.   python docs/build_solution_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).parent
SRC = HERE / "solution_document.md"
OUT = HERE / "solution_document.pdf"

BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=9.4, leading=11.8, alignment=TA_LEFT,
                      spaceAfter=3)
H1 = ParagraphStyle("h1", parent=BASE, fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=3)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=5,
                    spaceAfter=1.5, textColor=colors.HexColor("#1f3a5f"))
SMALL = ParagraphStyle("small", parent=BASE, fontSize=8.2, leading=10, spaceAfter=0)


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    return text


def build(src: Path = SRC, out: Path = OUT) -> int:
    lines = src.read_text(encoding="utf-8").splitlines()
    story, bullets, table = [], [], []

    def flush_bullets():
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(inline(b), BASE), leftIndent=8) for b in bullets],
                                      bulletType="bullet", start="•", leftIndent=8, bulletFontSize=7))
            bullets.clear()

    def flush_table():
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|?\s*:?-{2,}.*", r.strip())]
            data = [[Paragraph(inline(c.strip()), SMALL) for c in r.strip().strip("|").split("|")] for r in rows]
            ncol = max(len(r) for r in data)
            data = [r + [""] * (ncol - len(r)) for r in data]
            # column widths proportional to (capped) content length, so short columns stay narrow
            raw = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            need = [max(8, min(60, max(len(r[i]) if i < len(r) else 0 for r in raw))) for i in range(ncol)]
            avail = A4[0] - 26 * mm
            t = Table(data, repeatRows=1, colWidths=[avail * n / sum(need) for n in need])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
            story.append(t)
            story.append(Spacer(1, 3))
            table.clear()

    for line in lines:
        s = line.rstrip()
        if s.startswith("|"):
            flush_bullets(); table.append(s); continue
        flush_table()
        if not s.strip():
            flush_bullets(); continue
        if s.startswith("# "):
            flush_bullets(); story.append(Paragraph(inline(s[2:]), H1))
        elif s.startswith("## "):
            flush_bullets(); story.append(Paragraph(inline(s[3:]), H2))
        elif re.match(r"^\s*[-*] ", s):
            bullets.append(re.sub(r"^\s*[-*] ", "", s))
        else:
            flush_bullets(); story.append(Paragraph(inline(s), BASE))
    flush_bullets(); flush_table()

    pages = []
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm,
                            topMargin=11 * mm, bottomMargin=11 * mm, title="Kuku Promo Finder - Solution")
    doc.build(story, onLaterPages=lambda c, d: pages.append(d.page), onFirstPage=lambda c, d: pages.append(d.page))
    n = len(pages)
    print(f"wrote {out} ({n} page{'s' if n != 1 else ''})")
    return n


if __name__ == "__main__":
    sys.exit(0 if build() <= 2 else 1)
