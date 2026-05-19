"""Medical records indexer — split a multi-provider records bundle by provider and DOS.

HIPAA COMPLIANCE NOTES:
- All processing runs locally. No data is transmitted to external services.
- Index output references pages by number; full PHI is NOT embedded in the
  index unless explicitly requested.
- Log output shows provider names and page counts only, not patient data.
- Designed for use by authorized parties (attorneys, paralegals) who already
  have lawful access to the records.

Pipeline:
1. Page analysis: extract header text, detect provider, extract dates
2. Boundary detection: identify where one document ends and next begins
3. Segment grouping: consecutive same-provider pages = one document
4. Index generation: provider → DOS → page range → document type
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz


# --- Constants ---

HEADER_HEIGHT_PT = 120.0  # scan top ~1.7 inches for provider info
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")

# Date context: keywords that indicate what kind of date follows
DOS_KEYWORDS = [
    "date of service", "visit date", "encounter date", "date:", "dos:",
    "exam date", "procedure date", "session date", "date of visit",
    "treatment date", "date of injury", "injury date", "admission date",
]
NON_DOS_KEYWORDS = [
    "date of birth", "dob:", "dob ", "print date", "printed:",
    "fax date", "date prepared", "date received", "date paid",
    "statement date",
]

# Separator/cover page detection
SEPARATOR_PATTERNS = [
    re.compile(r"={5,}", re.I),
    re.compile(r"separator", re.I),
    re.compile(r"records?\s+from", re.I),
]
COVER_PATTERNS = [
    re.compile(r"medical records?\s+transmittal", re.I),
    re.compile(r"records?\s+retrieval", re.I),
    re.compile(r"cover\s+(sheet|page)", re.I),
    re.compile(r"confidential.*hipaa", re.I),
]

DOC_TYPE_PATTERNS = [
    (re.compile(r"office\s+visit|progress\s+note", re.I), "Office Visit"),
    (re.compile(r"emergency|er\s+report", re.I), "ER Report"),
    (re.compile(r"operative\s+report|surgery", re.I), "Operative Report"),
    (re.compile(r"discharge\s+summary", re.I), "Discharge Summary"),
    (re.compile(r"radiology|mri|ct\s+scan|x-ray|imaging", re.I), "Imaging Report"),
    (re.compile(r"physical\s+therapy|pt\s+note|pt\s+session", re.I), "PT Note"),
    (re.compile(r"lab\s+result|laboratory|quest\s+diagnostics", re.I), "Lab Result"),
    (re.compile(r"chiropractic|manipulation", re.I), "Chiropractic Note"),
]


@dataclass
class PageInfo:
    """Extracted metadata for a single page."""

    page_num: int  # 1-based
    header_text: str  # text from top region
    full_text: str
    provider_name: str | None = None
    dos: str | None = None
    doc_type: str | None = None
    is_separator: bool = False
    is_cover: bool = False
    word_count: int = 0


@dataclass
class DocumentSegment:
    """A contiguous group of pages forming one document."""

    start_page: int
    end_page: int
    provider: str
    dos: str | None
    doc_type: str
    page_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_page": self.start_page,
            "end_page": self.end_page,
            "provider": self.provider,
            "dos": self.dos,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
        }


# --- Page analysis ---

def analyze_page(page: fitz.Page, page_num: int) -> PageInfo:
    """Extract metadata from a single page."""
    full_text = page.get_text()
    words = page.get_text("words")
    word_count = len(words)

    # Header: text in top HEADER_HEIGHT_PT points
    header_words = [w[4] for w in words if w[1] <= HEADER_HEIGHT_PT]
    header_text = " ".join(header_words)

    info = PageInfo(
        page_num=page_num,
        header_text=header_text,
        full_text=full_text,
        word_count=word_count,
    )

    # Check for separator / cover pages
    for pat in SEPARATOR_PATTERNS:
        if pat.search(full_text):
            info.is_separator = True
            break
    for pat in COVER_PATTERNS:
        if pat.search(full_text):
            info.is_cover = True
            break

    if info.is_separator or info.is_cover:
        # Still try to extract provider from cover/separator text
        info.provider_name = _extract_provider_from_text(full_text, words)
        return info

    # Extract provider from header (the practice name is typically the
    # largest/boldest text in the top region — first line of the header)
    info.provider_name = _extract_provider_from_header(page, header_text, words)

    # Extract date of service
    info.dos = _extract_dos(full_text, words)

    # Classify document type
    info.doc_type = _classify_doc_type(full_text)

    return info


def _extract_provider_from_header(page: fitz.Page, header_text: str, words: list) -> str | None:
    """Extract the provider/practice name from the page header.

    Strategy: the practice name is typically the FIRST LINE of text in the
    header region, rendered in a larger font. We use get_text("dict") to
    find the largest-font span in the top region.
    """
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return _extract_provider_from_text(header_text, words)

    # Find the largest font in the header region
    header_spans: list[tuple[float, float, str]] = []
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox", (0, 0, 0, 0))
                if bbox[1] > HEADER_HEIGHT_PT:
                    continue
                text = span.get("text", "").strip()
                size = span.get("size", 10)
                if text and len(text) > 2:
                    header_spans.append((size, bbox[1], text))

    if not header_spans:
        return None

    # The practice name is typically the largest font in the header
    header_spans.sort(key=lambda s: (-s[0], s[1]))
    best = header_spans[0][2]

    # Skip generic headers that aren't provider names
    skip_patterns = [
        "medical records", "transmittal", "confidential",
        "cover", "separator", "===",
    ]
    if any(p in best.lower() for p in skip_patterns):
        # Try the next largest
        for _, _, text in header_spans[1:]:
            if not any(p in text.lower() for p in skip_patterns):
                return text
        return None

    return best


def _extract_provider_from_text(text: str, words: list) -> str | None:
    """Fallback: extract provider from 'Records from X' or 'Provider: X' patterns."""
    patterns = [
        re.compile(r"records?\s+from\s+(.+?)(?:\n|$)", re.I),
        re.compile(r"provider[:\s]+(.+?)(?:\n|$)", re.I),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            name = m.group(1).strip().rstrip("=").strip()
            if name and len(name) > 2:
                return name
    return None


def _extract_dos(text: str, words: list) -> str | None:
    """Extract the date of service from page text.

    Prefers dates near DOS-keyword contexts. Excludes dates near DOB/fax/print contexts.
    """
    text_lower = text.lower()
    all_dates = DATE_RE.findall(text)
    if not all_dates:
        return None

    # Score each date by its context
    scored: list[tuple[str, int]] = []
    for date_str in all_dates:
        idx = text_lower.find(date_str.lower()) if date_str.lower() in text_lower else -1
        if idx < 0:
            scored.append((date_str, 0))
            continue

        # Check surrounding 35 chars for context keywords (tight window
        # avoids picking up adjacent label keywords from other fields)
        window = text_lower[max(0, idx - 35) : idx + len(date_str) + 10]
        score = 0
        for kw in DOS_KEYWORDS:
            if kw in window:
                score += 10
                break
        for kw in NON_DOS_KEYWORDS:
            if kw in window:
                score -= 20
                break
        # Dates near the top of the page are more likely to be DOS
        if idx < 500:
            score += 3
        scored.append((date_str, score))

    # Pick the highest-scoring date
    scored.sort(key=lambda s: -s[1])
    best_date, best_score = scored[0]
    if best_score >= 0:
        return best_date
    # All dates scored negative (all near DOB/fax contexts) — return None
    return None


def _classify_doc_type(text: str) -> str:
    """Classify the document type based on text content."""
    for pat, doc_type in DOC_TYPE_PATTERNS:
        if pat.search(text):
            return doc_type
    return "Clinical Note"


# --- Boundary detection ---

def detect_boundaries(pages: list[PageInfo]) -> list[DocumentSegment]:
    """Group consecutive pages into document segments.

    Boundary rules:
    1. Separator/cover pages start a new segment
    2. Provider name change starts a new segment
    3. Date of service change (within same provider) starts a new segment
    4. Continuation pages (no provider header, similar to previous) stay grouped
    """
    if not pages:
        return []

    segments: list[DocumentSegment] = []
    current_start = 0
    current_provider = pages[0].provider_name or "Unknown"
    current_dos = pages[0].dos
    current_type = pages[0].doc_type or "Unknown"

    def flush(end_idx: int):
        nonlocal current_start
        seg = DocumentSegment(
            start_page=pages[current_start].page_num,
            end_page=pages[end_idx].page_num,
            provider=current_provider,
            dos=current_dos,
            doc_type=current_type,
            page_count=end_idx - current_start + 1,
        )
        segments.append(seg)
        current_start = end_idx + 1

    for i in range(1, len(pages)):
        prev = pages[i - 1]
        curr = pages[i]

        # Separator or cover → start new segment
        if curr.is_separator or curr.is_cover:
            flush(i - 1)
            # The separator/cover is its own segment
            segments.append(DocumentSegment(
                start_page=curr.page_num,
                end_page=curr.page_num,
                provider=curr.provider_name or current_provider,
                dos=None,
                doc_type="Cover Sheet" if curr.is_cover else "Separator",
                page_count=1,
            ))
            current_start = i + 1
            if curr.provider_name:
                current_provider = curr.provider_name
            current_dos = None
            current_type = "Unknown"
            continue

        # Provider change → new segment
        if curr.provider_name and curr.provider_name != current_provider:
            flush(i - 1)
            current_provider = curr.provider_name
            current_dos = curr.dos
            current_type = curr.doc_type or "Unknown"
            continue

        # DOS change within same provider → new segment
        if curr.dos and curr.dos != current_dos and curr.provider_name:
            flush(i - 1)
            current_dos = curr.dos
            current_type = curr.doc_type or current_type
            continue

        # Continuation page: no provider header but same general format
        # Update DOS/type if new info available
        if curr.dos and not current_dos:
            current_dos = curr.dos
        if curr.doc_type and curr.doc_type != "Clinical Note":
            current_type = curr.doc_type

    # Final segment
    if current_start < len(pages):
        flush(len(pages) - 1)

    return segments


# --- Index output ---

def build_index(
    pdf_path: str,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Analyze a medical records PDF and build a provider/DOS index.

    Args:
        pdf_path: Path to the records bundle PDF
        out_dir: Optional output directory for index files

    Returns:
        Index dict with segments, provider summary, and stats.
    """
    doc = fitz.open(pdf_path)

    # Phase 1: analyze each page
    pages: list[PageInfo] = []
    for i, page in enumerate(doc):
        info = analyze_page(page, page_num=i + 1)
        pages.append(info)

    doc.close()

    # Phase 2: detect document boundaries
    segments = detect_boundaries(pages)

    # Phase 3: build provider summary
    provider_summary: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        if seg.doc_type in ("Cover Sheet", "Separator"):
            continue
        provider_summary[seg.provider].append({
            "dos": seg.dos,
            "doc_type": seg.doc_type,
            "pages": f"{seg.start_page}-{seg.end_page}",
            "page_count": seg.page_count,
        })

    # Sort encounters by DOS within each provider
    for provider in provider_summary:
        provider_summary[provider].sort(key=lambda e: e.get("dos") or "")

    index = {
        "source_pdf": pdf_path,
        "total_pages": len(pages),
        "total_segments": len(segments),
        "providers": dict(provider_summary),
        "provider_count": len(provider_summary),
        "encounter_count": sum(len(v) for v in provider_summary.values()),
        "segments": [s.to_dict() for s in segments],
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # JSON index
        (out_path / "index.json").write_text(json.dumps(index, indent=2))

        # CSV for quick review
        csv_path = out_path / "index.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "provider", "dos", "doc_type", "start_page", "end_page", "page_count"
            ])
            w.writeheader()
            for seg in segments:
                if seg.doc_type in ("Cover Sheet", "Separator"):
                    continue
                w.writerow(seg.to_dict())

        # HTML summary
        _write_html_index(out_path / "index.html", index)

    return index


def _write_html_index(path: Path, index: dict):
    """Generate an HTML summary of the records index."""
    rows = ""
    for seg in index["segments"]:
        if seg["doc_type"] in ("Cover Sheet", "Separator"):
            continue
        rows += (
            f"<tr><td>{seg['provider']}</td><td>{seg.get('dos', '-')}</td>"
            f"<td>{seg['doc_type']}</td><td>{seg['start_page']}-{seg['end_page']}</td>"
            f"<td>{seg['page_count']}</td></tr>\n"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Medical Records Index</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
    .hipaa {{ background: #fff3cd; padding: 8px; border-radius: 4px; margin-bottom: 16px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Medical Records Index</h1>
  <div class="hipaa">CONFIDENTIAL — Contains references to Protected Health Information.
  Access restricted to authorized parties under HIPAA (45 CFR Parts 160, 164).</div>
  <div class="meta">
    Source: {index['source_pdf']}<br>
    Total pages: {index['total_pages']} |
    Providers: {index['provider_count']} |
    Encounters: {index['encounter_count']}
  </div>
  <table>
    <thead>
      <tr><th>Provider</th><th>Date of Service</th><th>Document Type</th><th>Pages</th><th>Count</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>"""
    path.write_text(html)


def format_index_report(index: dict) -> str:
    """Format a terminal-friendly summary of the index."""
    lines = []
    lines.append("=" * 72)
    lines.append("MEDICAL RECORDS INDEX")
    lines.append("=" * 72)
    lines.append(f"Source: {index['source_pdf']}")
    lines.append(f"Total pages: {index['total_pages']}")
    lines.append(f"Providers: {index['provider_count']}")
    lines.append(f"Encounters: {index['encounter_count']}")
    lines.append("")

    lines.append(f"{'Provider':<35} {'DOS':<12} {'Type':<20} {'Pages':<10}")
    lines.append("-" * 72)

    for seg in index["segments"]:
        if seg["doc_type"] in ("Cover Sheet", "Separator"):
            continue
        lines.append(
            f"{seg['provider']:<35} {(seg.get('dos') or '-'):<12} "
            f"{seg['doc_type']:<20} {seg['start_page']}-{seg['end_page']}"
        )

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)
