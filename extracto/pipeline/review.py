"""HTML review page generation for QA."""

from __future__ import annotations

import csv
from pathlib import Path

TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Extracto Review</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
    th, td { border: 1px solid #ddd; padding: 8px; }
    th { background: #f5f5f5; position: sticky; top: 0; }
    .flags { color: #c00; font-weight: 500; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); grid-gap: 12px; }
    .card { border: 1px solid #ddd; padding: 8px; border-radius: 6px; }
    img { max-width: 100%; height: auto; display: block; }
    .meta { font-size: 12px; color: #555; margin-top: 4px; }
  </style>
</head>
<body>
  <h1>Extracto Review</h1>
  <p>Directory: %%OUT_DIR%%</p>
  <h2>Summary</h2>
  <table>
    <thead><tr>%%HEADER_CELLS%%</tr></thead>
    <tbody>%%ROWS%%</tbody>
  </table>
  <h2>Overlays</h2>
  <div class="grid">%%IMAGES%%</div>
</body>
</html>"""


def make_review_html(run_dir: str) -> str:
    """Generate an HTML review page from a run's summary.csv and overlays."""
    run = Path(run_dir)
    csv_path = run / "summary.csv"
    overlays_dir = run / "overlays"

    rows_html = ""
    header_cells = ""
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header_cells = "".join(f"<th>{h}</th>" for h in reader.fieldnames)
            for r in reader:
                row = "".join(f"<td>{r.get(h, '')}</td>" for h in reader.fieldnames)
                rows_html += f"<tr>{row}</tr>\n"

    images_html = ""
    if overlays_dir.exists():
        for p in sorted(overlays_dir.glob("*.png")):
            images_html += f'<div class="card"><img src="{p.name}" alt="{p.name}"/><div class="meta">{p.name}</div></div>\n'

    html = (
        TEMPLATE.replace("%%OUT_DIR%%", str(run_dir))
        .replace("%%HEADER_CELLS%%", header_cells)
        .replace("%%ROWS%%", rows_html)
        .replace("%%IMAGES%%", images_html)
    )
    out_path = run / "index.html"
    out_path.write_text(html)
    return str(out_path)
