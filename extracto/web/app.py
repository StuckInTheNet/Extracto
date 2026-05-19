"""Extracto Web Demo — polished demo site for prospective clients.

Start with:
    python -m extracto.web.app

Then open http://localhost:8080 in your browser.

Development mode (auto-reload on code changes):
    python -m extracto.web.app --dev
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

from extracto.pipeline.auto import auto_extract_single
from extracto.pipeline.indexer import build_index
from extracto.splitting.splitter import split_pdf
from extracto.storage.db import ExtractoDB

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload
app.secret_key = os.environ.get("SECRET_KEY", "extracto-demo-secret-key-change-in-prod")

DEMO_PASSWORD = os.environ.get("EXTRACTO_DEMO_PASSWORD", "ExtractMore2026")

UPLOAD_DIR = Path("/tmp/extracto_demo")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(UPLOAD_DIR / "demo.db")


def get_db():
    return ExtractoDB(DB_PATH)


def reset_session():
    """Wipe the temp DB and uploaded files for a clean slate."""
    global DB_PATH
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = str(UPLOAD_DIR / "demo.db")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


def demo_auth_required(f):
    """Decorator to require demo password for protected routes."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("demo_authenticated"):
            return redirect(url_for("demo_login", next=request.path))
        return f(*args, **kwargs)
    return decorated


@app.route("/demo/login", methods=["GET", "POST"])
def demo_login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == DEMO_PASSWORD:
            session["demo_authenticated"] = True
            return redirect(request.args.get("next", url_for("demo")))
        error = "Incorrect password."
    return render_template("demo_login.html", error=error)


@app.route("/demo")
@demo_auth_required
def demo():
    db = get_db()
    stats = db.get_stats()
    docs = db.get_all_documents()
    db.close()
    return render_template("demo.html", stats=stats, documents=docs)


@app.route("/clear", methods=["POST"])
@demo_auth_required
def clear():
    """Clear all uploaded documents and reset the session database."""
    reset_session()
    return redirect(url_for("demo"))


@app.route("/upload", methods=["POST"])
@demo_auth_required
def upload():
    files = request.files.getlist("files")
    if not files:
        return redirect(url_for("demo"))

    results = []
    db = get_db()

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            continue

        save_path = UPLOAD_DIR / f.filename
        f.save(str(save_path))

        start = time.monotonic()
        result = auto_extract_single(str(save_path))
        elapsed = (time.monotonic() - start) * 1000

        doc_id = db.store_extraction(
            str(save_path),
            result["classified_type"],
            result["extraction"],
            confidence=result["classification_confidence"],
            processing_time_ms=elapsed,
        )

        results.append({
            "filename": f.filename,
            "type": result["classified_type"],
            "confidence": result["classification_confidence"],
            "time_ms": round(elapsed, 1),
            "doc_id": doc_id,
            "fields": {
                k: v for k, v in result["extraction"].items()
                if k not in ("form_type", "extraction_mode", "stats",
                             "enriched_via_generic", "marks_detected",
                             "overlays_detected")
                and v is not None
            },
        })

    db.close()
    return render_template("results.html", results=results)


@app.route("/split-extract", methods=["POST"])
@demo_auth_required
def split_extract():
    """Split a multi-form PDF into individual documents, then extract each."""
    f = request.files.get("file")
    if not f or not f.filename or not f.filename.lower().endswith(".pdf"):
        return redirect(url_for("demo"))

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    split_dir = UPLOAD_DIR / "splits" / Path(f.filename).stem
    pipeline_start = time.monotonic()

    # Step 1: Split
    manifest = split_pdf(str(save_path), str(split_dir))
    segments = manifest.get("segments", [])

    # Step 2: Classify + extract each segment
    results = []
    type_counts = {}
    db = get_db()

    for seg in segments:
        seg_pdf = seg["pdf"]
        start = time.monotonic()
        result = auto_extract_single(seg_pdf)
        elapsed = (time.monotonic() - start) * 1000

        form_type = result["classified_type"]
        type_counts[form_type] = type_counts.get(form_type, 0) + 1

        doc_id = db.store_extraction(
            seg_pdf,
            form_type,
            result["extraction"],
            confidence=result["classification_confidence"],
            processing_time_ms=elapsed,
        )

        results.append({
            "filename": Path(seg_pdf).name,
            "pages": f"{seg['start'] + 1}–{seg['end'] + 1}",
            "type": form_type,
            "confidence": result["classification_confidence"],
            "time_ms": round(elapsed, 1),
            "doc_id": doc_id,
            "fields": {
                k: v for k, v in result["extraction"].items()
                if k not in ("form_type", "extraction_mode", "stats",
                             "enriched_via_generic", "marks_detected",
                             "overlays_detected")
                and v is not None
            },
        })

    db.close()
    pipeline_elapsed = (time.monotonic() - pipeline_start) * 1000

    import fitz
    doc = fitz.open(str(save_path))
    total_pages = len(doc)
    doc.close()

    return render_template(
        "split_results.html",
        filename=f.filename,
        total_pages=total_pages,
        total_segments=len(results),
        type_counts=type_counts,
        pipeline_time_ms=round(pipeline_elapsed, 1),
        results=results,
    )


@app.route("/provider-docs")
def provider_docs():
    """Show all documents for a provider side by side."""
    ids = request.args.get("ids", "")
    if not ids:
        return redirect(url_for("demo"))
    doc_ids = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    db = get_db()
    docs = []
    for did in doc_ids:
        doc = db.get_document(did)
        if doc:
            if doc.get("raw_json"):
                try:
                    doc["raw_json"] = json.dumps(json.loads(doc["raw_json"]), indent=2)
                except Exception:
                    pass
            doc["has_pdf"] = doc.get("file_path") and Path(doc["file_path"]).exists()
            docs.append(doc)
    db.close()
    provider_name = docs[0]["file_name"].split("/")[-1] if docs else "Provider"
    return render_template("provider_docs.html", docs=docs, provider_name=provider_name)


@app.route("/document/<int:doc_id>")
def document(doc_id):
    db = get_db()
    doc = db.get_document(doc_id)
    db.close()
    if not doc:
        return "Document not found", 404
    # Pretty-print raw JSON for display
    if doc.get("raw_json"):
        try:
            doc["raw_json"] = json.dumps(json.loads(doc["raw_json"]), indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
    # Check if the source PDF exists for inline viewing
    doc["has_pdf"] = doc.get("file_path") and Path(doc["file_path"]).exists()
    return render_template("document.html", doc=doc)


@app.route("/document/<int:doc_id>/pdf")
def document_pdf(doc_id):
    """Serve the original PDF for inline viewing."""
    db = get_db()
    doc = db.get_document(doc_id)
    db.close()
    if not doc or not doc.get("file_path"):
        return "Not found", 404
    pdf_path = Path(doc["file_path"])
    if not pdf_path.exists():
        return "File not found", 404
    return send_file(str(pdf_path), mimetype="application/pdf")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    search_type = request.args.get("type", "field")
    if not query:
        return render_template("search.html", results=None, query="", search_type="field")

    db = get_db()
    if search_type == "diagnosis":
        results = db.search_by_diagnosis(query.upper())
    elif search_type == "cpt":
        results = db.search_by_cpt(query)
    elif search_type == "provider":
        results = db.search_by_provider(query)
    else:
        results = db.search_by_field(query)
    db.close()
    return render_template("search.html", results=results, query=query, search_type=search_type)


@app.route("/index-records", methods=["POST"])
@demo_auth_required
def index_records():
    import fitz

    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return redirect(url_for("demo"))

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    pipeline_start = time.monotonic()

    # Step 1: Index by provider/DOS
    idx = build_index(str(save_path))
    segments = [s for s in idx.get("segments", []) if s.get("doc_type") not in ("Cover Sheet", "Separator")]

    db = get_db()
    db.store_index(str(save_path), idx.get("segments", []))

    # Step 2: Extract each segment's pages into a separate PDF and run extraction
    source_doc = fitz.open(str(save_path))
    seg_dir = UPLOAD_DIR / "index_segments" / Path(f.filename).stem
    seg_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        start_pg = min(seg["start_page"], seg["end_page"]) - 1  # fitz is 0-based
        end_pg = max(seg["start_page"], seg["end_page"]) - 1

        if start_pg < 0 or end_pg >= len(source_doc):
            continue

        seg_pdf = seg_dir / f"pages_{seg['start_page']}-{seg['end_page']}.pdf"
        new_doc = fitz.open()
        for pno in range(start_pg, end_pg + 1):
            new_doc.insert_pdf(source_doc, from_page=pno, to_page=pno)
        if len(new_doc) == 0:
            new_doc.close()
            continue
        new_doc.save(str(seg_pdf))
        new_doc.close()

        # Extract
        try:
            start = time.monotonic()
            result = auto_extract_single(str(seg_pdf))
            elapsed = (time.monotonic() - start) * 1000

            doc_id = db.store_extraction(
                str(seg_pdf),
                result["classified_type"],
                result["extraction"],
                confidence=result["classification_confidence"],
                processing_time_ms=elapsed,
            )

            seg["doc_id"] = doc_id
            seg["classified_type"] = result["classified_type"]
            seg["confidence"] = result["classification_confidence"]
            seg["time_ms"] = round(elapsed, 1)
            # Build a clean fields dict for display
            raw = result["extraction"]
            skip = {"form_type", "extraction_mode", "stats",
                    "enriched_via_generic", "marks_detected",
                    "overlays_detected", "key_value_pairs",
                    "selected_controls", "acroform_fields",
                    "entities", "dates", "tables",
                    "section_headers"}
            fields = {}
            # Promote key-value pairs to top level
            for kv in raw.get("key_value_pairs", []):
                label = kv.get("label", "")
                value = kv.get("value", "")
                if label and value:
                    fields[label] = value
            # Add simple scalar fields
            for k, v in raw.items():
                if k not in skip and v is not None:
                    if isinstance(v, (str, int, float, bool)):
                        fields[k] = v
                    elif isinstance(v, list) and v and isinstance(v[0], str):
                        fields[k] = ", ".join(v)
            # Promote useful entity fields
            ents = raw.get("entities", {})
            if ents.get("icd10_codes"):
                fields["ICD-10 Codes"] = ", ".join(ents["icd10_codes"])
            if ents.get("mrn"):
                fields["MRN"] = ", ".join(ents["mrn"])
            if ents.get("phone_numbers"):
                fields["Phone"] = ", ".join(ents["phone_numbers"])
            # Promote dates
            for d in raw.get("dates", []):
                ctx = d.get("context") or "Date"
                fields[ctx] = d.get("date", "")
            seg["fields"] = fields
        except Exception:
            seg["fields"] = {}

    source_doc.close()
    db.close()

    pipeline_elapsed = (time.monotonic() - pipeline_start) * 1000

    # Group segments by provider
    providers = {}
    for seg in segments:
        prov = seg["provider"]
        if prov not in providers:
            providers[prov] = []
        providers[prov].append(seg)

    return render_template(
        "index_results.html",
        index=idx,
        segments=segments,
        providers=providers,
        filename=f.filename,
        pipeline_time_ms=round(pipeline_elapsed, 1),
    )


@app.route("/explore")
@demo_auth_required
def explore():
    """Interactive SQL explorer for the extracted data."""
    query = request.args.get("q", "").strip()
    results = None
    columns = None
    error = None
    row_count = 0

    # Example queries for the UI
    examples = [
        ("All documents", "SELECT id, file_name, form_type, ROUND(classification_confidence * 100) || '%' as confidence, ROUND(processing_time_ms) || 'ms' as time FROM documents ORDER BY id"),
        ("Form type breakdown", "SELECT form_type, COUNT(*) as count, ROUND(AVG(classification_confidence) * 100) as avg_confidence_pct, ROUND(AVG(processing_time_ms)) as avg_time_ms FROM documents GROUP BY form_type ORDER BY count DESC"),
        ("Patient names", "SELECT d.file_name, d.form_type, f.field_value as patient_name FROM documents d JOIN fields f ON d.id = f.document_id WHERE f.field_name IN ('patient_name', 'member_name') ORDER BY f.field_value"),
        ("CMS-1500 charges", "SELECT d.file_name, f.field_value as total_charge FROM documents d JOIN fields f ON d.id = f.document_id WHERE d.form_type = 'cms1500' AND f.field_name = 'total_charge' ORDER BY CAST(f.field_value AS REAL) DESC"),
        ("ICD-10 codes", "SELECT d.file_name, diag.icd10_code, diag.position FROM diagnoses diag JOIN documents d ON d.id = diag.document_id ORDER BY diag.icd10_code"),
        ("Service lines", "SELECT d.file_name, sl.cpt_code, sl.date_of_service, sl.charge, sl.units FROM service_lines sl JOIN documents d ON d.id = sl.document_id ORDER BY sl.cpt_code"),
        ("All field names", "SELECT DISTINCT field_name FROM fields ORDER BY field_name"),
        ("Table schema", "SELECT name, type FROM pragma_table_list WHERE name NOT LIKE 'sqlite_%' ORDER BY name"),
    ]

    if query:
        # Safety: only allow SELECT queries
        normalized = query.strip().lower()
        if not normalized.startswith("select") and not normalized.startswith("pragma") and not normalized.startswith("explain"):
            error = "Only SELECT queries are allowed in the explorer."
        else:
            try:
                import sqlite3
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                cur = conn.execute(query)
                rows = cur.fetchmany(500)  # Cap at 500 rows
                if rows:
                    columns = rows[0].keys()
                    results = [dict(r) for r in rows]
                    row_count = len(results)
                else:
                    results = []
                    columns = []
                    row_count = 0
                conn.close()
            except Exception as e:
                error = str(e)

    return render_template("explore.html", query=query, results=results, columns=columns, error=error, row_count=row_count, examples=examples)


@app.route("/contact", methods=["POST"])
def contact():
    """Store contact form submission and send email notification."""
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    org = request.form.get("org", "").strip()
    message = request.form.get("message", "").strip()

    if not name or not email:
        return jsonify({"error": "Name and email required"}), 400

    # Store in a simple JSON file
    import datetime
    submissions_file = UPLOAD_DIR / "contact_submissions.json"
    submissions = []
    if submissions_file.exists():
        try:
            submissions = json.loads(submissions_file.read_text())
        except Exception:
            pass
    submissions.append({
        "name": name,
        "email": email,
        "org": org,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    submissions_file.write_text(json.dumps(submissions, indent=2))


    return jsonify({"ok": True})


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """JSON API endpoint for programmatic access."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    result = auto_extract_single(str(save_path))
    return jsonify(result)


def main():
    dev_mode = "--dev" in sys.argv
    port = int(os.environ.get("PORT", "8080"))

    print()
    print("  Extracto Demo Server")
    print("  " + "-" * 40)
    if dev_mode:
        print("  MODE: development (auto-reload ON)")
    print(f"  URL:  http://localhost:{port}")
    print(f"  Data: {DB_PATH}")
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=dev_mode,
        use_reloader=dev_mode,
    )


if __name__ == "__main__":
    main()
