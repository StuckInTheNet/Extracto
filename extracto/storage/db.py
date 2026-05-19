"""SQLite storage layer for extracted document data.

Stores extraction results in a local database file. No server needed.
Each extraction run appends to the database, building a searchable
archive of all processed documents.

Tables:
- documents: one row per processed PDF (file, type, timestamp)
- fields: one row per extracted field (document_id, field_name, value)
- service_lines: one row per CMS-1500/EOB service line
- diagnoses: one row per ICD-10 code found
- index_entries: one row per records-index segment (provider, DOS, pages)

Usage:
    from extracto.storage.db import ExtractoDB
    db = ExtractoDB("extracto.db")
    db.store_extraction("form.pdf", "cms1500", extraction_dict)
    results = db.search_by_diagnosis("M54.2")
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class ExtractoDB:
    """SQLite-backed storage for Extracto extraction results."""

    def __init__(self, db_path: str = "extracto.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                form_type TEXT NOT NULL,
                classification_confidence REAL,
                processing_time_ms REAL,
                extracted_at TEXT NOT NULL,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id),
                field_name TEXT NOT NULL,
                field_value TEXT,
                field_type TEXT
            );

            CREATE TABLE IF NOT EXISTS service_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id),
                line_number INTEGER,
                date_of_service TEXT,
                cpt_code TEXT,
                pos TEXT,
                charge REAL,
                units INTEGER,
                reason_codes TEXT
            );

            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id),
                icd10_code TEXT NOT NULL,
                position INTEGER
            );

            CREATE TABLE IF NOT EXISTS index_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bundle_path TEXT NOT NULL,
                provider TEXT NOT NULL,
                date_of_service TEXT,
                doc_type TEXT,
                start_page INTEGER,
                end_page INTEGER,
                page_count INTEGER,
                indexed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(form_type);
            CREATE INDEX IF NOT EXISTS idx_documents_file ON documents(file_name);
            CREATE INDEX IF NOT EXISTS idx_fields_name ON fields(field_name);
            CREATE INDEX IF NOT EXISTS idx_fields_value ON fields(field_value);
            CREATE INDEX IF NOT EXISTS idx_diagnoses_code ON diagnoses(icd10_code);
            CREATE INDEX IF NOT EXISTS idx_service_lines_cpt ON service_lines(cpt_code);
            CREATE INDEX IF NOT EXISTS idx_service_lines_dos ON service_lines(date_of_service);
            CREATE INDEX IF NOT EXISTS idx_index_provider ON index_entries(provider);
            CREATE INDEX IF NOT EXISTS idx_index_dos ON index_entries(date_of_service);
        """)
        self.conn.commit()

    def store_extraction(
        self,
        file_path: str,
        form_type: str,
        extraction: dict[str, Any],
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
    ) -> int:
        """Store a single document's extraction results. Returns the document ID."""
        now = datetime.utcnow().isoformat()
        file_name = Path(file_path).name

        c = self.conn.cursor()
        c.execute(
            """INSERT INTO documents (file_path, file_name, form_type,
               classification_confidence, processing_time_ms, extracted_at, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_path, file_name, form_type, confidence, processing_time_ms,
             now, json.dumps(extraction, default=str)),
        )
        doc_id = c.lastrowid

        # Store individual fields
        skip_keys = {"form_type", "extraction_mode", "stats", "enriched_via_generic",
                     "service_lines", "diagnoses", "marks_detected", "overlays_detected"}
        for key, value in extraction.items():
            if key in skip_keys:
                continue
            if isinstance(value, (list, dict)):
                field_value = json.dumps(value, default=str)
                field_type = "json"
            elif isinstance(value, bool):
                field_value = str(value)
                field_type = "boolean"
            elif isinstance(value, (int, float)):
                field_value = str(value)
                field_type = "numeric"
            else:
                field_value = str(value) if value is not None else None
                field_type = "text"

            c.execute(
                "INSERT INTO fields (document_id, field_name, field_value, field_type) VALUES (?, ?, ?, ?)",
                (doc_id, key, field_value, field_type),
            )

        # Store service lines
        for i, line in enumerate(extraction.get("service_lines", [])):
            if isinstance(line, dict):
                c.execute(
                    """INSERT INTO service_lines
                       (document_id, line_number, date_of_service, cpt_code, pos, charge, units, reason_codes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, i + 1,
                     line.get("date_from") or line.get("date_of_service"),
                     line.get("cpt"),
                     line.get("pos"),
                     line.get("charge"),
                     line.get("units"),
                     json.dumps(line.get("reason_codes") or line.get("reason", []))),
                )

        # Store diagnoses
        diag_list = extraction.get("diagnoses", [])
        if isinstance(diag_list, list):
            for i, code in enumerate(diag_list):
                if isinstance(code, str):
                    c.execute(
                        "INSERT INTO diagnoses (document_id, icd10_code, position) VALUES (?, ?, ?)",
                        (doc_id, code, i + 1),
                    )

        self.conn.commit()
        return doc_id

    def store_index(self, bundle_path: str, segments: list[dict[str, Any]]):
        """Store records index entries from the indexer."""
        now = datetime.utcnow().isoformat()
        c = self.conn.cursor()
        for seg in segments:
            if seg.get("doc_type") in ("Cover Sheet", "Separator"):
                continue
            c.execute(
                """INSERT INTO index_entries
                   (bundle_path, provider, date_of_service, doc_type,
                    start_page, end_page, page_count, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (bundle_path, seg["provider"], seg.get("dos"),
                 seg.get("doc_type"), seg.get("start_page"),
                 seg.get("end_page"), seg.get("page_count"), now),
            )
        self.conn.commit()

    # --- Query methods ---

    def search_by_diagnosis(self, icd10_code: str) -> list[dict]:
        """Find all documents containing a specific ICD-10 code."""
        c = self.conn.cursor()
        c.execute("""
            SELECT d.id as doc_id, d.file_name, d.form_type, d.extracted_at, di.icd10_code
            FROM diagnoses di
            JOIN documents d ON di.document_id = d.id
            WHERE di.icd10_code = ?
            ORDER BY d.extracted_at DESC
        """, (icd10_code,))
        return [dict(r) for r in c.fetchall()]

    def search_by_cpt(self, cpt_code: str) -> list[dict]:
        """Find all service lines with a specific CPT code."""
        c = self.conn.cursor()
        c.execute("""
            SELECT d.id as doc_id, d.file_name, d.form_type, sl.date_of_service, sl.cpt_code, sl.charge, sl.units
            FROM service_lines sl
            JOIN documents d ON sl.document_id = d.id
            WHERE sl.cpt_code = ?
            ORDER BY sl.date_of_service
        """, (cpt_code,))
        return [dict(r) for r in c.fetchall()]

    def search_by_field(self, field_name: str, field_value: str | None = None) -> list[dict]:
        """Find documents by field name (and optionally value)."""
        c = self.conn.cursor()
        if field_value:
            c.execute("""
                SELECT d.id as doc_id, d.file_name, d.form_type, f.field_name, f.field_value
                FROM fields f
                JOIN documents d ON f.document_id = d.id
                WHERE f.field_name = ? AND f.field_value LIKE ?
                ORDER BY d.extracted_at DESC
            """, (field_name, f"%{field_value}%"))
        else:
            c.execute("""
                SELECT d.id as doc_id, d.file_name, d.form_type, f.field_name, f.field_value
                FROM fields f
                JOIN documents d ON f.document_id = d.id
                WHERE f.field_name = ?
                ORDER BY d.extracted_at DESC
            """, (field_name,))
        return [dict(r) for r in c.fetchall()]

    def search_by_provider(self, provider_name: str) -> list[dict]:
        """Find all records index entries for a provider."""
        c = self.conn.cursor()
        c.execute("""
            SELECT * FROM index_entries
            WHERE provider LIKE ?
            ORDER BY date_of_service
        """, (f"%{provider_name}%",))
        return [dict(r) for r in c.fetchall()]

    def get_all_documents(self, form_type: str | None = None) -> list[dict]:
        """List all stored documents, optionally filtered by type."""
        c = self.conn.cursor()
        if form_type:
            c.execute("""
                SELECT id, file_name, form_type, classification_confidence,
                       processing_time_ms, extracted_at
                FROM documents WHERE form_type = ?
                ORDER BY extracted_at DESC
            """, (form_type,))
        else:
            c.execute("""
                SELECT id, file_name, form_type, classification_confidence,
                       processing_time_ms, extracted_at
                FROM documents ORDER BY extracted_at DESC
            """)
        return [dict(r) for r in c.fetchall()]

    def get_document(self, doc_id: int) -> dict | None:
        """Get a single document with all its fields."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = c.fetchone()
        if not row:
            return None
        doc = dict(row)
        c.execute("SELECT field_name, field_value, field_type FROM fields WHERE document_id = ?", (doc_id,))
        doc["fields"] = [dict(r) for r in c.fetchall()]
        c.execute("SELECT * FROM service_lines WHERE document_id = ? ORDER BY line_number", (doc_id,))
        doc["service_lines"] = [dict(r) for r in c.fetchall()]
        c.execute("SELECT icd10_code, position FROM diagnoses WHERE document_id = ? ORDER BY position", (doc_id,))
        doc["diagnoses"] = [dict(r) for r in c.fetchall()]
        return doc

    def get_stats(self) -> dict:
        """Get database summary statistics."""
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM documents")
        total_docs = c.fetchone()[0]
        c.execute("SELECT form_type, COUNT(*) FROM documents GROUP BY form_type ORDER BY COUNT(*) DESC")
        by_type = {r[0]: r[1] for r in c.fetchall()}
        c.execute("SELECT COUNT(DISTINCT icd10_code) FROM diagnoses")
        unique_dx = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM service_lines")
        total_lines = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM index_entries")
        total_index = c.fetchone()[0]
        return {
            "total_documents": total_docs,
            "by_form_type": by_type,
            "unique_diagnoses": unique_dx,
            "total_service_lines": total_lines,
            "total_index_entries": total_index,
        }

    def close(self):
        self.conn.close()
