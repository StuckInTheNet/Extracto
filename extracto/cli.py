"""Extracto CLI - local form extraction tool."""

from __future__ import annotations

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(prog="extracto", description="Intelligent document extraction for medical and insurance forms")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_extract = sub.add_parser("extract", help="Extract structured data from PDF(s)")
    p_extract.add_argument("input", help="PDF file or directory")
    p_extract.add_argument("--out", default=None, help="Output directory (default: outputs/run_<timestamp>)")
    p_extract.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for flagging")
    p_extract.add_argument("--overlays", action="store_true", help="Generate visual overlay images for QA")
    p_extract.add_argument("--html", action="store_true", help="Generate HTML review page")
    p_extract.add_argument("--llm", action="store_true", help="Use LLM to correct low-confidence fields")
    p_extract.add_argument("--llm-model", default=None, help="OpenAI model name (default: gpt-4o-mini)")

    # --- split ---
    p_split = sub.add_parser("split", help="Split a multi-form PDF into individual documents")
    p_split.add_argument("pdf", help="Input PDF path")
    p_split.add_argument("--out", default="outputs/splits", help="Output directory")
    p_split.add_argument("--sim-threshold", type=float, default=0.3, help="Header Jaccard similarity threshold")

    # --- pipeline ---
    p_pipe = sub.add_parser("pipeline", help="Full pipeline: split + extract + review")
    p_pipe.add_argument("pdf", help="Input multi-form PDF")
    p_pipe.add_argument("--out", default="outputs/full_runs", help="Output root directory")
    p_pipe.add_argument("--conf-threshold", type=float, default=0.6)
    p_pipe.add_argument("--no-overlays", action="store_true")
    p_pipe.add_argument("--no-html", action="store_true")
    p_pipe.add_argument("--llm", action="store_true")
    p_pipe.add_argument("--llm-model", default=None)

    # --- overlay ---
    p_overlay = sub.add_parser("overlay", help="Generate debug overlay images for a PDF")
    p_overlay.add_argument("pdf", help="Input PDF path")
    p_overlay.add_argument("--out", default="outputs/debug", help="Output directory")
    p_overlay.add_argument("--dpi", type=int, default=300)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate extraction accuracy against ground truth")
    p_eval.add_argument("manifest", help="Path to dataset manifest.json")
    p_eval.add_argument("--out", default=None, help="Output directory for per-file results (optional)")
    p_eval.add_argument("--baseline", default=None, help="Path to previous eval_summary.json for comparison")
    p_eval.add_argument("--save-baseline", default=None, help="Save results as baseline to this path")
    p_eval.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")

    # --- auto ---
    p_auto = sub.add_parser("auto", help="Auto-classify and extract from mixed PDFs")
    p_auto.add_argument("input", help="PDF file or directory of mixed PDFs")
    p_auto.add_argument("--out", default=None, help="Output directory for per-file results")
    p_auto.add_argument("--db", default=None, help="SQLite database path to store results (e.g., extracto.db)")
    p_auto.add_argument("--json", action="store_true", help="Output raw JSON instead of summary")

    # --- index ---
    p_index = sub.add_parser("index", help="Index a medical records bundle by provider and date of service")
    p_index.add_argument("pdf", help="Multi-provider records bundle PDF")
    p_index.add_argument("--out", default=None, help="Output directory for index files (JSON, CSV, HTML)")
    p_index.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")

    args = parser.parse_args()

    if args.command == "extract":
        from extracto.pipeline.runner import process_input

        out_dir = args.out or f"outputs/run_{int(time.time())}"
        res = process_input(args.input, out_dir, conf_threshold=args.conf_threshold, overlays=args.overlays, llm=args.llm, llm_model=args.llm_model)
        if args.html:
            try:
                from extracto.pipeline.review import make_review_html

                index = make_review_html(out_dir)
                res["html"] = index
            except Exception as e:
                res["html_error"] = str(e)
        print(json.dumps(res, indent=2))

    elif args.command == "split":
        from extracto.splitting.splitter import split_pdf

        out = split_pdf(args.pdf, args.out, sim_threshold=args.sim_threshold)
        print(json.dumps(out, indent=2))

    elif args.command == "pipeline":
        from extracto.pipeline.runner import run_full

        out = run_full(args.pdf, args.out, conf_threshold=args.conf_threshold, overlays=not args.no_overlays, html=not args.no_html, llm=args.llm, llm_model=args.llm_model)
        print(json.dumps(out, indent=2))

    elif args.command == "overlay":
        from extracto.detection.overlay import draw_overlay

        paths = draw_overlay(args.pdf, args.out, dpi=args.dpi)
        print(json.dumps({"written": paths}, indent=2))

    elif args.command == "eval":
        from extracto.evaluation.evaluator import compare_baselines, evaluate_manifest, format_report

        out_dir = args.out or f"outputs/eval_{int(time.time())}"
        summary = evaluate_manifest(args.manifest, out_dir=out_dir)

        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(format_report(summary))
            if args.baseline:
                print()
                print(compare_baselines(summary, args.baseline))

        if args.save_baseline:
            from pathlib import Path

            Path(args.save_baseline).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_baseline).write_text(json.dumps(summary.to_dict(), indent=2))
            print(f"\nBaseline saved to {args.save_baseline}")

    elif args.command == "auto":
        from extracto.pipeline.auto import auto_extract_batch

        out_dir = args.out or f"outputs/auto_{int(time.time())}"
        result = auto_extract_batch(args.input, out_dir=out_dir)
        s = result["summary"]

        # Store in database if --db specified
        if args.db:
            from extracto.storage.db import ExtractoDB
            db = ExtractoDB(args.db)
            stored = 0
            for r in result["results"]:
                db.store_extraction(
                    r["file"], r["classified_type"], r["extraction"],
                    confidence=r["classification_confidence"],
                    processing_time_ms=r["processing_time_ms"],
                )
                stored += 1
            stats = db.get_stats()
            db.close()

        if args.json:
            print(json.dumps(s, indent=2))
        else:
            print(f"{'=' * 60}")
            print(f"EXTRACTO AUTO-EXTRACTION")
            print(f"{'=' * 60}")
            print(f"Total files processed: {s['total_files']}")
            print(f"Total time: {s['total_time_ms']:.0f}ms ({s['avg_time_ms']:.1f}ms/form)")
            print(f"Errors: {s['errors']}")
            print()
            print(f"Classification breakdown:")
            for ft, count in sorted(s["classification_breakdown"].items()):
                print(f"  {ft:<20} {count:>5} forms")
            print()
            print(f"Results written to {out_dir}/")
            if args.db:
                print(f"Database: {args.db} ({stats['total_documents']} documents, {stats['unique_diagnoses']} unique diagnoses, {stats['total_service_lines']} service lines)")


    elif args.command == "index":
        from extracto.pipeline.indexer import build_index, format_index_report

        out_dir = args.out or f"outputs/index_{int(time.time())}"
        index = build_index(args.pdf, out_dir=out_dir)

        if args.json:
            print(json.dumps(index, indent=2))
        else:
            print(format_index_report(index))
            print(f"Full index written to {out_dir}/")


if __name__ == "__main__":
    main()
