# src/rag_sn_in/ingestion/inspect_docling_tables.py
"""
Extract and save every table Docling detects in a PDF, so we can manually
judge extraction quality (real tables vs. false positives like ToC dot-leaders).
"""

import time
from pathlib import Path

import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
import logging

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = PROCESSED_DIR / "tables"

logging.basicConfig(level=logging.INFO) 

def get_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        print(f"CUDA available -> using GPU: {torch.cuda.get_device_name(0)}")
        return AcceleratorDevice.CUDA
    print("CUDA NOT available -> falling back to CPU")
    return AcceleratorDevice.CPU


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf")) + sorted(RAW_DIR.glob("*.PDF"))
    if not pdf_files:
        print("No PDFs found in data/raw/")
        return

    test_pdf = pdf_files[0]
    print(f"Running Docling on: {test_pdf.name}\n")

    device = get_device()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=device,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start = time.perf_counter()
    result = converter.convert(str(test_pdf))
    elapsed = time.perf_counter() - start
    print(f"Conversion time: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    doc = result.document
    tables = list(doc.tables) if hasattr(doc, "tables") else []
    print(f"Tables detected: {len(tables)}")

    if not tables:
        print("No tables found — nothing to extract.")
        return

    out_dir = TABLES_DIR / test_pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["# Table extraction index", ""]

    for i, table in enumerate(tables):
        try:
            md = table.export_to_markdown(doc)
        except Exception as e:
            md = f"(failed to export: {e})"

        # try to get page number / provenance info if available
        page_no = None
        try:
            if table.prov:
                page_no = table.prov[0].page_no
        except Exception:
            pass

        n_rows = md.count("\n") + 1
        n_chars = len(md)

        fname = f"table_{i:03d}_page{page_no if page_no else 'NA'}.md"
        (out_dir / fname).write_text(md, encoding="utf-8")

        # crude heuristic: flag likely-ToC tables (dot leaders, single real column)
        is_suspect_toc = ".........." in md or "....." in md

        flag = "⚠️ TOC-like" if is_suspect_toc else ""
        index_lines.append(
            f"- **Table {i:03d}** | page {page_no} | {n_chars} chars | {n_rows} lines | {flag}"
        )
        index_lines.append(f"  -> `{fname}`")

    index_path = out_dir / "_index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"\nSaved {len(tables)} tables to: {out_dir}")
    print(f"Index file: {index_path}")

    # print a quick console summary too
    print("\n--- Quick summary ---")
    for line in index_lines[2:]:
        if line.startswith("- **"):
            print(line)


if __name__ == "__main__":
    main()