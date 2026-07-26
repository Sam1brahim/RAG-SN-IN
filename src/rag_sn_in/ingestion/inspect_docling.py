# src/rag_sn_in/ingestion/inspect_docling_text.py
"""
Extract and save Docling's text output per page, so we can manually judge
reading order, heading detection, header/footer bleed, and hyphenation issues.
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
from docling_core.types.doc import DocItemLabel
import logging

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TEXT_DIR = PROCESSED_DIR / "text"

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

    out_dir = TEXT_DIR / test_pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Full document markdown (reading-order as Docling sees it) ---
    full_md = doc.export_to_markdown()
    (out_dir / "_full_document.md").write_text(full_md, encoding="utf-8")

    # --- 2) Per-page text, reconstructed from doc items + their provenance ---
    # group text-like items by page number
    pages: dict[int, list[tuple[str, str]]] = {}  # page_no -> [(label, text)]

    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue

        label = str(getattr(item, "label", "text"))

        page_no = None
        prov = getattr(item, "prov", None)
        if prov:
            try:
                page_no = prov[0].page_no
            except Exception:
                pass

        if page_no is None:
            continue

        pages.setdefault(page_no, []).append((label, text))

    print(f"Pages with text items: {len(pages)}")

    # --- 3) Save each page as its own .md file, labels shown for inspection ---
    index_lines = ["# Text extraction index", ""]

    for page_no in sorted(pages.keys()):
        items = pages[page_no]
        lines = []
        n_headings = 0
        for label, text in items:
            if label == str(DocItemLabel.SECTION_HEADER) or "heading" in label.lower() or "title" in label.lower():
                n_headings += 1
                lines.append(f"## [{label}] {text}")
            else:
                lines.append(f"[{label}] {text}")

        page_text = "\n\n".join(lines)
        n_chars = len(page_text)

        fname = f"page_{page_no:04d}.md"
        (out_dir / fname).write_text(page_text, encoding="utf-8")

        index_lines.append(
            f"- **Page {page_no:04d}** | {n_chars} chars | {len(items)} items | {n_headings} headings -> `{fname}`"
        )

    index_path = out_dir / "_index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"\nSaved per-page text to: {out_dir}")
    print(f"Full document markdown: {out_dir / '_full_document.md'}")
    print(f"Index file: {index_path}")

    # quick console summary
    print("\n--- Quick summary (first 20 pages) ---")
    for line in index_lines[2:22]:
        print(line)


if __name__ == "__main__":
    main()