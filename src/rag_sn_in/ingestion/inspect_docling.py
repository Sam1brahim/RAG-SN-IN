"""
Extract and save Docling's text output per page, so we can manually judge
reading order, heading detection, header/footer bleed, and hyphenation issues.

Now also captures tables per page (previously silently skipped, since
TableItem doesn't expose plain `.text` like text-like items do).

The full-document markdown now also embeds page markers
(`<!-- page: N -->`) at every page transition, so downstream chunking
code can recover page_start/page_end per chunk.

Runs over ALL PDFs found in data/raw/, one output subfolder per document.
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
from docling_core.types.doc import DocItemLabel, TableItem
import logging

# Anchor to project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # adjust depth as needed
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEXT_DIR = PROCESSED_DIR / "text"

logging.basicConfig(level=logging.INFO)

# ============================================================
# Page marker format
# ============================================================
# This MUST match the PAGE_MARKER_PATTERN used by the chunking
# script downstream. Keep these in sync.

def page_marker(page_no: int) -> str:
    return f"<!-- page: {page_no} -->"


def get_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        print(f"CUDA available -> using GPU: {torch.cuda.get_device_name(0)}")
        return AcceleratorDevice.CUDA
    print("CUDA NOT available -> falling back to CPU")
    return AcceleratorDevice.CPU


def get_item_page_no(item) -> int | None:
    """
    Extracts the page number from an item's provenance, if available.
    """
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    try:
        return prov[0].page_no
    except Exception:
        return None


def item_to_markdown(item, label: str, text: str) -> str:
    """
    Formats a single item as a markdown fragment, matching the style
    used for the per-page files (headings get `##`, tables get a
    `[TABLE]` marker, everything else is left as plain text).
    """
    if label == str(DocItemLabel.TABLE):
        return f"### [TABLE]\n\n{text}"

    if (
        label == str(DocItemLabel.SECTION_HEADER)
        or "heading" in label.lower()
        or "title" in label.lower()
    ):
        return f"## [{label}] {text}"

    return text


def process_pdf(test_pdf: Path, converter: DocumentConverter) -> None:
    print(f"\n{'='*80}")
    print(f"Running Docling on: {test_pdf.name}")
    print(f"{'='*80}\n")

    start = time.perf_counter()
    result = converter.convert(str(test_pdf))
    elapsed = time.perf_counter() - start
    print(f"Conversion time: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    doc = result.document

    out_dir = TEXT_DIR / test_pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Per-page items AND full-document reconstruction, built
    # together from the same iteration so page markers and per-page
    # content stay perfectly consistent. ---

    pages: dict[int, list[tuple[str, str]]] = {}
    full_doc_blocks: list[str] = []

    n_tables_total = 0
    current_page_no: int | None = None
    first_item = True

    for item, _level in doc.iterate_items():
        # --- Handle tables explicitly, since they have no plain `.text` ---
        if isinstance(item, TableItem):
            page_no = get_item_page_no(item)
            if page_no is None:
                continue

            try:
                table_md = item.export_to_markdown(doc=doc)
            except Exception as e:
                table_md = f"[TABLE EXPORT FAILED: {e}]"

            n_tables_total += 1
            label = str(DocItemLabel.TABLE)
            pages.setdefault(page_no, []).append((label, table_md))

            if page_no != current_page_no:
                if not first_item:
                    full_doc_blocks.append(page_marker(page_no))
                current_page_no = page_no
            first_item = False

            full_doc_blocks.append(item_to_markdown(item, label, table_md))
            continue

        # --- Handle everything else that has plain text ---
        text = getattr(item, "text", None)
        if not text:
            continue

        label = str(getattr(item, "label", "text"))

        page_no = get_item_page_no(item)
        if page_no is None:
            continue

        pages.setdefault(page_no, []).append((label, text))

        if page_no != current_page_no:
            if not first_item:
                full_doc_blocks.append(page_marker(page_no))
            current_page_no = page_no
        first_item = False

        full_doc_blocks.append(item_to_markdown(item, label, text))

    print(f"Pages with items: {len(pages)}")
    print(f"Tables captured: {n_tables_total}")

    # --- 2) Full document markdown, with page markers inserted at
    # every page transition. The very first page also gets a marker
    # at the top so downstream code always has a page number for the
    # start of the document. ---

    if pages:
        first_page_no = min(pages.keys())
        full_md = "\n\n".join(
            [page_marker(first_page_no)] + full_doc_blocks
        )
    else:
        full_md = ""

    full_md_filename = f"{test_pdf.stem}_full_document.md"
    full_md_path = out_dir / full_md_filename
    full_md_path.write_text(full_md, encoding="utf-8")

    # --- 3) Save each page as its own .md file, labels shown for inspection ---
    index_lines = ["# Text extraction index", ""]

    for page_no in sorted(pages.keys()):
        items = pages[page_no]
        lines = []
        n_headings = 0
        n_tables = 0

        for label, text in items:
            if label == str(DocItemLabel.TABLE):
                n_tables += 1
                lines.append(f"### [TABLE]\n\n{text}")
            elif label == str(DocItemLabel.SECTION_HEADER) or "heading" in label.lower() or "title" in label.lower():
                n_headings += 1
                lines.append(f"## [{label}] {text}")
            else:
                lines.append(f"[{label}] {text}")

        page_text = "\n\n".join(lines)
        n_chars = len(page_text)

        fname = f"page_{page_no:04d}.md"
        (out_dir / fname).write_text(page_text, encoding="utf-8")

        index_lines.append(
            f"- **Page {page_no:04d}** | {n_chars} chars | {len(items)} items | "
            f"{n_headings} headings | {n_tables} tables -> `{fname}`"
        )

    index_path = out_dir / "_index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"Saved per-page text to: {out_dir}")
    print(f"Full document markdown (with page markers): {full_md_path}")
    print(f"Index file: {index_path}")

    # quick console summary
    print("\n--- Quick summary (first 20 pages) ---")
    for line in index_lines[2:22]:
        print(line)


def main():
    seen = {}
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            seen[p.resolve()] = p
    pdf_files = sorted(seen.values())

    if not pdf_files:
        print("No PDFs found in data/raw/")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}")

    device = "cpu"  # get_device()

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

    n_ok = 0
    n_failed = 0
    failed_files: list[str] = []

    overall_start = time.perf_counter()

    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path, converter)
            n_ok += 1
        except Exception as e:
            n_failed += 1
            failed_files.append(pdf_path.name)
            print(f"\n[FAILED] {pdf_path.name}: {e}\n")

    overall_elapsed = time.perf_counter() - overall_start

    print(f"\n{'='*80}")
    print(f"DONE. {n_ok} succeeded, {n_failed} failed, "
          f"total time {overall_elapsed:.1f}s ({overall_elapsed/60:.2f} min)")
    if failed_files:
        print(f"Failed files: {failed_files}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()