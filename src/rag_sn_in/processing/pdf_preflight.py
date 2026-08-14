"""
pdf_preflight.py — Normalize PDFs before Docling ingestion.

Detects 2-up scanned pages (two logical pages per PDF page) and splits
them down the middle so Docling gets standard portrait pages.

Usage:
    python pdf_preflight.py "rapport_securite_2018 EPSF.pdf"
    python pdf_preflight.py "rapport_securite_2018 EPSF.pdf" --workdir ./normalized
"""

import argparse
from pathlib import Path

import pymupdf  # PyMuPDF


def split_2up_pdf(src_path: str, dst_path: str) -> str:
    """Split landscape 2-up pages into two portrait pages each."""
    doc = pymupdf.open(src_path)
    out = pymupdf.open()
    split_count = 0

    for page in doc:
        r = page.rect
        if r.width > r.height:
            # 2-up page: emit left half, then right half
            for left in (True, False):
                new_page = out.new_page(width=r.width / 2, height=r.height)
                clip = (
                    pymupdf.Rect(0, 0, r.width / 2, r.height)
                    if left
                    else pymupdf.Rect(r.width / 2, 0, r.width, r.height)
                )
                new_page.show_pdf_page(new_page.rect, doc, page.number, clip=clip)
            split_count += 1
        else:
            new_page = out.new_page(width=r.width, height=r.height)
            new_page.show_pdf_page(new_page.rect, doc, page.number)

    out.save(dst_path)
    print(f"[split] {split_count}/{len(doc)} pages were 2-up -> {dst_path}")
    doc.close()
    out.close()
    return dst_path


def preflight_pdf(pdf_path: str, workdir: str) -> str:
    """
    Inspect a PDF and normalize it if needed.
    Returns the path Docling should consume (original or split version).
    """
    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)
    landscape = sum(1 for p in doc if p.rect.width > p.rect.height)
    has_text = any(p.get_text().strip() for p in doc)
    doc.close()

    print(f"[preflight] {n_pages} pages | {landscape} landscape | text layer: {has_text}")

    if landscape > n_pages * 0.3:  # >30% landscape -> treat as 2-up scan
        Path(workdir).mkdir(parents=True, exist_ok=True)
        dst = str(Path(workdir) / f"{Path(pdf_path).stem}_split.pdf")
        return split_2up_pdf(pdf_path, dst)

    print("[preflight] no normalization needed")
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight PDF for Docling: detect and split 2-up pages."
    )
    parser.add_argument("pdf", help="Path to the source PDF")
    parser.add_argument(
        "--workdir",
        default=r"E:\Project RAG-SN-IN\data\raw"
    )
    args = parser.parse_args()

    final_path = preflight_pdf(args.pdf, args.workdir)
    print(f"[done] Docling-ready file: {final_path}")


if __name__ == "__main__":
    main()