"""
Quick structural inspection of PDF documents using PyMuPDF.

Goal: understand font sizes, page structure, and potential headers
BEFORE committing to a chunking/extraction strategy.
"""

from pathlib import Path
from collections import Counter
import fitz  # PyMuPDF


RAW_DIR = Path("data/raw")


def inspect_pdf(pdf_path: Path, sample_pages: int = 5) -> None:
    """Print structural diagnostics for a single PDF."""
    doc = fitz.open(pdf_path)

    print(f"\n{'='*70}")
    print(f"FILE: {pdf_path.name}")
    print(f"{'='*70}")
    print(f"Total pages: {len(doc)}")

    font_sizes = Counter()
    font_names = Counter()

    # Sample first N pages + a few from the middle for representativeness
    pages_to_check = list(range(min(sample_pages, len(doc))))

    for page_num in pages_to_check:
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    font_sizes[size] += 1
                    font_names[span["font"]] += 1

    print(f"\n--- Font size distribution (first {len(pages_to_check)} pages) ---")
    for size, count in font_sizes.most_common(10):
        print(f"  size={size:>5} | count={count}")

    print(f"\n--- Font name distribution ---")
    for name, count in font_names.most_common(5):
        print(f"  {name:<30} count={count}")

    # Sample raw text from page 1 to eyeball structure
    print(f"\n--- Raw text sample (page 1, first 800 chars) ---")
    print(doc[0].get_text()[:800])

    # Check for embedded tables heuristically (very rough: look for many short lines with numbers)
    page1_text = doc[0].get_text()
    lines = page1_text.split("\n")
    numeric_lines = sum(1 for l in lines if any(c.isdigit() for c in l) and len(l.strip()) < 40)
    print(f"\n--- Heuristic table signal (page 1) ---")
    print(f"Short numeric-heavy lines: {numeric_lines} / {len(lines)} total lines")

    doc.close()


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR.resolve()}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}")

    for pdf_path in pdf_files:
        inspect_pdf(pdf_path)


if __name__ == "__main__":
    main()