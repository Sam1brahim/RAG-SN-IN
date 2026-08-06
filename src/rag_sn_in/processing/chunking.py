from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(
    r"E:\Project RAG-SN-IN\data\processed\text"
)

CHUNKS_DIR = Path(
    r"E:\Project RAG-SN-IN\data\processed\chunks"
)

FULL_DOCUMENT_SUBSTRING = "_full_document"

MODEL_NAME = "google/embeddinggemma-300m"
MAX_TOKENS = 256

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

def count_tokens(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )

# ============================================================
# Regular expressions
# ============================================================

CHAPTER_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"CHAPITRE\s+"
    r"([0-9]+(?:\.[0-9]+)*)"
    r"(?:\s*[-–—:.)]?\s*(.*))?"
    r"\s*$",
    re.IGNORECASE
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6})\s+(.+?)\s*$"
)

# ------------------------------------------------------------
# NOTE: This pattern must match whatever page marker your
# PDF -> Markdown extraction step inserts into the text.
# Adjust it to your actual format, e.g.:
#   <!-- page: 42 -->
#   [[PAGE 42]]
#   \f (form feed) etc.
# ------------------------------------------------------------
PAGE_MARKER_PATTERN = re.compile(
    r"<!--\s*page:\s*([0-9]+)\s*-->",
    re.IGNORECASE
)

# ============================================================
# Real start-of-content anchor
# ============================================================

REAL_CHAPTER1_START_PATTERN = re.compile(
    r"CHAPITRE\s+1\b[^\n]*\n"
    r"(?:\s*\n)*"
    r"(?:#{1,6}\s*)?"
    r"1\.1\s+INTRODUCTION\b[^\n]*\n"
    r"(?:\s*\n)*"
    r"[^\n]*Conformément\s+à\s+l['’]article\s+2111-9\s+du\s+code\s+des\s+transports",
    re.IGNORECASE
)

CHAPTER1_LINE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?CHAPITRE\s+1\b",
    re.IGNORECASE
)

INTRO_LINE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?1\.1\s+INTRODUCTION\b",
    re.IGNORECASE
)

BOILERPLATE_PATTERN = re.compile(
    r"Conform[ée]ment\s+à\s+l['’]article\s+2111-9\s+du\s+code\s+des\s+transports",
    re.IGNORECASE
)

def find_real_content_start(text: str) -> int:
    """
    Finds the character offset where the *real* Chapter 1 begins.
    """

    lines = text.splitlines(keepends=True)

    offsets = []
    running_offset = 0
    for line in lines:
        offsets.append(running_offset)
        running_offset += len(line)

    n = len(lines)

    LOOKAHEAD_INTRO = 15
    LOOKAHEAD_BOILERPLATE = 15

    candidates = []

    for i in range(n):
        if not CHAPTER1_LINE_PATTERN.match(lines[i]):
            continue

        intro_index = None
        for j in range(i + 1, min(i + 1 + LOOKAHEAD_INTRO, n)):
            if INTRO_LINE_PATTERN.match(lines[j]):
                intro_index = j
                break

        if intro_index is None:
            continue

        boilerplate_found = False
        for k in range(
            intro_index + 1,
            min(intro_index + 1 + LOOKAHEAD_BOILERPLATE, n)
        ):
            if BOILERPLATE_PATTERN.search(lines[k]):
                boilerplate_found = True
                break

        if not boilerplate_found:
            continue

        candidates.append(i)

    if not candidates:
        tqdm.write(
            "WARNING: Could not find the real Chapter 1 anchor. "
            "Falling back to parsing the entire document."
        )
        return 0

    if len(candidates) > 1:
        tqdm.write(
            f"NOTE: Found {len(candidates)} candidate anchors at "
            f"lines {candidates}. Using the FIRST one (line "
            f"{candidates[0]})."
        )

    first_line = candidates[0]

    tqdm.write(
        f"Real Chapter 1 content anchor found at line {first_line} "
        f"(character offset {offsets[first_line]}). "
        f"Discarding everything before it."
    )

    preview_lines = lines[first_line:first_line + 8]
    tqdm.write("----- Anchor preview -----")
    tqdm.write("".join(preview_lines))
    tqdm.write("---------------------------")

    return offsets[first_line]

# ============================================================
# Page tracking
# ============================================================

def strip_page_markers_with_map(
    text: str
) -> tuple[str, list[tuple[int, int]]]:
    """
    Removes page markers from the text and builds a mapping from
    character offsets (in the cleaned text) to page numbers.

    Returns:
        cleaned_text: text with page markers removed
        offset_page_map: list of (offset_in_cleaned_text, page_number)
                         sorted by offset, indicating the page that
                         starts at/after that offset.
    """

    cleaned_parts: list[str] = []
    offset_page_map: list[tuple[int, int]] = []

    current_page = 1
    last_end = 0
    cleaned_len = 0

    for match in PAGE_MARKER_PATTERN.finditer(text):
        # Text before this marker belongs to `current_page`.
        segment = text[last_end:match.start()]
        cleaned_parts.append(segment)
        cleaned_len += len(segment)

        current_page = int(match.group(1))
        offset_page_map.append((cleaned_len, current_page))

        last_end = match.end()

    # Remaining tail after the last marker.
    tail = text[last_end:]
    cleaned_parts.append(tail)

    cleaned_text = "".join(cleaned_parts)

    # Ensure there's always at least one entry so lookups never fail.
    if not offset_page_map:
        offset_page_map.append((0, current_page))

    return cleaned_text, offset_page_map

def page_at_offset(
    offset: int,
    offset_page_map: list[tuple[int, int]]
) -> int:
    """
    Returns the page number active at the given character offset,
    based on the offset -> page map built by
    strip_page_markers_with_map.
    """

    page = offset_page_map[0][1]

    for marker_offset, marker_page in offset_page_map:
        if marker_offset <= offset:
            page = marker_page
        else:
            break

    return page

# ============================================================
# Data structures
# ============================================================

@dataclass
class Section:
    title: str
    level: int
    content: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    # Character offsets (in the page-marker-stripped text) of each
    # content line, parallel to `content`.
    content_offsets: list[int] = field(default_factory=list)
    # Offset of the heading line itself (for the section title).
    heading_offset: int = 0

@dataclass
class Chunk:
    text: str
    token_count: int
    section_path: list[str]
    source_file: str
    chapter_number: str | None = None
    page_start: int | None = None
    page_end: int | None = None

# ============================================================
# Chapter handling
# ============================================================

def parse_chapter_heading(
    line: str
) -> tuple[str | None, str | None]:
    match = CHAPTER_PATTERN.match(line)

    if not match:
        return None, None

    number = match.group(1)
    suffix = (match.group(2) or "").strip()

    title = f"CHAPITRE {number}"

    if suffix:
        title += f" — {suffix}"

    return number, title

def is_chapter_heading(line: str) -> bool:
    return CHAPTER_PATTERN.match(line) is not None

# ============================================================
# Markdown parser
# ============================================================

def parse_markdown(text: str) -> list[Section]:
    chapters: list[Section] = []

    current_chapter: Section | None = None
    current_chapter_number: str | None = None

    section_stack: list[Section] = []

    offset = 0

    for raw_line in text.splitlines(keepends=True):
        line_offset = offset
        offset += len(raw_line)

        line = raw_line.rstrip("\n").rstrip()

        chapter_number, chapter_title = parse_chapter_heading(line)

        if chapter_number is not None:
            if (
                current_chapter is not None
                and chapter_number == current_chapter_number
            ):
                continue

            current_chapter = Section(
                title=f"CHAPITRE {chapter_number}",
                level=1,
                heading_offset=line_offset
            )

            chapters.append(current_chapter)

            current_chapter_number = chapter_number
            section_stack.clear()

            continue

        if current_chapter is None:
            continue

        heading_match = MARKDOWN_HEADING_PATTERN.match(line)

        if heading_match:
            hashes, title = heading_match.groups()

            if is_chapter_heading(line):
                continue

            level = len(hashes) + 1

            section = Section(
                title=title.strip(),
                level=level,
                heading_offset=line_offset
            )

            while (
                section_stack
                and section_stack[-1].level >= level
            ):
                section_stack.pop()

            if section_stack:
                section_stack[-1].children.append(section)
            else:
                current_chapter.children.append(section)

            section_stack.append(section)

            continue

        if section_stack:
            section_stack[-1].content.append(line)
            section_stack[-1].content_offsets.append(line_offset)
        else:
            current_chapter.content.append(line)
            current_chapter.content_offsets.append(line_offset)

    return chapters

# ============================================================
# Section formatting
# ============================================================

def clean_lines(lines: list[str]) -> str:
    result: list[str] = []
    previous_blank = False

    for line in lines:
        line = line.rstrip()

        if not line.strip():
            if not previous_blank:
                result.append("")
            previous_blank = True
        else:
            result.append(line)
            previous_blank = False

    return "\n".join(result).strip()

def format_direct_content(section: Section) -> str:
    parts = [section.title]

    content = clean_lines(section.content)

    if content:
        parts.append(content)

    return "\n\n".join(parts)

def section_offset_range(section: Section) -> tuple[int, int]:
    """
    Returns (min_offset, max_offset) covering the heading and all
    direct content lines of this section (not children).
    """

    offsets = [section.heading_offset] + section.content_offsets

    if not offsets:
        offsets = [section.heading_offset]

    return min(offsets), max(offsets)

# ============================================================
# Chunk creation
# ============================================================

def create_chunk(
    parts: list[str],
    section_path: list[str],
    source_file: str,
    chapter_number: str | None,
    page_start: int | None,
    page_end: int | None
) -> Chunk | None:
    text = "\n\n".join(
        part.strip()
        for part in parts
        if part.strip()
    ).strip()

    if not text:
        return None

    return Chunk(
        text=text,
        token_count=count_tokens(text),
        section_path=section_path.copy(),
        source_file=source_file,
        chapter_number=chapter_number,
        page_start=page_start,
        page_end=page_end
    )

def split_text_by_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]

def add_content_to_chunks(
    text: str,
    section_path: list[str],
    chunks: list[Chunk],
    source_file: str,
    chapter_number: str | None,
    page_start: int | None,
    page_end: int | None
) -> None:
    """
    NOTE: page_start/page_end here represent the page range covering
    the *entire* section's direct content. When a section's content
    must be split into multiple chunks (because it exceeds
    MAX_TOKENS), all resulting chunks are conservatively tagged with
    this same page range, since finer-grained per-paragraph page
    tracking would require mapping each paragraph back to its
    original offsets individually.
    """

    if not text.strip():
        return

    token_count = count_tokens(text)

    if token_count <= MAX_TOKENS:
        chunk = create_chunk(
            parts=[text],
            section_path=section_path,
            source_file=source_file,
            chapter_number=chapter_number,
            page_start=page_start,
            page_end=page_end
        )

        if chunk:
            chunks.append(chunk)

        return

    blocks = split_text_by_paragraphs(text)

    current_blocks: list[str] = []
    current_tokens = 0

    def flush(parts: list[str]) -> None:
        chunk = create_chunk(
            parts=parts,
            section_path=section_path,
            source_file=source_file,
            chapter_number=chapter_number,
            page_start=page_start,
            page_end=page_end
        )
        if chunk:
            chunks.append(chunk)

    for block in blocks:
        block_tokens = count_tokens(block)

        if block_tokens > MAX_TOKENS:
            if current_blocks:
                flush(current_blocks)
                current_blocks = []
                current_tokens = 0

            line_blocks = [
                line.strip()
                for line in block.splitlines()
                if line.strip()
            ]

            line_buffer: list[str] = []
            line_tokens = 0

            for line in line_blocks:
                line_count = count_tokens(line)

                if (
                    line_buffer
                    and line_tokens + line_count > MAX_TOKENS
                ):
                    flush(["\n".join(line_buffer)])
                    line_buffer = []
                    line_tokens = 0

                line_buffer.append(line)
                line_tokens += line_count

            if line_buffer:
                flush(["\n".join(line_buffer)])

            continue

        if (
            current_blocks
            and current_tokens + block_tokens > MAX_TOKENS
        ):
            flush(current_blocks)
            current_blocks = []
            current_tokens = 0

        current_blocks.append(block)
        current_tokens += block_tokens

    if current_blocks:
        flush(current_blocks)

def process_section(
    section: Section,
    parent_headers: list[str],
    chunks: list[Chunk],
    source_file: str,
    chapter_number: str | None,
    offset_page_map: list[tuple[int, int]]
) -> None:
    current_path = parent_headers + [section.title]

    direct_text = format_direct_content(section)

    if direct_text.strip():
        min_offset, max_offset = section_offset_range(section)
        page_start = page_at_offset(min_offset, offset_page_map)
        page_end = page_at_offset(max_offset, offset_page_map)

        add_content_to_chunks(
            text=direct_text,
            section_path=current_path,
            chunks=chunks,
            source_file=source_file,
            chapter_number=chapter_number,
            page_start=page_start,
            page_end=page_end
        )

    for child in section.children:
        process_section(
            section=child,
            parent_headers=current_path,
            chunks=chunks,
            source_file=source_file,
            chapter_number=chapter_number,
            offset_page_map=offset_page_map
        )

# ============================================================
# File discovery
# ============================================================

def find_full_document_files(root: Path) -> list[Path]:
    """
    Recursively finds all markdown files whose filename contains
    the FULL_DOCUMENT_SUBSTRING marker.
    """

    return sorted(
        p for p in root.rglob("*.md")
        if FULL_DOCUMENT_SUBSTRING in p.stem
    )

# ============================================================
# Document ID derivation
# ============================================================

def derive_document_id(path: Path) -> str:
    """
    Derives a stable document_id from the file/folder name.

    Adjust this logic to match your naming convention
    (e.g. "drr_001_full_document.md" -> "drr_001").
    """

    name = path.parent.name

    name = re.sub(
        re.escape(FULL_DOCUMENT_SUBSTRING),
        "",
        name,
        flags=re.IGNORECASE
    )

    return name.strip("_- ").lower() or path.stem

# ============================================================
# File processing
# ============================================================

def chunk_markdown_file(path: Path) -> list[Chunk]:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    start_offset = find_real_content_start(text)
    text = text[start_offset:]

    # Extract page markers BEFORE parsing sections, so section
    # offsets line up with the cleaned (marker-free) text.
    text, offset_page_map = strip_page_markers_with_map(text)

    chapters = parse_markdown(text)
    chunks: list[Chunk] = []

    for chapter in chapters:
        chapter_number_match = re.search(
            r"CHAPITRE\s+([0-9]+(?:\.[0-9]+)*)",
            chapter.title,
            re.IGNORECASE
        )

        chapter_number = (
            chapter_number_match.group(1)
            if chapter_number_match
            else None
        )

        process_section(
            section=chapter,
            parent_headers=[],
            chunks=chunks,
            source_file=path.name,
            chapter_number=chapter_number,
            offset_page_map=offset_page_map
        )

    return chunks

def process_single_file(path: Path) -> tuple[str, int]:
    """
    Processes a single full-document markdown file, writing its
    chunks to a jsonl file named after the file's parent folder,
    saved under CHUNKS_DIR.

    Returns:
        (folder_name, number_of_chunks)
    """

    folder_name = path.parent.name
    document_id = derive_document_id(path)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = CHUNKS_DIR / f"{folder_name}.jsonl"

    tqdm.write(f"Processing: {path}")

    chunks = chunk_markdown_file(path)

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for index, chunk in enumerate(
            tqdm(
                chunks,
                desc=f"Writing {folder_name}",
                leave=False
            )
        ):
            record = {
                "chunk_id": f"{document_id}_chunk_{index:04d}",
                "document_id": document_id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "section_path": chunk.section_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    tqdm.write(
        f"  -> {len(chunks)} chunks written to {output_file}\n"
    )

    return folder_name, len(chunks)

def process_document() -> None:
    files = find_full_document_files(DATA_DIR)

    if not files:
        raise FileNotFoundError(
            f"No files containing '{FULL_DOCUMENT_SUBSTRING}' "
            f"found under: {DATA_DIR}"
        )

    print(f"Found {len(files)} full-document file(s) to process.")
    print()

    results: list[tuple[str, int]] = []

    for path in tqdm(files, desc="Documents"):
        folder_name, chunk_count = process_single_file(path)
        results.append((folder_name, chunk_count))

    print()
    print("Summary:")
    for folder_name, chunk_count in results:
        print(f"  {folder_name}: {chunk_count} chunks")

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    process_document()