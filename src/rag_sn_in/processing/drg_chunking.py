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

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data" / "processed" / "text"
CHUNKS_DIR = ROOT_DIR / "data" / "processed" / "chunks"

FULL_DOCUMENT_SUBSTRING = "_full_document"

MODEL_NAME = "google/embeddinggemma-300m"
MAX_TOKENS = 512

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)


def count_tokens(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


# ============================================================
# Regular expressions & Markers
# ============================================================

PAGE_MARKER_PATTERN = re.compile(
    r"<!--\s*page:\s*([0-9]+)\s*-->",
    re.IGNORECASE,
)

HEADING_LINE_PATTERN = re.compile(
    r"^\s*(#{1,6})\s+(.+?)\s*$"
)

TABLE_HEADER_SEP_PATTERN = re.compile(
    r"^\s*\|[-|\s:]+\|\s*$"
)

# Anchor regex to discard cover page and table of contents
REAL_START_PATTERN = re.compile(
    r"(?:#{1,6}\s*(?:\[section_header\]\s*)?)?(?:DÉFINITIONS\s+ET\s+PRINCIPALES\s+ABRÉVIATIONS|DEFINITIONS\s+ET\s+PRINCIPALES\s+ABREVIATIONS|CHAPITRE\s+1\b)",
    re.IGNORECASE,
)


def find_real_content_start(text: str) -> int:
    """
    Finds the character offset where the real DRG content begins
    (e.g., Définitions et principales abréviations), skipping table of contents and cover.
    """
    lines = text.splitlines(keepends=True)
    running_offset = 0

    for line in lines:
        stripped = line.strip()
        # Avoid matching TOC table lines
        if "..." in stripped:
            running_offset += len(line)
            continue

        if re.search(r"^\s*(?:#{1,6}\s*)?(?:\[section_header\]\s*)?D[éeE]finitions\s+et\s+principales\s+abr[éeE]viations\b", stripped, re.IGNORECASE):
            return running_offset

        running_offset += len(line)

    # Fallback to search in text directly
    m = REAL_START_PATTERN.search(text)
    if m:
        return m.start()

    return 0


# ============================================================
# Page tracking
# ============================================================

def strip_page_markers_with_map(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Removes page markers from the text and builds a mapping from
    character offsets (in the cleaned text) to page numbers.
    """
    cleaned_parts: list[str] = []
    offset_page_map: list[tuple[int, int]] = []

    current_page = 1
    last_end = 0
    cleaned_len = 0

    for match in PAGE_MARKER_PATTERN.finditer(text):
        segment = text[last_end:match.start()]
        cleaned_parts.append(segment)
        cleaned_len += len(segment)

        current_page = int(match.group(1))
        offset_page_map.append((cleaned_len, current_page))

        last_end = match.end()

    tail = text[last_end:]
    cleaned_parts.append(tail)
    cleaned_text = "".join(cleaned_parts)

    if not offset_page_map:
        offset_page_map.append((0, current_page))

    return cleaned_text, offset_page_map


def page_at_offset(
    offset: int,
    offset_page_map: list[tuple[int, int]],
) -> int:
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
    content_offsets: list[int] = field(default_factory=list)
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
# Header Classification for DRG Structure
# ============================================================

KNOWN_CHAPTERS = {
    "1": "Chapitre 1 : Informations générales",
    "2": "Chapitre 2 : Gares et services",
    "3": "Chapitre 3 : Modalités d'accès au service",
    "4": "Chapitre 4 : Tarification",
}


def classify_header(raw_title: str) -> tuple[int | None, str, str | None]:
    """
    Classifies a header title to determine its logical level (1, 2, 3, etc.)
    and returns (level, normalized_title, chapter_number).
    Returns (None, raw_title, None) if the line is a table placeholder or ignored.
    """
    raw_title = raw_title.strip()
    raw_title = re.sub(r"^\[section_header\]\s*", "", raw_title, flags=re.IGNORECASE).strip()

    if not raw_title or raw_title.upper() == "[TABLE]":
        return None, raw_title, None

    # Standalone Top-Level Sections
    if re.match(r"^D[éeE]finitions\s+et\s+principales\s+abr[éeE]viations\b", raw_title, re.IGNORECASE):
        return 1, "Définitions et principales abréviations", "0"

    # Chapter pattern: 'Chapitre 1 : Informations générales', 'Chapitre 2 : Gares et services', etc.
    m_chap = re.match(r"^Chapitre\s+([0-9]+)\s*:\s*(.+)$", raw_title, re.IGNORECASE)
    if m_chap:
        num = m_chap.group(1)
        rest = m_chap.group(2).strip()
        return 1, f"Chapitre {num} : {rest}", num

    # Numbered subsections: '1.1. Objectifs...', '2.2.1.1. Mise à disposition...', '4.1.6.2. Charges...'
    m_sub = re.match(r"^([0-9]+(?:\.[0-9]+)+)\.?\s+(.+)$", raw_title)
    if m_sub:
        num = m_sub.group(1)
        rest = m_sub.group(2).strip()
        parts = num.split(".")
        level = len(parts) + 1  # 1.1 -> level 2, 2.2.1 -> level 3, 2.2.1.1 -> level 4
        chap_num = parts[0]
        return level, f"{num}. {rest}", chap_num

    # Lettered subheadings: 'A - Information collective...', 'B - Prestation transmanche', 'a) Construction...'
    m_letter = re.match(r"^([A-Z]|[a-z]\))\s*[-–—:]?\s*(.+)$", raw_title)
    if m_letter:
        return 4, raw_title, None

    # Bullet headers: '❖ Des prestations...', '· Nettoyage des zones...', '-Bouches à eau'
    m_bullet = re.match(r"^[❖·■•\–—\-]\s*(.+)$", raw_title)
    if m_bullet:
        return 4, m_bullet.group(1).strip(), None

    # Specific named subheadings without numbers (e.g., 'Principe de la tarification binomiale', 'Construction des tarifs binômes')
    return 3, raw_title, None


# ============================================================
# Markdown Parser
# ============================================================

def parse_markdown(text: str) -> list[Section]:
    root_sections: list[Section] = []
    section_stack: list[Section] = []

    offset = 0
    current_chapter_num: str | None = None

    for raw_line in text.splitlines(keepends=True):
        line_offset = offset
        offset += len(raw_line)

        line = raw_line.rstrip("\n").rstrip()

        heading_match = HEADING_LINE_PATTERN.match(line)
        if heading_match:
            hashes, raw_title = heading_match.groups()
            level, clean_title, chap_num = classify_header(raw_title)

            if level is None:
                # E.g. ### [TABLE], keep in content of current section
                if section_stack:
                    section_stack[-1].content.append(line)
                    section_stack[-1].content_offsets.append(line_offset)
                continue

            if chap_num is not None and chap_num != "0":
                current_chapter_num = chap_num

            # Contextually insert parent chapter if missing
            if level >= 2 and current_chapter_num in KNOWN_CHAPTERS:
                chap_title = KNOWN_CHAPTERS[current_chapter_num]
                if not root_sections or root_sections[-1].title != chap_title:
                    chap_sec = Section(
                        title=chap_title,
                        level=1,
                        heading_offset=line_offset,
                    )
                    root_sections.append(chap_sec)
                    section_stack = [chap_sec]

            section = Section(
                title=clean_title,
                level=level,
                heading_offset=line_offset,
            )

            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()

            if section_stack:
                section_stack[-1].children.append(section)
            else:
                root_sections.append(section)

            section_stack.append(section)
            continue

        if section_stack:
            section_stack[-1].content.append(line)
            section_stack[-1].content_offsets.append(line_offset)
        elif line.strip():
            # Content before any header
            if not root_sections:
                root_sections.append(
                    Section(title="DOCUMENT", level=1, heading_offset=line_offset)
                )
                section_stack.append(root_sections[0])
            section_stack[-1].content.append(line)
            section_stack[-1].content_offsets.append(line_offset)

    return root_sections


# ============================================================
# Section Formatting & Chunking
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
    content = clean_lines(section.content)
    # If section has no content of its own and only has children, don't emit an orphan title
    if not content:
        return ""
    return f"{section.title}\n\n{content}"


def section_offset_range(section: Section) -> tuple[int, int]:
    offsets = [section.heading_offset] + section.content_offsets
    if not offsets:
        offsets = [section.heading_offset]
    return min(offsets), max(offsets)


def create_chunk(
    parts: list[str],
    section_path: list[str],
    source_file: str,
    chapter_number: str | None,
    page_start: int | None,
    page_end: int | None,
) -> Chunk | None:
    text = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    if not text:
        return None

    return Chunk(
        text=text,
        token_count=count_tokens(text),
        section_path=section_path.copy(),
        source_file=source_file,
        chapter_number=chapter_number,
        page_start=page_start,
        page_end=page_end,
    )


def is_markdown_table_block(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].startswith("|") and lines[0].endswith("|"):
        if TABLE_HEADER_SEP_PATTERN.match(lines[1]) or re.search(r"\|[-:]+\|", lines[1]):
            return True
    return False


def split_table_block(table_block: str, max_tokens: int) -> list[str]:
    lines = [line.strip() for line in table_block.splitlines() if line.strip()]
    if len(lines) <= 2:
        return [table_block]

    header_line = lines[0]
    sep_line = lines[1]
    data_rows = lines[2:]

    chunks_tables: list[str] = []
    current_rows: list[str] = []

    for row in data_rows:
        test_table = "\n".join([header_line, sep_line] + current_rows + [row])
        if count_tokens(test_table) <= max_tokens:
            current_rows.append(row)
        else:
            if current_rows:
                chunks_tables.append("\n".join([header_line, sep_line] + current_rows))
                current_rows = []

            single_row_table = "\n".join([header_line, sep_line, row])
            if count_tokens(single_row_table) <= max_tokens:
                current_rows.append(row)
            else:
                chunks_tables.append(single_row_table)

    if current_rows:
        chunks_tables.append("\n".join([header_line, sep_line] + current_rows))

    return chunks_tables or [table_block]


def split_text_into_blocks(text: str) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in raw_blocks if b.strip()]


def add_content_to_chunks(
    text: str,
    section_path: list[str],
    chunks: list[Chunk],
    source_file: str,
    chapter_number: str | None,
    page_start: int | None,
    page_end: int | None,
) -> None:
    if not text.strip():
        return

    if count_tokens(text) <= MAX_TOKENS:
        chunk = create_chunk(
            parts=[text],
            section_path=section_path,
            source_file=source_file,
            chapter_number=chapter_number,
            page_start=page_start,
            page_end=page_end,
        )
        if chunk:
            chunks.append(chunk)
        return

    blocks = split_text_into_blocks(text)
    current_blocks: list[str] = []
    current_tokens = 0

    def flush(parts: list[str]) -> None:
        chunk = create_chunk(
            parts=parts,
            section_path=section_path,
            source_file=source_file,
            chapter_number=chapter_number,
            page_start=page_start,
            page_end=page_end,
        )
        if chunk:
            chunks.append(chunk)

    for block in blocks:
        if is_markdown_table_block(block):
            table_tokens = count_tokens(block)
            if table_tokens > MAX_TOKENS:
                if current_blocks:
                    flush(current_blocks)
                    current_blocks = []
                    current_tokens = 0
                sub_tables = split_table_block(block, MAX_TOKENS)
                for sub_t in sub_tables:
                    flush([sub_t])
                continue

        block_tokens = count_tokens(block)

        if block_tokens > MAX_TOKENS:
            if current_blocks:
                flush(current_blocks)
                current_blocks = []
                current_tokens = 0

            # Split paragraph by sentences or bullet points
            sentences = re.split(r"(?<=[.!?])\s+|\n+", block)
            sent_buffer: list[str] = []
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                test_str = "\n".join(sent_buffer + [sent]) if sent_buffer else sent
                if count_tokens(test_str) <= MAX_TOKENS:
                    sent_buffer.append(sent)
                else:
                    if sent_buffer:
                        flush(["\n".join(sent_buffer)])
                        sent_buffer = []
                    # If a single sentence exceeds MAX_TOKENS, split by words
                    if count_tokens(sent) > MAX_TOKENS:
                        words = sent.split()
                        w_buffer: list[str] = []
                        for w in words:
                            test_w = " ".join(w_buffer + [w]) if w_buffer else w
                            if count_tokens(test_w) <= MAX_TOKENS:
                                w_buffer.append(w)
                            else:
                                if w_buffer:
                                    flush([" ".join(w_buffer)])
                                    w_buffer = []
                                w_buffer.append(w)
                        if w_buffer:
                            flush([" ".join(w_buffer)])
                    else:
                        sent_buffer.append(sent)

            if sent_buffer:
                flush(["\n".join(sent_buffer)])

            continue

        test_parts = current_blocks + [block]
        if count_tokens("\n\n".join(test_parts)) <= MAX_TOKENS:
            current_blocks.append(block)
        else:
            if current_blocks:
                flush(current_blocks)
            current_blocks = [block]

    if current_blocks:
        flush(current_blocks)


def process_section(
    section: Section,
    parent_headers: list[str],
    chunks: list[Chunk],
    source_file: str,
    chapter_number: str | None,
    offset_page_map: list[tuple[int, int]],
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
            page_end=page_end,
        )

    for child in section.children:
        process_section(
            section=child,
            parent_headers=current_path,
            chunks=chunks,
            source_file=source_file,
            chapter_number=chapter_number,
            offset_page_map=offset_page_map,
        )


# ============================================================
# File Processing & Pipeline
# ============================================================

def derive_document_id(path: Path) -> str:
    name = path.parent.name
    name = re.sub(re.escape(FULL_DOCUMENT_SUBSTRING), "", name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip()).strip("_").lower()
    return cleaned or path.stem


def chunk_markdown_file(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    start_offset = find_real_content_start(text)
    text = text[start_offset:]

    text, offset_page_map = strip_page_markers_with_map(text)
    root_sections = parse_markdown(text)

    chunks: list[Chunk] = []

    for section in root_sections:
        m_chap = re.search(r"([0-9]+)", section.title)
        chap_num = m_chap.group(1) if m_chap else None

        process_section(
            section=section,
            parent_headers=[],
            chunks=chunks,
            source_file=path.name,
            chapter_number=chap_num,
            offset_page_map=offset_page_map,
        )

    return chunks


def process_single_file(path: Path, output_subfolder: str = "max token 512") -> tuple[str, int]:
    document_id = derive_document_id(path)

    target_dir = CHUNKS_DIR / output_subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    jsonl_output = target_dir / f"{document_id}.jsonl"
    md_output = target_dir / f"{document_id}.md"

    tqdm.write(f"Processing: {path}")
    chunks = chunk_markdown_file(path)

    # Write JSONL
    with jsonl_output.open("w", encoding="utf-8") as f_jsonl, md_output.open("w", encoding="utf-8") as f_md:
        for idx, chunk in enumerate(tqdm(chunks, desc=f"Writing {document_id}", leave=False)):
            chunk_id = f"{document_id}_chunk_{idx:04d}"
            record = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "section_path": chunk.section_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Markdown export for visual inspection
            f_md.write(f"## {chunk_id} (Tokens: {chunk.token_count}, Pages: {chunk.page_start}-{chunk.page_end})\n\n")
            f_md.write(f"**Path**: `{' > '.join(chunk.section_path)}`\n\n")
            f_md.write(f"{chunk.text}\n\n---\n\n")

    tqdm.write(f"  -> {len(chunks)} chunks written to {jsonl_output} and {md_output}\n")
    return document_id, len(chunks)


def run_drg_chunking() -> None:
    drg_files = sorted(
        p for p in DATA_DIR.rglob("*.md")
        if FULL_DOCUMENT_SUBSTRING in p.stem and "drg" in str(p).lower()
    )

    if not drg_files:
        raise FileNotFoundError(f"No DRG files containing '{FULL_DOCUMENT_SUBSTRING}' found under {DATA_DIR}")

    print(f"Found {len(drg_files)} document(s) to process with MAX_TOKENS={MAX_TOKENS}:")
    for p in drg_files:
        print(f" - {p}")
    print()

    results = []
    for path in drg_files:
        doc_id, count = process_single_file(path, output_subfolder="max token 512")
        results.append((doc_id, count))

    print("\nProcessing complete:")
    for doc_id, count in results:
        print(f"  {doc_id}: {count} chunks")


if __name__ == "__main__":
    run_drg_chunking()
