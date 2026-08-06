from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "data" / "eval" / "raw.jsonl"
REPORT_PATH = ROOT / "data" / "eval" / "validation_report.json"
CHUNK_SOURCES = [
    ROOT / "data" / "processed" / "chunks" / "max token 512",
    ROOT / "data" / "processed" / "chunks",
]

EXPECTED_KEYS = {"question", "evidence_quote", "gold_chunk_ids", "gold_chunk_roles", "turn1_context", "year_pinned"}
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SELF_NOTE_RE = re.compile(r"\b(je dois|je vais|corriger|hallucination|illisible|rejete|rejeter|mal citée)\b", re.I)
YEAR_PIN_TERMS_RE = re.compile(
    r"\b(20\d{2}|cette année|année prochaine|année dernière|an dernier|par rapport à l'an dernier|nouveau cette année|à partir de l'horaire)\b",
    re.I,
)


def nfc(s: Any) -> str:
    return unicodedata.normalize("NFC", "" if s is None else str(s))


def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", nfc(s).replace("\u00a0", " ")).strip()


def norm_for_compare(s: Any) -> str:
    s = norm_space(s).casefold()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    return s


def tokens(s: Any) -> list[str]:
    return WORD_RE.findall(norm_for_compare(s))


def token_set(s: Any) -> set[str]:
    return set(tokens(s))


def numbers(s: Any) -> set[str]:
    return {x.replace(",", ".") for x in NUMBER_RE.findall(nfc(s))}


def max_common_ngram(a: list[str], b: list[str], cap: int = 30) -> int:
    if not a or not b:
        return 0
    b_pos: dict[str, list[int]] = defaultdict(list)
    for i, tok in enumerate(b):
        b_pos[tok].append(i)
    best = 0
    for i, tok in enumerate(a):
        for j in b_pos.get(tok, []):
            k = 1
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k] and k < cap:
                k += 1
            if k > best:
                best = k
                if best >= cap:
                    return best
    return best


def load_chunks() -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    chunks: dict[str, dict[str, str]] = {}
    issues: list[dict[str, Any]] = []
    for source_dir in CHUNK_SOURCES:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        issues.append({"severity": "blocking", "check": "source_json", "file": str(path), "line": line_no, "message": str(e)})
                        continue
                    cid = obj.get("chunk_id")
                    text = obj.get("text")
                    if not cid or not isinstance(text, str):
                        issues.append({"severity": "major", "check": "source_schema", "file": str(path), "line": line_no, "message": "missing chunk_id/text"})
                        continue
                    chunks.setdefault(cid, {"text": text, "norm": norm_space(text), "file": str(path.relative_to(ROOT)), "doc": str(cid).split("_chunk_")[0]})
    return chunks, issues


def main() -> int:
    chunks, issues = load_chunks()
    rows: list[dict[str, Any]] = []
    blank_lines: list[int] = []

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                blank_lines.append(line_no)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append({"severity": "blocking", "check": "eval_json", "line": line_no, "message": str(e)})
                continue
            obj["_line"] = line_no
            rows.append(obj)

    per_doc = defaultdict(lambda: Counter())
    evidence_usage = defaultdict(list)
    answer_span_usage = defaultdict(list)
    question_norm_usage = defaultdict(list)
    q_tokens: dict[int, list[str]] = {}
    offenders_copy: list[dict[str, Any]] = []
    starter_counts = Counter()
    type_counts = Counter()

    for obj in rows:
        line = obj["_line"]
        doc = "unknown"
        ids = obj.get("gold_chunk_ids")
        if isinstance(ids, list) and ids and isinstance(ids[0], str):
            doc = ids[0].split("_chunk_")[0]
        per_doc[doc]["total"] += 1

        keys = set(obj.keys()) - {"_line"}
        if keys != EXPECTED_KEYS:
            issues.append({"severity": "blocking", "check": "schema_keys", "line": line, "message": f"keys={sorted(keys)} expected={sorted(EXPECTED_KEYS)}"})
        if not isinstance(obj.get("question"), str) or not norm_space(obj.get("question")):
            issues.append({"severity": "blocking", "check": "schema_question", "line": line, "message": "question must be a non-empty string"})
        if not isinstance(obj.get("evidence_quote"), str) or not norm_space(obj.get("evidence_quote")):
            issues.append({"severity": "blocking", "check": "schema_evidence", "line": line, "message": "evidence_quote must be a non-empty string"})
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            issues.append({"severity": "blocking", "check": "schema_gold_ids", "line": line, "message": "gold_chunk_ids must be a non-empty string list"})
            ids = []
        roles = obj.get("gold_chunk_roles")
        if not isinstance(roles, dict):
            issues.append({"severity": "blocking", "check": "schema_roles", "line": line, "message": "gold_chunk_roles must be an object"})
            roles = {}
        if set(roles.keys()) != set(ids):
            issues.append({"severity": "blocking", "check": "roles_match_ids", "line": line, "message": f"roles keys {sorted(roles.keys())} != ids {sorted(ids)}"})
        if obj.get("turn1_context") is not None and not isinstance(obj.get("turn1_context"), str):
            issues.append({"severity": "blocking", "check": "schema_turn1", "line": line, "message": "turn1_context must be null or string"})
        if not isinstance(obj.get("year_pinned"), bool):
            issues.append({"severity": "blocking", "check": "schema_year_pinned", "line": line, "message": "year_pinned must be boolean"})

        q = nfc(obj.get("question", ""))
        quote = nfc(obj.get("evidence_quote", ""))
        q_toks = tokens(q)
        q_tokens[line] = q_toks
        if q_toks:
            starter_counts[q_toks[0]] += 1
        question_norm_usage[norm_for_compare(q)].append(line)
        if len(q_toks) < 4:
            issues.append({"severity": "minor", "check": "question_too_short", "line": line, "message": f"question has {len(q_toks)} words"})
        if "[...]" in quote or "…" in quote:
            issues.append({"severity": "major", "check": "evidence_omission_marker", "line": line, "message": "evidence_quote contains an omission marker, so it is not a continuous verbatim quote"})
        q_words = len(tokens(quote))
        if q_words < 6:
            issues.append({"severity": "minor", "check": "evidence_too_short", "line": line, "message": f"evidence_quote has {q_words} words"})
        elif q_words > 90:
            issues.append({"severity": "minor", "check": "evidence_too_long", "line": line, "message": f"evidence_quote has {q_words} words"})

        cited_texts = []
        cited_token_sets = []
        for cid in ids:
            if cid not in chunks:
                issues.append({"severity": "blocking", "check": "unknown_chunk", "line": line, "message": f"{cid} not found in source chunks"})
                continue
            cited_texts.append(chunks[cid]["text"])
            cited_token_sets.append(token_set(chunks[cid]["text"]))
            if chunks[cid]["doc"] != doc:
                issues.append({"severity": "major", "check": "mixed_doc_citation", "line": line, "message": f"{cid} belongs to {chunks[cid]['doc']} but first id doc is {doc}"})

        quote_cmp = norm_space(quote)
        quote_found_cited = [cid for cid in ids if cid in chunks and quote_cmp and quote_cmp in chunks[cid]["norm"]]
        quote_found_any = [cid for cid, rec in chunks.items() if quote_cmp and quote_cmp in rec["norm"]]
        if quote and not quote_found_cited:
            sev = "blocking" if "[...]" not in quote else "major"
            issues.append({"severity": sev, "check": "evidence_not_verbatim", "line": line, "message": "evidence_quote is not a whitespace-normalized verbatim substring of any cited chunk"})
            if quote_found_any:
                issues.append({"severity": "major", "check": "evidence_in_uncited_chunk", "line": line, "message": f"quote appears in uncited chunk(s): {quote_found_any[:5]}"})
        evidence_usage[norm_for_compare(quote)].append(line)
        answer_span_usage[(norm_for_compare(quote), tuple(sorted(ids)))].append(line)

        for cid in ids:
            role = nfc(roles.get(cid, ""))
            if not norm_space(role):
                issues.append({"severity": "blocking", "check": "empty_role", "line": line, "message": f"empty role for {cid}"})
            if SELF_NOTE_RE.search(role):
                issues.append({"severity": "blocking", "check": "role_self_note", "line": line, "message": f"role for {cid} contains process/self-note text"})
            if len(tokens(role)) < 4:
                issues.append({"severity": "minor", "check": "role_too_thin", "line": line, "message": f"role for {cid} has fewer than 4 words"})

        if cited_texts:
            all_cited = "\n".join(cited_texts)
            cited_nums = numbers(all_cited)
            for num in sorted(numbers(q) - cited_nums):
                issues.append({"severity": "major", "check": "possible_invented_number", "line": line, "message": f"numeric token {num!r} appears in question but not in cited chunk text"})
            q_has_pin_term = bool(YEAR_PIN_TERMS_RE.search(q))
            if obj.get("year_pinned") is True and not q_has_pin_term:
                issues.append({"severity": "major", "check": "year_pinned_without_term", "line": line, "message": "year_pinned=true but question has no explicit year/change term"})
            if obj.get("year_pinned") is False and q_has_pin_term:
                issues.append({"severity": "minor", "check": "year_pin_maybe_missed", "line": line, "message": "question mentions a year/change term but year_pinned=false"})

            max_copy = 0
            max_copy_cid = None
            for cid in ids:
                if cid not in chunks:
                    continue
                m = max_common_ngram(q_toks, tokens(chunks[cid]["text"]))
                if m > max_copy:
                    max_copy = m
                    max_copy_cid = cid
            if max_copy >= 5:
                offenders_copy.append({"line": line, "words": max_copy, "chunk": max_copy_cid, "question": q})
                if max_copy >= 8:
                    issues.append({"severity": "major", "check": "copies_8plus_words", "line": line, "message": f"question copies {max_copy} consecutive words from {max_copy_cid}"})

            if len(ids) > 1:
                per_doc[doc]["multi"] += 1
                for cid, cset in zip(ids, cited_token_sets):
                    overlap = len((set(q_toks) | set(tokens(quote))) & cset)
                    if overlap < 2:
                        issues.append({"severity": "major", "check": "possibly_unnecessary_chunk", "line": line, "message": f"{cid} has <2 content-token overlap with question/evidence"})
            if obj.get("turn1_context"):
                per_doc[doc]["turn1"] += 1
            if obj.get("year_pinned"):
                per_doc[doc]["year_pinned"] += 1
            informal = bool(re.search(r"\b(genre|truc|machin|bah|ben|hein|du coup|en fait|on|je|j'|c'est quoi|ça)\b", norm_for_compare(q))) or (q[:1].islower())
            if informal:
                per_doc[doc]["informal_est"] += 1
            per_doc[doc]["evidence_fail"] += 1 if (quote and not quote_found_cited) else 0

    for qnorm, lines in question_norm_usage.items():
        if len(lines) > 1:
            issues.append({"severity": "blocking", "check": "duplicate_question_exact", "line": lines[0], "message": f"exact duplicate question on lines {lines}"})
    for span, lines in answer_span_usage.items():
        if len(lines) > 1:
            issues.append({"severity": "major", "check": "duplicate_answer_span", "line": lines[0], "message": f"same evidence+gold span reused on lines {lines}"})
    for quote, lines in evidence_usage.items():
        if len(lines) > 1 and quote:
            issues.append({"severity": "minor", "check": "evidence_reused", "line": lines[0], "message": f"same evidence_quote used on lines {lines[:12]}"})

    lines_sorted = sorted(q_tokens)
    near_dupes = []
    for i, l1 in enumerate(lines_sorted):
        t1 = set(q_tokens[l1])
        if len(t1) < 5:
            continue
        for l2 in lines_sorted[i + 1:]:
            t2 = set(q_tokens[l2])
            if len(t2) < 5:
                continue
            inter = len(t1 & t2)
            if inter < 5:
                continue
            jac = inter / max(1, len(t1 | t2))
            if jac >= 0.82:
                near_dupes.append({"lines": [l1, l2], "jaccard": round(jac, 3)})
    for nd in near_dupes[:200]:
        issues.append({"severity": "major", "check": "near_duplicate_question", "line": nd["lines"][0], "message": f"near duplicate with line {nd['lines'][1]} (jaccard={nd['jaccard']})"})

    severity_counts = Counter(i["severity"] for i in issues)
    check_counts = Counter(i["check"] for i in issues)
    total = len(rows)
    summary = {
        "eval_file": str(EVAL_PATH),
        "source_chunk_files": len({rec["file"] for rec in chunks.values()}),
        "source_chunks_loaded": len(chunks),
        "content_rows": total,
        "physical_lines": total + len(blank_lines),
        "blank_lines": blank_lines,
        "severity_counts": dict(severity_counts),
        "check_counts": dict(check_counts),
        "per_doc": {},
        "copy_5plus_rate": round(len(offenders_copy) / max(1, total), 4),
        "copy_8plus_count": sum(1 for o in offenders_copy if o["words"] >= 8),
        "top_question_starters": starter_counts.most_common(15),
        "near_duplicate_pairs": len(near_dupes),
        "type_counts": dict(type_counts),
    }
    for doc in sorted(per_doc):
        c = per_doc[doc]
        summary["per_doc"][doc] = {
            "total": c["total"],
            "multi_chunk": c["multi"],
            "multi_chunk_pct": round(100 * c["multi"] / max(1, c["total"]), 2),
            "turn1_context": c["turn1"],
            "turn1_pct": round(100 * c["turn1"] / max(1, c["total"]), 2),
            "year_pinned": c["year_pinned"],
            "year_pinned_pct": round(100 * c["year_pinned"] / max(1, c["total"]), 2),
            "informal_est": c["informal_est"],
            "informal_est_pct": round(100 * c["informal_est"] / max(1, c["total"]), 2),
            "evidence_fail": c["evidence_fail"],
        }

    report = {"summary": summary, "issues": issues, "copy_offenders_5plus": offenders_copy[:300], "near_duplicate_pairs_sample": near_dupes[:200]}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report={REPORT_PATH}")
    return 1 if severity_counts.get("blocking") else 0


if __name__ == "__main__":
    raise SystemExit(main())
