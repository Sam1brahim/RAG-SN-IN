"""
Build per-year *golden* RAG eval sets from the processed chunks.

Guarantees
----------
- >= `--target` answerable questions per document year (no negatives /
  out-of-scope items: every question is grounded in real chunks).
- gold_answer is entailed by the gold chunk text (token-overlap gate).
- question intent matches answer cues (deadline->time, amount->EUR/%, ...).
- exact + near-dedupe, per-(chunk, qtype) uniqueness, per-type diversity caps.
- non year-pinned questions get cross-year twin gold ids when another
  millésime carries a near-identical section (fairness for retrieval eval).
- gold ids are verified against Qdrant payload['id'] unless --skip-qdrant-check.

Output: data/eval/rag_eval_set_<doc_id>.jsonl  (schema matches the reference
sample: id, question, gold_chunk_ids, gold_answer, question_type, difficulty,
requires_year, answerable, notes, collection, match_field, document_id,
section_path, page_start, page_end)

Usage
-----
  python -m rag_sn_in.eval.build_golden_eval
  python -m rag_sn_in.eval.build_golden_eval --target 560 --skip-qdrant-check
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_sn_in.eval.generate_curated_eval import (
    ACRONYM_RE,
    BAD_Q_SUBSTR,
    EMAIL_RE,
    GENERIC_TOPICS,
    STOP_ACRO,
    answer_supported,
    clean,
    content_tokens,
    nice_topic,
    norm,
    q_key,
    section_meta,
    sentences,
    tokens,
    topic_in_answer,
    with_article,
)

ROOT = Path(__file__).resolve().parents[3]
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
OUT_DIR = ROOT / "data" / "eval"

COLLECTION = "DRR_SNCF"
MATCH_FIELD = "payload.id"

MIN_TOKENS = 70
MAX_ANSWER = 420
NEAR_DUP_JACCARD = 0.8

YEAR_RE = re.compile(r"(\d{4})$")
LIST_RE = re.compile(r"(?m)^\s*[-–—•●]\s+\S+")
TABLE_ROW_RE = re.compile(r"(?m)^\s*\|.+\|\s*$")

DEF_CUE = re.compile(
    r"(?i)(s['’]entend|d[ée]signe|correspond\s+[àa]|est\s+entendue?|"
    r"on\s+entend\s+par)"
)
OBLIGATION_CUE = re.compile(r"(?i)\b(doit|doivent|est\s+tenu|sont\s+tenus)\b")
DEADLINE_CUE = re.compile(
    r"(?i)(\d+\s*(?:jours?|heures?|mois|semaines?)|d[ée]lai|au\s+plus\s+tard|"
    r"\bJ\s*-\s*\d|\bavant\s+le\b|\b[ée]ch[ée]ance)"
)
AMOUNT_CUE = re.compile(
    r"(?i)(redevance|tarif|€|euros?|\d+[.,]?\d*\s*%|factur[ée]|montant)"
)
CONDITION_CUE = re.compile(
    r"(?i)(en\s+cas\s+de|lorsque|d[èe]s\s+lors\s+que|sous\s+r[ée]serve|"
    r"à\s+condition)"
)
ACTOR_CUE = re.compile(
    r"(?i)\b(SNCF\s+R[ée]seau|entreprise\s+ferroviaire|\bEF\b|demandeur|"
    r"candidat|gestionnaire|autorit[ée])\b"
)
ACTOR_ANSWER_CUE = re.compile(
    r"(?i)(responsable|incombe|assure|compétent|est\s+charg[ée]|appartient)"
)
EXPLAINER_CUE = re.compile(
    r"(?i)(est une?|est le|est la|permet|d[ée]signe|correspond|application|"
    r"outil|syst[èe]me|processus|interface|via)"
)

# Per-type caps applied during selection (diversity guard; sum >> target).
TYPE_CAPS = {
    "definition": 90,
    "process": 85,
    "obligation": 60,
    "deadline": 70,
    "amount": 70,
    "condition": 70,
    "actor": 50,
    "contact": 30,
    "system": 55,
    "factual": 175,
    "list": 60,
    "table": 45,
    "section": 75,
    "multi_chunk": 80,
}


def load_chunks(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def usable(chunk: dict[str, Any]) -> bool:
    if chunk.get("token_count", 0) < MIN_TOKENS:
        return False
    text = chunk.get("text") or ""
    if text.count("|") > 40 and len(clean(text)) < 200:
        return False
    return True


def is_table_chunk(text: str) -> bool:
    if "[TABLE]" in text.upper():
        return True
    return len(TABLE_ROW_RE.findall(text)) >= 3


def pick_answer(sents: list[str], fallback: str) -> str:
    if sents:
        joined = " ".join(sents[:2])
        return norm(joined)[:MAX_ANSWER]
    return norm(fallback)[:MAX_ANSWER]


def build_candidates(
    chunk: dict[str, Any], year: str, rng: random.Random
) -> list[dict[str, Any]]:
    """Grounded question candidates for one chunk (one per family max)."""
    cleaned = clean(chunk.get("text") or "")
    sents = sentences(cleaned)
    text = chunk.get("text") or ""
    table = is_table_chunk(text)
    if not sents and not table:
        return []

    sec_num, sec_label = section_meta(chunk)
    topic = nice_topic(sec_label)
    topic_l = topic.lower() if topic else None
    topic_art = with_article(topic_l) if topic_l else None

    cands: list[dict[str, Any]] = []

    def add(question: str, answer: str, qtype: str, difficulty: str) -> None:
        question, answer = norm(question), norm(answer)[:MAX_ANSWER]
        if len(question) < 18 or len(answer) < 50:
            return
        if question.count(" ") > 22:
            return
        ql = question.lower()
        if any(b in ql for b in BAD_Q_SUBSTR):
            return
        if re.search(r"(?i)\b(cadre|notion|principes)\s*\?$", question):
            return
        if not answer_supported(answer, text):
            return
        cands.append(
            {
                "question": question,
                "gold_answer": answer,
                "gold_chunk_ids": [chunk["chunk_id"]],
                "question_type": qtype,
                "difficulty": difficulty,
                "year_pinned": str(year) in question,
            }
        )

    def first_hit(cue: re.Pattern[str]) -> str | None:
        return next((s for s in sents if cue.search(s)), None)

    # --- definition ---
    hit = first_hit(DEF_CUE)
    if hit and topic and topic_in_answer(topic, hit):
        add(
            rng.choice(
                [
                    f"C'est quoi {topic_art} ?",
                    f"Que signifie {topic_art} au sens du DRR {year} ?",
                    f"Qu'entend-on par {topic_art} dans le DRR {year} ?",
                ]
            ),
            hit,
            "definition",
            rng.choice(["easy", "medium"]),
        )

    # --- process ---
    hit = first_hit(OBLIGATION_CUE)
    if hit and topic and topic_in_answer(topic, hit):
        add(
            rng.choice(
                [
                    f"Comment faire pour {topic_art} ?",
                    f"Comment procéder pour {topic_art} selon le DRR {year} ?",
                    f"Quelle est la marche à suivre pour {topic_art} (DRR {year}) ?",
                ]
            ),
            hit,
            "process",
            "medium",
        )
        add(
            rng.choice(
                [
                    f"Qu'est-ce qui est obligatoire concernant {topic_l} ?",
                    f"Que doit respecter l'entreprise ferroviaire pour {topic_l} "
                    f"selon le DRR {year} ?",
                ]
            ),
            hit,
            "obligation",
            "medium",
        )

    # --- deadline ---
    hit = first_hit(DEADLINE_CUE)
    if hit and topic and topic_in_answer(topic, hit):
        add(
            rng.choice(
                [
                    f"Quel délai pour {topic_l} ?",
                    f"Avant quand faut-il s'occuper de {topic_l} ?",
                    f"Quelle échéance le DRR {year} fixe-t-il pour {topic_l} ?",
                ]
            ),
            hit,
            "deadline",
            "hard",
        )

    # --- amount ---
    hit = first_hit(AMOUNT_CUE)
    if hit and topic and topic_in_answer(topic, hit):
        add(
            rng.choice(
                [
                    f"Quel montant ou quelle redevance pour {topic_l} ?",
                    f"Comment est calculée la facturation liée à {topic_l} ?",
                    f"Quelles règles tarifaires s'appliquent à {topic_l} "
                    f"dans le DRR {year} ?",
                ]
            ),
            hit,
            "amount",
            "hard",
        )

    # --- condition ---
    hit = first_hit(CONDITION_CUE)
    if hit and topic and topic_in_answer(topic, hit):
        add(
            rng.choice(
                [
                    f"Que se passe-t-il en cas de problème avec {topic_l} ?",
                    f"Dans quelles conditions le DRR {year} applique-t-il "
                    f"{topic_l} ?",
                    f"Quand {topic_l} entre-t-il en jeu selon le DRR {year} ?",
                ]
            ),
            hit,
            "condition",
            "hard",
        )

    # --- actor ---
    hit = next(
        (s for s in sents if ACTOR_CUE.search(s) and ACTOR_ANSWER_CUE.search(s)),
        None,
    )
    if hit and topic and topic_in_answer(topic, hit):
        add(
            rng.choice(
                [
                    f"Qui est responsable de {topic_l} ?",
                    f"Quel acteur intervient pour {topic_l} selon le DRR {year} ?",
                ]
            ),
            hit,
            "actor",
            "easy",
        )

    # --- contact ---
    emails = EMAIL_RE.findall(text)
    if emails:
        hit = next((s for s in sents if EMAIL_RE.search(s)), None)
        if hit and topic:
            add(
                rng.choice(
                    [
                        f"À quelle adresse e-mail écrire pour {topic_l} ?",
                        f"Où envoyer une demande concernant {topic_l} ?",
                    ]
                ),
                hit,
                "contact",
                "easy",
            )

    # --- system / acronym ---
    for acro in ACRONYM_RE.findall(text):
        if acro in STOP_ACRO or len(acro) < 3:
            continue
        occ = len(re.findall(rf"\b{acro}\b", text))
        has_expansion = bool(re.search(rf"\b{acro}\b\s*[:(]|[:(]\s*{acro}\b", text))
        if occ < 2 and not has_expansion:
            continue
        hit = next(
            (
                s
                for s in sents
                if re.search(rf"\b{acro}\b", s) and EXPLAINER_CUE.search(s)
            ),
            None,
        )
        if hit:
            add(
                rng.choice(
                    [
                        f"C'est quoi {acro} ?",
                        f"À quoi sert {acro} dans le DRR {year} ?",
                        f"{acro} sert à quoi ?",
                    ]
                ),
                hit,
                rng.choice(["definition", "system"]),
                "easy",
            )
        break

    # --- list ---
    if (LIST_RE.search(text) or cleaned.count(";") >= 2) and topic:
        answer = pick_answer(sents[:2], cleaned)
        if topic_in_answer(topic, answer):
            add(
                rng.choice(
                    [
                        f"Quels éléments composent {topic_l} ?",
                        f"Quelles catégories le DRR {year} énumère-t-il pour "
                        f"{topic_l} ?",
                    ]
                ),
                answer,
                "list",
                "medium",
            )

    # --- table ---
    if table and topic:
        rows = [norm(r) for r in TABLE_ROW_RE.findall(text)[:10]]
        answer = " | ".join(rows) if rows else pick_answer(sents[:2], cleaned)
        if len(answer) >= 50:
            add(
                rng.choice(
                    [
                        f"Que montre le tableau du DRR {year} concernant "
                        f"{topic_l} ?",
                        f"Quelles données le tableau sur {topic_l} fournit-il ?",
                    ]
                ),
                answer,
                "table",
                "hard",
            )

    # --- factual (lead sentence) ---
    if topic and sents and topic_in_answer(topic, sents[0]):
        add(
            rng.choice(
                [
                    f"Tu peux m'expliquer {topic_art} ?",
                    f"Qu'est-ce que je dois savoir sur {topic_art} ?",
                    f"Que dit le DRR {year} à propos de {topic_l} ?",
                ]
            ),
            sents[0],
            "factual",
            "easy",
        )

    # --- generic section overview (works even without a "nice" topic) ---
    raw_label = sec_label
    if not raw_label:
        for part in reversed(chunk.get("section_path") or []):
            part = part.strip()
            if part and not part.upper().startswith("CHAPITRE"):
                raw_label = part
                break
    if raw_label:
        raw_label = re.sub(r"^[•●\-–—\uf0b7]+\s*", "", raw_label)
        raw_label = re.sub(r"^\d+(?:\.\d+)*\s+", "", raw_label).strip()
        if raw_label.isupper() and len(raw_label) > 12:
            raw_label = raw_label.capitalize()
    if (
        raw_label
        and 8 <= len(raw_label) <= 80
        and not GENERIC_TOPICS.search(raw_label)
        and sents
    ):
        answer = pick_answer(sents[:2], cleaned)
        if topic_in_answer(raw_label, answer, min_hits=1):
            add(
                rng.choice(
                    [
                        f"Que précise le DRR {year} dans la section « {raw_label} » ?",
                        f"Quelles informations la partie « {raw_label} » du "
                        f"DRR {year} donne-t-elle ?",
                        f"Résume l'essentiel de « {raw_label} » selon le DRR {year}.",
                    ]
                ),
                answer,
                "factual",
                "easy",
            )

    # --- section anchor (works even without a nice topic) ---
    if sec_num and sec_label and sents:
        answer = pick_answer(sents[:2], cleaned)
        add(
            rng.choice(
                [
                    f"Que prévoit la section {sec_num} du DRR {year} "
                    f"({sec_label}) ?",
                    f"Quelles règles la section {sec_num} ({sec_label}) du "
                    f"DRR {year} définit-elle ?",
                ]
            ),
            answer,
            "section",
            "medium",
        )

    return cands


def build_multi_chunk(
    chunks: list[dict[str, Any]], year: str, rng: random.Random, limit: int
) -> list[dict[str, Any]]:
    """Pairs of consecutive substantive chunks spanning two sections."""
    group = sorted(
        (c for c in chunks if usable(c)), key=lambda c: c["chunk_id"]
    )
    items: list[dict[str, Any]] = []
    for a, b in zip(group, group[1:]):
        _, label_a = section_meta(a)
        _, label_b = section_meta(b)
        if not label_a or not label_b or label_a == label_b:
            continue
        if abs((a.get("page_start") or 0) - (b.get("page_start") or 0)) > 2:
            continue
        ans_a = pick_answer(sentences(clean(a["text"]))[:1], clean(a["text"]))
        ans_b = pick_answer(sentences(clean(b["text"]))[:1], clean(b["text"]))
        items.append(
            {
                "question": (
                    f"Dans le DRR {year}, que faut-il retenir conjointement "
                    f"sur « {label_a} » et « {label_b} » ?"
                ),
                "gold_answer": (
                    f"[{label_a}] {ans_a[:190]} || [{label_b}] {ans_b[:190]}"
                )[:MAX_ANSWER],
                "gold_chunk_ids": [a["chunk_id"], b["chunk_id"]],
                "question_type": "multi_chunk",
                "difficulty": "hard",
                "year_pinned": True,
            }
        )
    rng.shuffle(items)
    return items[:limit]


def section_twins(
    all_chunks: dict[str, list[dict[str, Any]]]
) -> dict[str, list[str]]:
    """chunk_id -> twin chunk ids in OTHER years with near-identical section."""
    by_sec: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for doc, chunks in all_chunks.items():
        for c in chunks:
            if not usable(c):
                continue
            sec_num, _ = section_meta(c)
            if not sec_num:
                continue
            prev = by_sec[sec_num].get(doc)
            if prev is None or c["token_count"] > prev["token_count"]:
                by_sec[sec_num][doc] = c

    twins: dict[str, list[str]] = {}
    for per_doc in by_sec.values():
        if len(per_doc) < 2:
            continue
        for doc, chunk in per_doc.items():
            base = set(tokens(clean(chunk["text"])))
            for other_doc, other in per_doc.items():
                if other_doc == doc:
                    continue
                other_toks = set(tokens(clean(other["text"])))
                union = len(base | other_toks) or 1
                if len(base & other_toks) / union >= 0.55:
                    twins.setdefault(chunk["chunk_id"], []).append(
                        other["chunk_id"]
                    )
    return twins


def select_year_set(
    cands: list[dict[str, Any]], target: int, rng: random.Random
) -> list[dict[str, Any]]:
    rng.shuffle(cands)
    counts: Counter[str] = Counter()
    seen_q: set[str] = set()
    seen_cq: set[tuple[str, str]] = set()
    chosen: list[dict[str, Any]] = []
    chosen_toks: list[set[str]] = []

    for cand in cands:
        if len(chosen) >= target:
            break
        qk = q_key(cand["question"])
        if qk in seen_q:
            continue
        qt = cand["question_type"]
        if counts[qt] >= TYPE_CAPS.get(qt, 40):
            continue
        cid = cand["gold_chunk_ids"][0]
        if (cid, qt) in seen_cq:
            continue
        toks = tokens(cand["question"])
        window = chosen_toks[-250:]
        if any(
            len(toks & p) / (len(toks | p) or 1) >= NEAR_DUP_JACCARD
            for p in window
        ):
            continue
        seen_q.add(qk)
        seen_cq.add((cid, qt))
        counts[qt] += 1
        chosen.append(cand)
        chosen_toks.append(toks)
    return chosen


def qdrant_indexed_ids() -> set[str] | None:
    """All payload['id'] values in the collection; None if unavailable."""
    try:
        from rag_sn_in.database.client import get_client

        client = get_client()
        ids: set[str] = set()
        offset = None
        while True:
            pts, offset = client.scroll(
                COLLECTION, limit=1000, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for p in pts:
                pid = (p.payload or {}).get("id")
                if pid:
                    ids.add(pid)
            if offset is None:
                break
        return ids
    except Exception as exc:  # lock held / client missing -> warn & skip
        print(f"WARNING: Qdrant check skipped ({exc.__class__.__name__}: {exc})")
        return None


def make_row(eval_id: str, item: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    default_notes = (
        "auto-golden; grounded; chunk-pair"
        if item["question_type"] == "multi_chunk"
        else "auto-golden; grounded; single-chunk"
    )
    return {
        "id": eval_id,
        "question": item["question"],
        "gold_chunk_ids": item["gold_chunk_ids"],
        "gold_answer": item["gold_answer"],
        "question_type": item["question_type"],
        "difficulty": item["difficulty"],
        "requires_year": True,
        "answerable": True,
        "notes": item.get("notes", default_notes),
        "collection": COLLECTION,
        "match_field": MATCH_FIELD,
        "document_id": chunk["document_id"],
        "section_path": chunk.get("section_path") or [],
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
    }


def build_year(
    doc_id: str,
    chunks: list[dict[str, Any]],
    twins: dict[str, list[str]],
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    year = YEAR_RE.search(doc_id).group(1)

    cands: list[dict[str, Any]] = []
    for chunk in chunks:
        if usable(chunk):
            cands.extend(build_candidates(chunk, year, rng))
    cands.extend(build_multi_chunk(chunks, year, rng, limit=TYPE_CAPS["multi_chunk"] * 2))

    chosen = select_year_set(cands, target, rng)

    by_id = {c["chunk_id"]: c for c in chunks}
    rows: list[dict[str, Any]] = []
    for item in chosen:
        primary = by_id[item["gold_chunk_ids"][0]]
        extra: list[str] = []
        if not item["year_pinned"]:
            extra = [t for t in twins.get(primary["chunk_id"], []) if t not in item["gold_chunk_ids"]]
        if extra:
            item = {**item, "gold_chunk_ids": item["gold_chunk_ids"] + extra}
            item["notes"] = "auto-golden; grounded; cross-year-twins"
        rows.append(item)
    rng.shuffle(rows)

    final: list[dict[str, Any]] = []
    for i, item in enumerate(rows, start=1):
        primary = by_id[item["gold_chunk_ids"][0]]
        final.append(make_row(f"eval_{i:04d}", item, primary))
    return final


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=560,
                        help="questions per year BEFORE qdrant filtering")
    parser.add_argument("--skip-qdrant-check", action="store_true")
    args = parser.parse_args()

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    all_chunks = {p.stem: load_chunks(p) for p in chunk_files}
    print({doc: len(cs) for doc, cs in all_chunks.items()})

    twins = section_twins(all_chunks)
    print(f"cross-year twin links: {sum(len(v) for v in twins.values())}")

    indexed = None if args.skip_qdrant_check else qdrant_indexed_ids()
    if indexed is not None:
        print(f"qdrant payload ids: {len(indexed)}")

    for seed, (doc_id, chunks) in enumerate(sorted(all_chunks.items()), start=1):
        rows = build_year(doc_id, chunks, twins, args.target, seed=1000 + seed)

        # every gold id (own year or cross-year twin) must be reachable
        # through Qdrant payload['id'].
        if indexed is not None:
            before = len(rows)
            rows = [
                r for r in rows
                if all(cid in indexed for cid in r["gold_chunk_ids"])
            ]
            dropped = before - len(rows)
            if dropped:
                print(f"[{doc_id}] dropped {dropped} rows w/ ids missing in Qdrant")

        out_path = OUT_DIR / f"rag_eval_set_{doc_id}.jsonl"
        write_jsonl(rows, out_path)
        print(
            f"[{doc_id}] wrote {len(rows)} -> {out_path.name} | types:",
            dict(Counter(r["question_type"] for r in rows)),
        )

        print(f"[{doc_id}] audit sample:")
        rng = random.Random(42)
        for row in rng.sample(rows, k=min(3, len(rows))):
            print(f"  {row['id']} [{row['question_type']}] {row['question']}")
            print(f"    gold={row['gold_chunk_ids']} ans={row['gold_answer'][:110]}…")

    print("\nDone.")


if __name__ == "__main__":
    main()
