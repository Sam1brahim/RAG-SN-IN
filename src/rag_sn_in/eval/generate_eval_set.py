"""
Build a large, diverse, deduplicated RAG evaluation set from DRR chunks,
with optional faithfulness checks against raw PDFs.

Design goals
------------
- Every answerable item is grounded in real chunk_id(s).
- No duplicate questions (exact + aggressive near-duplicate stems).
- Diverse question families (process, definition, deadline, actor,
  amount, condition, list, table, systems, cross-year, negative…).
- Cap overused linguistic frames so the set is not template spam.

Usage
-----
  python -m rag_sn_in.eval.generate_eval_set
  python -m rag_sn_in.eval.generate_eval_set --pdf-check-sample 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "eval" / "rag_eval_set.jsonl"

MIN_TOKENS = 70
MAX_ANSWER = 650
RNG = random.Random(7)

YEAR_FROM_DOC = {
    "drr-2025": "2025",
    "drr-2026": "2026",
    "drr-2027": "2027",
}

HEADER_RE = re.compile(r"\[section_header\]\s*(.+?)(?:\n|$)", re.I)
SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
SPACE_RE = re.compile(r"\s+")
TABLE_ROW_RE = re.compile(r"(?m)^\s*\|.+\|\s*$")

ACRONYM_RE = re.compile(r"\b([A-ZÁÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]{2,6})\b")
STOP_ACRONYMS = {
    "DRR", "RFN", "SNCF", "UE", "EU", "PDF", "HTTP", "HTTPS", "WWW",
    "CHAPITRE", "TABLE", "ANNEXE", "ART", "CF", "NB", "ETC", "SI",
}

DEFINITION_RE = re.compile(
    r"(?i)\b(s['’]entend(?:\s+comme)?|d[ée]signe|correspond\s+[àa]|"
    r"est\s+(?:entendue?|d[ée]finie?)\s+comme|on\s+entend\s+par|"
    r"d[ée]finition)\b"
)
OBLIGATION_RE = re.compile(
    r"(?i)\b(doit|doivent|est\s+tenu|sont\s+tenus|est\s+obligatoire|"
    r"sont\s+obligatoires|il\s+appartient|est\s+requis|veut)\b"
)
DEADLINE_RE = re.compile(
    r"(?i)(d[ée]lai(?:\s+de)?\s+\d+|\d+\s*(?:jours?|heures?|mois|ans?|"
    r"semaines?)|\bau\s+plus\s+tard\b|\bavant\s+le\b)"
)
AMOUNT_RE = re.compile(
    r"(?i)(\d+(?:[.,]\d+)?\s*(?:€|euros?|EUR|%)|redevance|tarif|"
    r"montant|prix|composante\s+[AB])"
)
ACTOR_RE = re.compile(
    r"(?i)\b(SNCF\s+R[ée]seau|entreprise\s+ferroviaire|\bEF\b|\bGI\b|"
    r"\bSGC\b|demandeur|candidat|autorit[ée]|ART)\b"
)
CONDITION_RE = re.compile(
    r"(?i)\b(lorsque|d[èe]s\s+lors\s+que|si\s+l[ae'\s]|en\s+cas\s+de|"
    r"sous\s+r[ée]serve|à\s+condition|sauf|hors\s+cas)\b"
)
LIST_RE = re.compile(r"(?m)^\s*[-–—•●]\s+\S+")
SYSTEM_RE = re.compile(
    r"(?i)\b(DINAMIC|DANC|DECOFER|ORES|CIS|GPF|GOC|SCo|CCL)\b"
)

BAD_TOPIC_RE = re.compile(
    r"(?i)("
    r"^\d+\)|"
    r"^[ivx]+\)|"
    r"^[•●\-–—]|"
    r"une facture|"
    r"facture de r[ée]gularisation|"
    r"^tableau|"
    r"\[table\]|"
    r"cas particuliers de l'utilisation"
    r")"
)

# Max items that may share the same linguistic frame family
FRAME_CAPS = {
    "section_lookup": 700,
    "process_how": 900,
    "definition": 700,
    "deadline": 500,
    "amount": 500,
    "actor": 500,
    "condition": 600,
    "list": 500,
    "system": 500,
    "yes_no": 400,
    "table": 400,
    "factual_detail": 800,
    "cross_year": 350,
    "multi_chunk": 500,
    "negative": 220,
}


def load_chunks() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def norm_space(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    text = text.replace("\uf0b7", "-").replace("\u00a0", " ")
    text = re.sub(r"\[TABLE\]", "", text, flags=re.I)
    text = re.sub(r"\[section_header\]\s*", "", text, flags=re.I)
    return norm_space(text)


def clean_label(label: str | None) -> str | None:
    if not label:
        return None
    label = re.sub(r"^[\u2022\u25cf\u25a0●•\-–—]+\s*", "", label.strip())
    label = re.sub(r"^[ivx]+\)\s*", "", label, flags=re.I)
    if label.upper() in {"[TABLE]", "TABLE"}:
        return None
    return label or None


def section_title(chunk: dict[str, Any]) -> str | None:
    for part in reversed(chunk.get("section_path") or []):
        part = part.strip()
        if part.startswith("[section_header]"):
            title = part.replace("[section_header]", "", 1).strip()
            if title and not title.upper().startswith("CHAPITRE"):
                return clean_label(title)
        if part and not part.upper().startswith("CHAPITRE"):
            return clean_label(part)
    m = HEADER_RE.search(chunk.get("text") or "")
    if m:
        return clean_label(m.group(1))
    return None


def parse_section(title: str | None) -> tuple[str | None, str | None]:
    if not title:
        return None, None
    m = SECTION_NUM_RE.match(title.strip())
    if not m:
        return None, title.strip()
    return m.group(1), clean_label(m.group(2))


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for s in SENTENCE_RE.split(text):
        s = norm_space(s)
        if len(s) < 45:
            continue
        if s.count("|") >= 4:
            continue
        out.append(s)
    return out


def year_of(chunk: dict[str, Any]) -> str:
    return YEAR_FROM_DOC.get(chunk["document_id"], chunk["document_id"])


def is_table_chunk(text: str) -> bool:
    if "[TABLE]" in text.upper():
        return True
    return len(TABLE_ROW_RE.findall(text)) >= 3


def pick_answer(cands: list[str], fallback: str) -> str:
    if cands:
        return max(cands, key=len)[:MAX_ANSWER]
    return norm_space(fallback)[:MAX_ANSWER]


def question_dedupe_key(question: str) -> str:
    """Exact dedupe key (years kept — year-pinned Qs are distinct)."""
    q = question.lower()
    q = re.sub(r"[«»\"“”]", " ", q)
    q = re.sub(r"[^\wÀ-ÿ%]+", " ", q, flags=re.UNICODE)
    return SPACE_RE.sub(" ", q).strip()


def question_tokens(question: str) -> set[str]:
    key = question_dedupe_key(question)
    return {t for t in key.split() if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def frame_family(question: str, qtype: str) -> str:
    q = question.lower()
    if qtype == "negative":
        return "negative"
    if qtype == "cross_year":
        return "cross_year"
    if qtype == "multi_chunk":
        return "multi_chunk"
    if qtype == "table":
        return "table"
    if "comment" in q and ("procéd" in q or "faire" in q or "demander" in q):
        return "process_how"
    if "définit" in q or "désigne" in q or "qu'est-ce" in q or "que signifie" in q:
        return "definition"
    if "délai" in q or "quand" in q or "échéance" in q:
        return "deadline"
    if "montant" in q or "redevance" in q or "tarif" in q or "combien" in q:
        return "amount"
    if "qui " in q or q.startswith("qui") or "responsable" in q:
        return "actor"
    if "condition" in q or "lorsque" in q or "dans quel cas" in q:
        return "condition"
    if "quels sont" in q or "quelles sont" in q or "énumère" in q:
        return "list"
    if "dinamic" in q or "danc" in q or "système" in q or "outil" in q:
        return "system"
    if q.startswith("est-ce") or "est-il exact" in q or " vrai que" in q:
        return "yes_no"
    if qtype == "section" or ("section" in q and ("prévoit" in q or "règles" in q)):
        return "section_lookup"
    return qtype if qtype in FRAME_CAPS else "factual_detail"


class DedupeIndex:
    def __init__(self, jaccard_threshold: float = 0.88) -> None:
        self.exact: set[str] = set()
        self.frames: Counter[str] = Counter()
        self._token_sets: list[set[str]] = []
        self.jaccard_threshold = jaccard_threshold
        self.rejected_exact = 0
        self.rejected_near = 0
        self.rejected_frame = 0

    def accept(self, question: str, qtype: str) -> bool:
        ek = question_dedupe_key(question)
        if ek in self.exact:
            self.rejected_exact += 1
            return False

        toks = question_tokens(question)
        # Compare against recent items only for speed, plus all if small
        # Full scan is OK for ~10k candidates.
        for prev in self._token_sets:
            if jaccard(toks, prev) >= self.jaccard_threshold:
                self.rejected_near += 1
                return False

        fam = frame_family(question, qtype)
        cap = FRAME_CAPS.get(fam, 600)
        if self.frames[fam] >= cap:
            self.rejected_frame += 1
            return False

        self.exact.add(ek)
        self.frames[fam] += 1
        self._token_sets.append(toks)
        return True


def extract_acronyms(text: str) -> list[str]:
    found = []
    seen = set()
    for m in ACRONYM_RE.finditer(text):
        a = m.group(1)
        if a in STOP_ACRONYMS or a in seen:
            continue
        # Prefer acronyms that appear with an expansion nearby
        seen.add(a)
        found.append(a)
    return found[:6]


def make_item(
    *,
    eval_id: str,
    question: str,
    gold_chunk_ids: list[str],
    gold_answer: str,
    question_type: str,
    difficulty: str,
    requires_year: bool,
    answerable: bool = True,
    chunk: dict[str, Any] | None = None,
    notes: str = "",
    pdf_check: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": eval_id,
        "question": norm_space(question),
        "gold_chunk_ids": gold_chunk_ids,
        "gold_answer": norm_space(gold_answer)[:MAX_ANSWER],
        "question_type": question_type,
        "difficulty": difficulty,
        "requires_year": requires_year,
        "answerable": answerable,
    }
    if chunk is not None:
        item["document_id"] = chunk["document_id"]
        item["section_path"] = chunk.get("section_path") or []
        item["page_start"] = chunk.get("page_start")
        item["page_end"] = chunk.get("page_end")
    if notes:
        item["notes"] = notes
    if pdf_check is not None:
        item["pdf_check"] = pdf_check
    return item


def candidates_for_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = chunk.get("text") or ""
    if chunk.get("token_count", 0) < MIN_TOKENS:
        return []
    cleaned = clean_text(text)
    if len(cleaned) < 90:
        return []

    title = section_title(chunk)
    sec_num, sec_label = parse_section(title)
    year = year_of(chunk)
    sents = sentences(cleaned)
    table = is_table_chunk(text)
    if not sents and not table:
        return []

    topic = sec_label or title or "ce dispositif"
    if topic and BAD_TOPIC_RE.search(topic.strip()):
        topic = "ce dispositif"
    # Drop obviously broken / too-short topical labels
    if topic and (len(topic) < 8 or topic.count("?") > 0):
        topic = "ce dispositif"
    topic_l = topic.lower()
    cands: list[dict[str, Any]] = []

    def add(
        question: str,
        answer: str,
        qtype: str,
        difficulty: str,
        requires_year: bool = True,
        notes: str = "",
    ) -> None:
        q = norm_space(question)
        a = norm_space(answer)
        if len(a) < 35 or len(q) < 30:
            return
        # reject questions that still carry OCR junk bullets
        if re.search(r"[•●]", q):
            return
        if BAD_TOPIC_RE.search(q):
            return
        cands.append(
            {
                "question": q,
                "gold_answer": a[:MAX_ANSWER],
                "question_type": qtype,
                "difficulty": difficulty,
                "requires_year": requires_year,
                "notes": notes,
                "chunk": chunk,
            }
        )

    # --- process / how ---
    for sent in sents:
        if OBLIGATION_RE.search(sent) and (
            "doit" in sent.lower() or "doivent" in sent.lower()
        ):
            variants = [
                f"Comment faut-il procéder, selon le DRR {year}, "
                f"concernant {topic_l} ?",
                f"Que doit faire l'acteur concerné, d'après le DRR {year}, "
                f"dans le cadre de {topic_l} ?",
                f"Quelle marche à suivre le DRR {year} impose-t-il pour "
                f"{topic_l} ?",
            ]
            add(RNG.choice(variants), sent, "process", "medium")
            break

    # --- definitions ---
    for sent in sents:
        if DEFINITION_RE.search(sent):
            variants = [
                f"Que désigne « {topic} » dans le DRR {year} ?",
                f"Quelle définition le DRR {year} donne-t-il de {topic_l} ?",
                f"Qu'entend-on par « {topic} » au sens du DRR {year} ?",
            ]
            add(RNG.choice(variants), sent, "definition", "medium")
            break

    # --- deadlines ---
    for sent in sents:
        if DEADLINE_RE.search(sent):
            variants = [
                f"Quel délai le DRR {year} fixe-t-il pour {topic_l} ?",
                f"En combien de temps doit être réalisée l'action liée à "
                f"{topic_l} selon le DRR {year} ?",
                f"Quelle échéance s'applique à {topic_l} dans le DRR {year} ?",
            ]
            add(RNG.choice(variants), sent, "deadline", "hard")
            break

    # --- amounts / tariffs ---
    for sent in sents:
        if AMOUNT_RE.search(sent):
            variants = [
                f"Quels montants, redevances ou valeurs le DRR {year} "
                f"associe-t-il à {topic_l} ?",
                f"Comment est tarifé / valorisé {topic_l} dans le DRR {year} ?",
                f"Quels éléments chiffrés le DRR {year} indique-t-il pour "
                f"{topic_l} ?",
            ]
            add(RNG.choice(variants), sent, "amount", "hard")
            break

    # --- actors ---
    for sent in sents:
        if ACTOR_RE.search(sent) and (
            "responsable" in sent.lower()
            or "assuré" in sent.lower()
            or "compétent" in sent.lower()
            or "SNCF" in sent
            or " EF " in f" {sent} "
        ):
            variants = [
                f"Qui intervient / est responsable pour {topic_l} "
                f"selon le DRR {year} ?",
                f"Quel acteur le DRR {year} mobilise-t-il dans le cadre de "
                f"{topic_l} ?",
            ]
            add(RNG.choice(variants), sent, "actor", "medium")
            break

    # --- conditions ---
    for sent in sents:
        if CONDITION_RE.search(sent):
            variants = [
                f"Dans quelles conditions le DRR {year} prévoit-il "
                f"l'application de {topic_l} ?",
                f"Quand {topic_l} s'applique-t-il d'après le DRR {year} ?",
                f"Quels cas de figure déclenchent {topic_l} dans le "
                f"DRR {year} ?",
            ]
            add(RNG.choice(variants), sent, "condition", "hard")
            break

    # --- lists ---
    if LIST_RE.search(text) or cleaned.count(";") >= 2:
        answer = pick_answer(sents[:2], cleaned)
        variants = [
            f"Quels éléments / catégories le DRR {year} énumère-t-il "
            f"pour {topic_l} ?",
            f"Quelles sont les composantes prévues par le DRR {year} "
            f"concernant {topic_l} ?",
        ]
        add(RNG.choice(variants), answer, "list", "medium")

    # --- systems / named tools ---
    systems = SYSTEM_RE.findall(cleaned)
    if systems:
        sys = systems[0]
        # find a sentence mentioning it
        hit = next((s for s in sents if re.search(re.escape(sys), s, re.I)), None)
        if hit:
            variants = [
                f"Quel rôle joue {sys} dans le DRR {year} "
                f"(contexte : {topic_l}) ?",
                f"Que dit le DRR {year} à propos de {sys} ?",
                f"Comment {sys} intervient-il pour {topic_l} selon le "
                f"DRR {year} ?",
            ]
            add(RNG.choice(variants), hit, "system", "medium")

    # --- acronym focused ---
    for acro in extract_acronyms(cleaned):
        hit = next((s for s in sents if re.search(rf"\b{acro}\b", s)), None)
        if not hit:
            continue
        # Prefer sentences that look definitional or explanatory
        if len(hit) < 60:
            continue
        variants = [
            f"Que signifie ou que couvre {acro} dans le DRR {year} ?",
            f"Que précise le DRR {year} au sujet de {acro} ?",
        ]
        add(RNG.choice(variants), hit, "definition", "medium")
        break

    # --- yes/no grounded ---
    if sents and RNG.random() < 0.45:
        sent = RNG.choice(sents[:3])
        # Build a verifiable yes question from a clear assertion
        if len(sent) > 80:
            snippet = sent[:140].rstrip(" ,;")
            add(
                f"D'après le DRR {year}, est-il exact que : {snippet} … ?",
                sent,
                "yes_no",
                "easy",
                notes="Gold answer is the supporting passage (expect yes).",
            )

    # --- section anchor (limited; more natural wording) ---
    if sec_num and sec_label and sents:
        answer = pick_answer(sents[:2], cleaned)
        variants = [
            f"Résume les règles du DRR {year} applicables à « {sec_label} » "
            f"(réf. {sec_num}).",
            f"Quelles dispositions concrètes le DRR {year} pose-t-il en "
            f"{sec_num} sur {sec_label.lower()} ?",
            f"Pour un candidat au RFN en {year}, que changerait la lecture "
            f"du point {sec_num} ({sec_label}) ?",
        ]
        add(RNG.choice(variants), answer, "section", "easy")

    # --- table ---
    if table:
        rows = [norm_space(r) for r in TABLE_ROW_RE.findall(text)[:12]]
        answer = " | ".join(rows) if rows else pick_answer(sents[:2], cleaned)
        if len(answer) >= 35:
            variants = [
                f"Quelles données le tableau du DRR {year} fournit-il pour "
                f"{topic_l} ?",
                f"Extrais les informations clés du tableau DRR {year} relatif "
                f"à {topic_l}.",
            ]
            add(
                RNG.choice(variants),
                answer,
                "table",
                "hard",
                notes="Table excerpt; score retrieval primarily via chunk_id.",
            )

    # --- factual fallback ---
    if not cands and sents:
        add(
            f"Quel point important le DRR {year} développe-t-il à propos de "
            f"{topic_l} ?",
            pick_answer(sents[:2], cleaned),
            "factual",
            "easy",
        )

    RNG.shuffle(cands)
    return cands


def build_cross_year(
    chunks: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_sec: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for chunk in chunks:
        if chunk.get("token_count", 0) < MIN_TOKENS:
            continue
        sec_num, sec_label = parse_section(section_title(chunk))
        if not sec_num or not sec_label:
            continue
        doc = chunk["document_id"]
        prev = by_sec[sec_num].get(doc)
        if prev is None or chunk["token_count"] > prev["token_count"]:
            by_sec[sec_num][doc] = chunk

    items: list[dict[str, Any]] = []
    frames = [
        (
            "Compare les dispositions des DRR {years} pour la section "
            "{sec} (« {label} »). Y a-t-il des divergences notables ?"
        ),
        (
            "Le traitement de « {label} » (section {sec}) est-il identique "
            "dans les DRR {years} ? Appuie-toi sur chaque millésime."
        ),
        (
            "Quelles précisions les DRR {years} apportent-ils chacun sur "
            "{label} (réf. {sec}) ?"
        ),
    ]
    for sec_num, per_doc in by_sec.items():
        if len(per_doc) < 2:
            continue
        docs = sorted(per_doc.keys())
        label = parse_section(section_title(per_doc[docs[0]]))[1] or sec_num
        years = [YEAR_FROM_DOC[d] for d in docs]
        gold_ids = [per_doc[d]["chunk_id"] for d in docs]
        answers = []
        for d in docs:
            cleaned = clean_text(per_doc[d]["text"])
            answers.append(
                f"[{YEAR_FROM_DOC[d]}] "
                + pick_answer(sentences(cleaned)[:1], cleaned)[:260]
            )
        q = RNG.choice(frames).format(
            years=", ".join(years),
            sec=sec_num,
            label=label,
        )
        items.append(
            {
                "question": q,
                "gold_answer": " || ".join(answers)[:1200],
                "question_type": "cross_year",
                "difficulty": "hard",
                "requires_year": True,
                "gold_chunk_ids": gold_ids,
                "document_id": docs[-1],
                "section_path": per_doc[docs[-1]].get("section_path") or [],
                "page_start": per_doc[docs[-1]].get("page_start"),
                "page_end": per_doc[docs[-1]].get("page_end"),
                "answerable": True,
                "notes": "Multi-year gold chunks.",
            }
        )
    RNG.shuffle(items)
    return items[:limit]


def build_negatives(n: int) -> list[dict[str, Any]]:
    seeds = [
        "tarifs d'atterrissage à Roissy-CDG",
        "renouvellement du permis de conduire B",
        "déclaration des revenus fonciers",
        "vaccination recommandée par la HAS",
        "licence de pêche en eau douce",
        "accès au métro londonien (TfL)",
        "abonnement de streaming vidéo",
        "trading d'actions sur Euronext",
        "drones de loisir en zone urbaine",
        "résiliation d'assurance habitation",
        "horaires d'ouverture du Louvre",
        "dépôt de marque à l'INPI",
        "composition du Conseil constitutionnel",
        "visa touristique pour le Japon",
        "construction d'éoliennes offshore",
        "contestation d'amende de stationnement",
        "régime social des auto-entrepreneurs",
        "homologation d'un médicament (EMA)",
        "transferts de joueurs selon la FIFA",
        "ouverture d'un compte crypto régulé",
        "normes HACCP en restauration collective",
        "immatriculation d'un véhicule d'occasion",
        "demande de logement social (DALO)",
        "cotisations URSSAF d'une SASU",
        "règles de survol du Mont-Blanc",
    ]
    frames = [
        "Quelles règles s'appliquent à : {topic} ?",
        "Peux-tu expliquer la procédure officielle pour {topic} ?",
        "Quel texte encadre {topic} en France ?",
        "Donne les étapes concrètes concernant {topic}.",
        "Y a-t-il des délais légaux pour {topic} ?",
        "Qui est l'autorité compétente pour {topic} ?",
        "Quels documents faut-il fournir pour {topic} ?",
        "Quel est le coût typique associé à {topic} ?",
    ]
    items: list[dict[str, Any]] = []
    i = 0
    while len(items) < n:
        topic = seeds[i % len(seeds)]
        frame = frames[(i // len(seeds)) % len(frames)]
        # make unique by index tag only in notes; question body uniqueness via combo
        q = frame.format(topic=topic)
        if i >= len(seeds) * len(frames):
            q = f"{q} (cas n°{i})"
        items.append(
            {
                "question": q,
                "gold_chunk_ids": [],
                "gold_answer": (
                    "Question hors périmètre du DRR SNCF Réseau. "
                    "Le système ne doit pas inventer une réponse à partir du corpus."
                ),
                "question_type": "negative",
                "difficulty": "easy",
                "requires_year": False,
                "answerable": False,
                "notes": f"negative-{i}",
            }
        )
        i += 1
    return items


def build_multi_chunk(
    chunks: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Pair consecutive substantive chunks from the same document."""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        if c.get("token_count", 0) >= MIN_TOKENS:
            by_doc[c["document_id"]].append(c)

    items: list[dict[str, Any]] = []
    for doc, group in by_doc.items():
        group = sorted(group, key=lambda x: x["chunk_id"])
        year = YEAR_FROM_DOC.get(doc, doc)
        for a, b in zip(group, group[1:]):
            ta = section_title(a)
            tb = section_title(b)
            if not ta or not tb or ta == tb:
                continue
            if abs((a.get("page_start") or 0) - (b.get("page_start") or 0)) > 2:
                continue
            ans_a = pick_answer(sentences(clean_text(a["text"]))[:1], clean_text(a["text"]))
            ans_b = pick_answer(sentences(clean_text(b["text"]))[:1], clean_text(b["text"]))
            items.append(
                {
                    "question": (
                        f"En t'appuyant sur le DRR {year}, relie les règles "
                        f"concernant « {ta} » et « {tb} » : que faut-il "
                        f"retenir conjointement ?"
                    ),
                    "gold_answer": f"[{ta}] {ans_a[:280]} || [{tb}] {ans_b[:280]}",
                    "question_type": "multi_chunk",
                    "difficulty": "hard",
                    "requires_year": True,
                    "gold_chunk_ids": [a["chunk_id"], b["chunk_id"]],
                    "document_id": doc,
                    "section_path": b.get("section_path") or [],
                    "page_start": a.get("page_start"),
                    "page_end": b.get("page_end"),
                    "answerable": True,
                    "notes": "Requires retrieving both neighboring chunks.",
                }
            )
    RNG.shuffle(items)
    return items[:limit]


def pdf_paths() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for p in RAW_DIR.glob("*.PDF"):
        mapping[p.stem.lower()] = p
    for p in RAW_DIR.glob("*.pdf"):
        mapping[p.stem.lower()] = p
    return mapping


def pdf_page_text(pdf_path: Path, page_num: int) -> str:
    """1-indexed page number as stored in chunks."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    try:
        idx = page_num - 1
        if idx < 0 or idx >= doc.page_count:
            return ""
        return doc.load_page(idx).get_text("text")
    finally:
        doc.close()


def check_item_against_pdf(
    item: dict[str, Any],
    pdfs: dict[str, Path],
) -> str:
    """
    Return 'pass' | 'fail' | 'skip'.
    Uses token overlap between gold_answer and raw PDF page text.
    """
    if not item.get("answerable", True):
        return "skip"
    doc_id = item.get("document_id")
    page = item.get("page_start")
    if not doc_id or page is None:
        return "skip"
    pdf = pdfs.get(str(doc_id).lower())
    if not pdf:
        return "skip"
    try:
        page_text = pdf_page_text(pdf, int(page)).lower()
    except Exception:
        return "skip"
    if len(page_text) < 40:
        return "skip"
    ans = item.get("gold_answer", "").lower()
    toks = [t for t in re.findall(r"[a-zà-ÿ0-9%]{5,}", ans, flags=re.I)][:15]
    if not toks:
        return "skip"
    hit = sum(1 for t in toks if t in page_text)
    return "pass" if hit / len(toks) >= 0.35 else "fail"


def generate(
    *,
    max_per_chunk: int = 4,
    cross_year_limit: int = 280,
    multi_chunk_limit: int = 400,
    negative_n: int = 180,
    pdf_check_sample: int = 0,
) -> list[dict[str, Any]]:
    chunks = load_chunks()
    index = DedupeIndex()
    raw: list[dict[str, Any]] = []

    # Per-chunk diverse candidates
    for chunk in chunks:
        kept = 0
        for cand in candidates_for_chunk(chunk):
            if kept >= max_per_chunk:
                break
            if not index.accept(cand["question"], cand["question_type"]):
                continue
            raw.append(
                {
                    **cand,
                    "gold_chunk_ids": [cand["chunk"]["chunk_id"]],
                    "document_id": cand["chunk"]["document_id"],
                    "section_path": cand["chunk"].get("section_path") or [],
                    "page_start": cand["chunk"].get("page_start"),
                    "page_end": cand["chunk"].get("page_end"),
                    "answerable": True,
                }
            )
            kept += 1

    for item in build_cross_year(chunks, limit=cross_year_limit):
        if index.accept(item["question"], item["question_type"]):
            raw.append(item)

    for item in build_multi_chunk(chunks, limit=multi_chunk_limit):
        if index.accept(item["question"], "multi_chunk"):
            # multi_chunk shares factual_detail / hard family; use type name
            raw.append(item)

    for item in build_negatives(negative_n):
        if index.accept(item["question"], item["question_type"]):
            raw.append(item)

    RNG.shuffle(raw)

    pdfs = pdf_paths()
    # Optional PDF spot-check on a sample of answerable items
    checked = 0
    if pdf_check_sample > 0 and pdfs:
        answerable_idx = [i for i, it in enumerate(raw) if it.get("answerable", True)]
        sample_idx = set(RNG.sample(answerable_idx, k=min(pdf_check_sample, len(answerable_idx))))
    else:
        sample_idx = set()

    final: list[dict[str, Any]] = []
    pdf_stats = Counter()
    for i, item in enumerate(raw, start=1):
        pdf_status = None
        if (i - 1) in sample_idx:
            pdf_status = check_item_against_pdf(item, pdfs)
            pdf_stats[pdf_status] += 1
            checked += 1
        chunk_ref = None
        final.append(
            make_item(
                eval_id=f"eval_{i:04d}",
                question=item["question"],
                gold_chunk_ids=item.get("gold_chunk_ids") or [],
                gold_answer=item["gold_answer"],
                question_type=item["question_type"],
                difficulty=item["difficulty"],
                requires_year=item.get("requires_year", True),
                answerable=item.get("answerable", True),
                chunk=None,
                notes=item.get("notes", ""),
                pdf_check=pdf_status,
            )
            | {
                k: item[k]
                for k in (
                    "document_id",
                    "section_path",
                    "page_start",
                    "page_end",
                )
                if k in item and item[k] is not None
            }
        )

    # Drop PDF-fail items if we checked them (keep only pass/skip)
    if sample_idx:
        cleaned: list[dict[str, Any]] = []
        for it in final:
            if it.get("pdf_check") == "fail":
                continue
            cleaned.append(it)
        # re-id
        for i, it in enumerate(cleaned, start=1):
            it["id"] = f"eval_{i:04d}"
        final = cleaned

    return final, index, pdf_stats


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-chunk", type=int, default=4)
    parser.add_argument("--cross-year-limit", type=int, default=280)
    parser.add_argument("--multi-chunk-limit", type=int, default=400)
    parser.add_argument("--negative-n", type=int, default=180)
    parser.add_argument("--pdf-check-sample", type=int, default=250)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    items, index, pdf_stats = generate(
        max_per_chunk=args.max_per_chunk,
        cross_year_limit=args.cross_year_limit,
        multi_chunk_limit=args.multi_chunk_limit,
        negative_n=args.negative_n,
        pdf_check_sample=args.pdf_check_sample,
    )
    write_jsonl(items, args.out)

    print(f"Wrote {len(items)} eval items -> {args.out}")
    print("by type:", dict(Counter(i["question_type"] for i in items)))
    print("by doc:", dict(Counter(i.get("document_id", "none") for i in items)))
    print("by difficulty:", dict(Counter(i["difficulty"] for i in items)))
    print("frame caps used:", dict(index.frames))
    print("exact unique keys:", len(index.exact))
    print(
        "rejected exact/near/frame:",
        index.rejected_exact,
        index.rejected_near,
        index.rejected_frame,
    )
    answerable = sum(1 for i in items if i.get("answerable", True))
    print(f"answerable={answerable} negative={len(items) - answerable}")
    if pdf_stats:
        print("pdf_check_sample:", dict(pdf_stats))


if __name__ == "__main__":
    main()
