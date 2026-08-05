"""
Curated RAG eval set — natural questions, Qdrant-reachable gold ids only.

Quality bar (stricter than the synthetic bulk set):
- gold_chunk_ids ⊆ Qdrant payload['id'] (DRR_SNCF / drr-2025 today)
- gold_answer entailed by chunk (token overlap)
- question intent must match answer cues (deadline→dates, amount→€/%, …)
- natural operator phrasing; reject stiff templates
- exact + near-dedupe

  python -m rag_sn_in.eval.generate_curated_eval --target 180
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks" / "drr-2025.jsonl"
OUT_PATH = ROOT / "data" / "eval" / "rag_eval_set.jsonl"
INDEXED_IDS_PATH = ROOT / "data" / "eval" / "_indexed_chunk_ids.txt"

MIN_TOKENS = 100
MAX_ANSWER = 380
RNG = random.Random(13)

SPACE_RE = re.compile(r"\s+")
HEADER_RE = re.compile(r"\[section_header\]\s*(.+?)(?:\n|$)", re.I)
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,5})\b")

STOP_ACRO = {
    "DRR", "RFN", "SNCF", "UE", "PDF", "HTTP", "HTTPS", "ART", "ETC", "NB",
    "ATE", "STI", "RINF", "CE", "AN", "OU", "DE", "LA", "LE", "ET", "EN",
}

BAD_Q_SUBSTR = [
    "résume les règles",
    "relie les règles",
    "est-il exact que",
    "que changerait la lecture",
    "mobilise-t-il",
    "dispositions concrètes",
    "document de référence du réseau",
    "éléments / catégories",
    "cas de figure déclenchent",
]


def norm(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()


def clean(text: str) -> str:
    text = text.replace("\uf0b7", "-").replace("\u00a0", " ")
    text = re.sub(r"\[TABLE\]", "", text, flags=re.I)
    text = re.sub(r"\[section_header\]\s*", "", text, flags=re.I)
    return norm(text)


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zà-ÿ0-9%]{3,}", s.lower()))


def content_tokens(s: str) -> set[str]:
    stop = {
        "dans", "des", "les", "une", "que", "qui", "est", "sont", "avec", "par",
        "sur", "aux", "du", "de", "la", "le", "et", "ou", "au", "en", "ce",
        "ces", "cette", "pour", "pas", "plus", "dont", "ainsi", "tout", "tous",
        "drr", "2025", "sncf", "réseau", "selon", "être", "fait",
    }
    return {t for t in tokens(s) if t not in stop and len(t) > 3}


def answer_supported(answer: str, chunk_text: str, min_ratio: float = 0.78) -> bool:
    a = tokens(answer)
    if len(a) < 6:
        return False
    return (len(a & tokens(chunk_text)) / len(a)) >= min_ratio


def topic_in_answer(topic: str, answer: str, min_hits: int = 2) -> bool:
    tt = content_tokens(topic)
    if not tt:
        return False
    at = content_tokens(answer)
    return len(tt & at) >= min(min_hits, max(1, len(tt) // 2 + 1))


def load_indexed_ids() -> set[str]:
    if INDEXED_IDS_PATH.exists():
        return {
            ln.strip()
            for ln in INDEXED_IDS_PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
    ids: set[str] = set()
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["chunk_id"])
    return ids


def load_chunks(indexed: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["chunk_id"] not in indexed:
                continue
            if c.get("token_count", 0) < MIN_TOKENS:
                continue
            # skip pure tiny / table-only noise
            if c["text"].count("|") > 40 and len(clean(c["text"])) < 200:
                continue
            out.append(c)
    return out


def section_meta(chunk: dict[str, Any]) -> tuple[str | None, str | None]:
    title = None
    for part in reversed(chunk.get("section_path") or []):
        if part.startswith("[section_header]"):
            title = part.replace("[section_header]", "", 1).strip()
            break
    if not title:
        # Clean (tag-free) section paths: last part that is not a chapter label
        for part in reversed(chunk.get("section_path") or []):
            part = part.strip()
            if part and not part.upper().startswith("CHAPITRE"):
                title = part
                break
    if not title:
        m = HEADER_RE.search(chunk.get("text") or "")
        title = m.group(1).strip() if m else None
    if not title:
        return None, None
    title = re.sub(r"^[\u2022\u25cf●•\-–—]+\s*", "", title).strip()
    m = SECTION_RE.match(title)
    if m:
        return m.group(1), m.group(2).strip()
    return None, title


def sentences(text: str) -> list[str]:
    out = []
    for s in SENT_RE.split(text):
        s = norm(s)
        if len(s) < 55 or s.count("|") >= 3:
            continue
        if s.lower().startswith("chapitre"):
            continue
        out.append(s)
    return out


def nice_topic(label: str | None) -> str | None:
    if not label:
        return None
    label = norm(label)
    if len(label) < 10 or len(label) > 80:
        return None
    if re.match(r"^(\d+|[ivx]+|[a-z])[.)]\s*", label, re.I):
        return None
    if GENERIC_TOPICS.search(label):
        return None
    if label.isupper() and len(label) > 12:
        label = label.capitalize()
    return label


def with_article(noun_phrase: str) -> str:
    """Light French article helper for readability."""
    np = noun_phrase.strip()
    if re.match(r"^(le|la|les|l'|un|une|des|du|de la)\b", np, re.I):
        return np
    words = np.split()
    first = words[0].lower() if words else ""
    # plural
    if first.endswith("s") and not first.endswith(("is", "us", "rs")):
        return f"les {np}"
    # common feminine endings in admin French
    if re.search(
        r"(?i)(tion|sion|té|tude|ence|ance|ure|ade|ée|elle|ère)\b",
        first,
    ) or re.search(
        r"(?i)(redevance|capacité|demande|ouverture|priorité|"
        r"description|désignation|composition|acceptation|"
        r"renonciation|prestation|circulation)$",
        first,
    ):
        return f"la {np}"
    if first.startswith(("a", "e", "i", "o", "u", "é", "è", "à", "â", "ê", "î", "ô", "ù", "h")):
        return f"l'{np}"
    return f"le {np}"


GENERIC_TOPICS = re.compile(
    r"(?i)^(introduction|objectif|description du r[ée]seau|"
    r"principes g[ée]n[ée]raux|chapitre\b|g[ée]n[ée]ralit[ée]s)$"
)


def q_key(q: str) -> str:
    q = re.sub(r"[^\wà-ÿ%]+", " ", q.lower())
    return norm(q)


def verify_in_qdrant(chunk_ids: list[str]) -> dict[str, bool]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from rag_sn_in.database.client import get_client

    client = get_client()
    out: dict[str, bool] = {}
    for cid in chunk_ids:
        hits, _ = client.scroll(
            collection_name="DRR_SNCF",
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=cid))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        out[cid] = bool(hits)
    return out


# Seeded natural questions for known DRR entities (filled if chunk matches)
SEED_QUERIES: list[tuple[str, re.Pattern[str], str, str]] = [
    (
        "Comment demander une acceptation de non-conformité (DANC) avant départ ?",
        re.compile(r"(?i)6\.2\.2\.2|demande d'acceptation de non-conformit"),
        "process",
        "medium",
    ),
    (
        "C'est quoi une DANC ?",
        re.compile(r"(?i)demande d'acceptation de non-conformit[ée]\s*\(DANC\)"),
        "definition",
        "easy",
    ),
    (
        "À quoi sert CIS pour un sillon international ?",
        re.compile(r"(?i)Charging Information System|\bCIS\b.{0,40}sillon"),
        "system",
        "easy",
    ),
    (
        "Comment est calculée la redevance de circulation électrique (RCE) ?",
        re.compile(r"(?i)5\.3\.2 REDEVANCE DE CIRCULATION ÉLECTRIQUE|\bRCE\b"),
        "amount",
        "hard",
    ),
    (
        "C'est quoi un accord-cadre voyageurs ?",
        re.compile(r"(?i)accord-cadre\s+voyageurs"),
        "definition",
        "easy",
    ),
    (
        "Que faire en cas de perturbation internationale sur le réseau ?",
        re.compile(r"(?i)perturbation internationale"),
        "condition",
        "hard",
    ),
    (
        "Où envoyer une demande de relevage ?",
        re.compile(r"(?i)relevage\.dcf@sncf\.fr|demande.{0,40}relevage"),
        "contact",
        "easy",
    ),
    (
        "Quel est l'objectif du Document de référence du réseau ?",
        re.compile(r"(?i)1\.2\s+OBJECTIF"),
        "definition",
        "easy",
    ),
    (
        "Comment commande-t-on un sillon national dans GESICO ?",
        re.compile(r"(?i)demande de sillon national.{0,40}GESICO|via l'interface unifi[ée]e GESICO"),
        "process",
        "medium",
    ),
    (
        "Comment renoncer à un sillon attribué ?",
        re.compile(r"(?i)[Rr]enonciation.{0,40}sillon|renoncer.{0,30}sillon"),
        "process",
        "medium",
    ),
    (
        "C'est quoi la redevance de marché (RM) ?",
        re.compile(r"(?i)5\.3\.4 REDEVANCE DE MARCH|\bredevance de march[ée]\b"),
        "amount",
        "medium",
    ),
    (
        "Qui assure la gestion opérationnelle des circulations ?",
        re.compile(r"(?i)(SGC|service de gestion des circulations).{0,80}(gestion op[ée]rationnelle|circulations)"),
        "actor",
        "easy",
    ),
    (
        "Comment déclarer la composition réelle du convoi ?",
        re.compile(r"(?i)composition r[ée]elle du convoi"),
        "process",
        "medium",
    ),
    (
        "Que dit le DRR sur les gares de voyageurs ?",
        re.compile(r"(?i)7\.2\.1 GARES DE VOYAGEURS|voies des gares de voyageurs font partie"),
        "factual",
        "easy",
    ),
    (
        "C'est quoi DINAMIC ?",
        re.compile(r"(?i)\bDINAMIC\b"),
        "system",
        "easy",
    ),
    (
        "C'est quoi GESICO ?",
        re.compile(r"(?i)interface unifi[ée]e GESICO|GESICO-\s*DSDM"),
        "system",
        "easy",
    ),
    (
        "Comment obtenir une attestation d'assurance pour accéder au réseau ?",
        re.compile(r"(?i)attestation d'assurance"),
        "process",
        "medium",
    ),
    (
        "Quand une ligne est-elle déclarée saturée ?",
        re.compile(r"(?i)d[ée]clar(?:ation|ée) satur"),
        "condition",
        "hard",
    ),
    (
        "Comment est calculée la redevance de saturation (RS) ?",
        re.compile(r"(?i)redevance de saturation|\bRS\b"),
        "amount",
        "hard",
    ),
    (
        "Qu'est-ce qu'une demande de sillon de dernière minute ?",
        re.compile(r"(?i)sillons? de derni[eè]re minute"),
        "definition",
        "medium",
    ),
    (
        "Comment graisser les rails avec le matériel roulant selon le DRR ?",
        re.compile(r"(?i)graissage des rails"),
        "process",
        "medium",
    ),
    (
        "C'est quoi la prestation de remisage de nuit en gare ?",
        re.compile(r"(?i)remisage de mat[ée]riel roulant voyageurs de nuit"),
        "factual",
        "medium",
    ),
    (
        "Quel abattement SNCF Réseau applique-t-il sur les fenêtres travaux ?",
        re.compile(r"(?i)5\.6\.4\.5 Abattement|Abattement SNCF"),
        "amount",
        "hard",
    ),
    (
        "Que doit contenir une demande d'ouverture supplémentaire de ligne ou gare ?",
        re.compile(r"(?i)ouvertures? suppl[ée]mentaires? de lignes"),
        "process",
        "medium",
    ),
    (
        "À quoi sert le correspondant opérationnel désigné par l'EF ?",
        re.compile(r"(?i)correspondant op[ée]rationnel"),
        "actor",
        "medium",
    ),
    (
        "Quelles sont les conditions de remorque à respecter pour un train ?",
        re.compile(r"(?i)conditions de remorque"),
        "obligation",
        "medium",
    ),
    (
        "Comment contacter les services de communications ferroviaires (TEL) ?",
        re.compile(r"(?i)communications ferroviaires \(TEL\)|services de communications ferroviaires"),
        "contact",
        "easy",
    ),
    (
        "C'est quoi une situation de crise ferroviaire ?",
        re.compile(r"(?i)situation de crise ferroviaire"),
        "definition",
        "medium",
    ),
    (
        "Quel montant d'aide pour la redevance de marché sur LGV vs ligne classique ?",
        re.compile(r"(?i)montant de l'aide est égal|10%\s+de la redevance de march"),
        "amount",
        "hard",
    ),
    (
        "Faut-il une licence et un certificat de sécurité pour accéder au RFN ?",
        re.compile(r"(?i)certificat de s[ée]curit|licence.{0,40}entreprise ferroviaire"),
        "obligation",
        "medium",
    ),
]


def pick_supporting_sentence(sents: list[str], pattern: re.Pattern[str]) -> str | None:
    hits = [s for s in sents if pattern.search(s)]
    if not hits:
        return None
    # prefer longer explanatory sentences
    return max(hits, key=len)


def is_low_quality_chunk(chunk: dict[str, Any]) -> bool:
    text = chunk.get("text") or ""
    # table-of-contents / sigle glossary rows dominate
    if text.count("|") >= 8 and text.count("....") >= 3:
        return True
    if text.count("|") >= 20:
        return True
    return False


def score_seed_match(chunk: dict[str, Any], pattern: re.Pattern[str], question: str) -> tuple[float, str]:
    """Return (score, supporting_answer). Higher is better."""
    if is_low_quality_chunk(chunk):
        return (-1.0, "")
    sents = sentences(clean(chunk["text"]))
    if not sents:
        return (-1.0, "")
    hit = pick_supporting_sentence(sents, pattern)
    if not hit:
        return (-1.0, "")
    if not answer_supported(hit, chunk["text"], min_ratio=0.75):
        return (-1.0, "")
    # question content words should appear in chunk
    q_words = content_tokens(question)
    c_words = content_tokens(chunk["text"])
    overlap = len(q_words & c_words) / max(len(q_words), 1)
    if overlap < 0.25:
        return (-1.0, "")
    score = overlap * 10 + min(len(hit), 300) / 100 + chunk.get("token_count", 0) / 500
    # boost if pattern hits near start (often the defining section)
    pos = pattern.search(chunk["text"])
    if pos and pos.start() < 180:
        score += 2.0
    return (score, hit)


def build_seed_items(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used_q: set[str] = set()

    for question, pattern, qtype, diff in SEED_QUERIES:
        if q_key(question) in used_q:
            continue
        best: tuple[float, dict[str, Any], str] | None = None
        for chunk in chunks:
            if not pattern.search(chunk["text"]):
                continue
            score, answer = score_seed_match(chunk, pattern, question)
            if score < 0:
                continue
            # intent checks
            if qtype == "contact":
                emails = EMAIL_RE.findall(chunk["text"])
                if not emails:
                    continue
                email_sent = next((s for s in sentences(clean(chunk["text"])) if EMAIL_RE.search(s)), None)
                if email_sent:
                    answer = email_sent
                elif emails[0] not in answer:
                    answer = f"{answer} Contacter : {emails[0]}."
            if qtype == "amount" and not re.search(
                r"(?i)(redevance|tarif|€|euro|\d+\s*%|factur|aide)", answer
            ):
                continue
            if best is None or score > best[0]:
                best = (score, chunk, answer)

        if best is None:
            continue
        _, chunk, answer = best
        items.append(
            {
                "question": question,
                "gold_answer": norm(answer)[:MAX_ANSWER],
                "gold_chunk_ids": [chunk["chunk_id"]],
                "document_id": chunk["document_id"],
                "section_path": chunk.get("section_path") or [],
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "question_type": qtype,
                "difficulty": diff,
                "requires_year": True,
                "answerable": True,
                "notes": "seed-natural; human-phrased; best-chunk-match",
            }
        )
        used_q.add(q_key(question))
    return items


def build_candidates(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []

    def add(
        chunk: dict[str, Any],
        question: str,
        answer: str,
        qtype: str,
        difficulty: str,
        notes: str = "",
    ) -> None:
        question, answer = norm(question), norm(answer)[:MAX_ANSWER]
        if len(question) < 18 or len(answer) < 50:
            return
        ql = question.lower()
        if any(b in ql for b in BAD_Q_SUBSTR):
            return
        # reject long/awkward auto topics
        if question.count(" ") > 18:
            return
        if re.search(r"(?i)c'est quoi (les )?articles |c'est quoi [A-Z]{4,} \?", question):
            return
        if re.search(r"(?i)\b(cadre|notion|principes)\s*\?$", question):
            return
        if not answer_supported(answer, chunk["text"]):
            return
        cands.append(
            {
                "question": question,
                "gold_answer": answer,
                "gold_chunk_ids": [chunk["chunk_id"]],
                "document_id": chunk["document_id"],
                "section_path": chunk.get("section_path") or [],
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "question_type": qtype,
                "difficulty": difficulty,
                "requires_year": True,
                "answerable": True,
                "notes": notes or "curated-natural; indexed drr-2025",
            }
        )

    for chunk in chunks:
        cleaned = clean(chunk["text"])
        sents = sentences(cleaned)
        if not sents:
            continue
        _, sec_label = section_meta(chunk)
        topic = nice_topic(sec_label)
        topic_l = topic.lower() if topic else None

        # definitions
        for sent in sents:
            if not re.search(
                r"(?i)(s['’]entend|d[ée]signe|correspond\s+[àa]|est\s+entendue?|"
                r"on\s+entend\s+par)",
                sent,
            ):
                continue
            if topic and topic_in_answer(topic, sent):
                add(
                    chunk,
                    f"C'est quoi {with_article(topic_l)} ?",
                    sent,
                    "definition",
                    "easy",
                )
                add(
                    chunk,
                    f"Que signifie {with_article(topic_l)} pour un candidat au réseau ?",
                    sent,
                    "definition",
                    "medium",
                )
            break

        # deadlines — answer must contain time cue
        for sent in sents:
            if not re.search(
                r"(?i)(\d+\s*(?:jours?|heures?|mois)|d[ée]lai|au plus tard|J-\d|A-\d)",
                sent,
            ):
                continue
            if topic and topic_in_answer(topic, sent):
                add(
                    chunk,
                    f"Quel délai pour {topic_l} ?",
                    sent,
                    "deadline",
                    "hard",
                )
                add(
                    chunk,
                    f"Avant quand faut-il s'occuper de {topic_l} ?",
                    sent,
                    "deadline",
                    "hard",
                )
            break

        # amounts — answer must look financial
        for sent in sents:
            if not re.search(
                r"(?i)(redevance|tarif|€|euros?|\d+[.,]?\d*\s*%|factur[ée])",
                sent,
            ):
                continue
            if topic and topic_in_answer(topic, sent):
                add(
                    chunk,
                    f"Comment est facturée {with_article(topic_l)} ?",
                    sent,
                    "amount",
                    "hard",
                )
                add(
                    chunk,
                    f"Quelles règles tarifaires pour {topic_l} ?",
                    sent,
                    "amount",
                    "hard",
                )
            break

        # process / obligations
        for sent in sents:
            if not re.search(r"(?i)\b(doit|doivent|est tenu)\b", sent):
                continue
            if topic and topic_in_answer(topic, sent):
                add(
                    chunk,
                    f"Comment faire pour {topic_l} ?",
                    sent,
                    "process",
                    "medium",
                )
                add(
                    chunk,
                    f"Qu'est-ce qui est obligatoire concernant {topic_l} ?",
                    sent,
                    "obligation",
                    "medium",
                )
            break

        # conditions
        for sent in sents:
            if not re.search(r"(?i)(en cas de|lorsque|d[èe]s lors que)", sent):
                continue
            if topic and topic_in_answer(topic, sent):
                add(
                    chunk,
                    f"Que se passe-t-il en cas de souci avec {topic_l} ?",
                    sent,
                    "condition",
                    "hard",
                )
            break

        # contacts
        emails = EMAIL_RE.findall(chunk["text"])
        if emails:
            sent = next((s for s in sents if EMAIL_RE.search(s)), None)
            if sent and topic:
                add(
                    chunk,
                    f"À quelle adresse e-mail écrire pour {topic_l} ?",
                    sent,
                    "contact",
                    "easy",
                )

        # well-supported acronyms with explanatory sentence
        # require either (ACRO) expansion pattern or >=2 occurrences
        for acro in ACRONYM_RE.findall(chunk["text"]):
            if acro in STOP_ACRO or len(acro) < 3:
                continue
            # skip Title-Case-looking ALLCAPS French words used as headers
            if acro in {
                "NOTION", "ACCEPTATION", "DESCRIPTION", "INTRODUCTION",
                "OBJECTIF", "PRINCIPES", "CONDITIONS", "SERVICES",
            }:
                continue
            occ = len(re.findall(rf"\b{acro}\b", chunk["text"]))
            has_expansion = bool(
                re.search(rf"\b{acro}\b\s*[:(]|[:(]\s*{acro}\b", chunk["text"])
            )
            if occ < 2 and not has_expansion:
                continue
            hit = next(
                (
                    s
                    for s in sents
                    if re.search(rf"\b{acro}\b", s)
                    and re.search(
                        r"(?i)(est une?|est le|est la|permet|d[ée]signe|"
                        r"correspond|application|outil|syst[èe]me|processus|"
                        r"interface|via)",
                        s,
                    )
                    and len(s) > 80
                ),
                None,
            )
            if not hit:
                continue
            add(
                chunk,
                f"C'est quoi {acro} ?",
                hit,
                "definition",
                "easy",
            )
            add(
                chunk,
                f"{acro} sert à quoi ?",
                hit,
                "system",
                "medium",
            )
            break

        # natural factual (only if topic words appear in lead sentence)
        if topic and sents and topic_in_answer(topic, sents[0]):
            add(
                chunk,
                f"Tu peux m'expliquer {with_article(topic_l)} ?",
                sents[0],
                "factual",
                "easy",
            )
            add(
                chunk,
                f"Qu'est-ce que je dois savoir sur {topic_l} ?",
                sents[0],
                "factual",
                "easy",
            )

    return cands


def build_negatives(n: int = 20) -> list[dict[str, Any]]:
    qs = [
        "Comment renouveler mon permis de conduire B ?",
        "Quel est le prix d'un abonnement Navigo ?",
        "Comment déposer une marque à l'INPI ?",
        "Comment déclarer mes revenus fonciers ?",
        "Quels sont les horaires du Louvre le mardi ?",
        "Comment contester une amende de stationnement ?",
        "Comment immatriculer une voiture d'occasion ?",
        "C'est quoi la procédure DALO ?",
        "Comment ouvrir un compte-titres ?",
        "Quelle franchise d'assurance habitation est typique ?",
        "Comment résilier un forfait mobile ?",
        "Qui siège au Conseil constitutionnel ?",
        "Comment obtenir un Kbis en ligne ?",
        "Quelles aides pour isoler des combles ?",
        "Comment adhérer à une mutuelle d'entreprise ?",
        "Quel vaccin grippe la HAS recommande-t-elle ?",
        "Comment obtenir une licence de pêche ?",
        "Quelles règles FIFA pour un transfert ?",
        "Comment homologuer un drone de loisir ?",
        "Comment fonctionne le crowdfunding immobilier ?",
    ]
    return [
        {
            "question": q,
            "gold_answer": (
                "Hors périmètre du DRR SNCF Réseau — ne pas inventer "
                "une réponse à partir du corpus."
            ),
            "gold_chunk_ids": [],
            "question_type": "negative",
            "difficulty": "easy",
            "requires_year": False,
            "answerable": False,
            "notes": "curated-negative",
        }
        for q in qs[:n]
    ]


def select_set(cands: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    # Prefer seed notes first
    cands = sorted(
        cands,
        key=lambda x: (0 if "seed" in x.get("notes", "") else 1, RNG.random()),
    )
    caps = {
        "definition": 35,
        "process": 30,
        "obligation": 20,
        "deadline": 25,
        "amount": 25,
        "system": 20,
        "contact": 15,
        "condition": 20,
        "factual": 30,
        "actor": 15,
        "negative": 20,
    }
    counts: Counter[str] = Counter()
    seen_q: set[str] = set()
    seen_ct: set[tuple[str, str]] = set()
    chosen: list[dict[str, Any]] = []

    for cand in cands:
        if len(chosen) >= target:
            break
        qk = q_key(cand["question"])
        if qk in seen_q:
            continue
        qt = cand["question_type"]
        if counts[qt] >= caps.get(qt, 20):
            continue
        cid = (cand.get("gold_chunk_ids") or ["none"])[0]
        if cid != "none" and (cid, qt) in seen_ct:
            continue
        qtoks = tokens(cand["question"])
        if any(
            len(qtoks & tokens(p["question"])) / (len(qtoks | tokens(p["question"])) or 1)
            >= 0.8
            for p in chosen[-60:]
        ):
            continue
        seen_q.add(qk)
        if cid != "none":
            seen_ct.add((cid, qt))
        counts[qt] += 1
        chosen.append(cand)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--skip-qdrant-check", action="store_true")
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Keep only hand-phrased seed questions (+ negatives).",
    )
    args = parser.parse_args()

    indexed = load_indexed_ids()
    chunks = load_chunks(indexed)
    print(f"indexed={len(indexed)} usable_chunks={len(chunks)}")

    seeds = build_seed_items(chunks)
    auto = [] if args.gold_only else build_candidates(chunks)
    print(f"seeds={len(seeds)} auto_cands={len(auto)}")

    if args.gold_only:
        positives = seeds
    else:
        positives = select_set(seeds + auto, target=args.target)
    items = positives + build_negatives(args.negatives)
    RNG.shuffle(items)

    gold_ids = sorted({c for it in items for c in it.get("gold_chunk_ids") or []})
    if not args.skip_qdrant_check and gold_ids:
        print(f"Qdrant-checking {len(gold_ids)} gold ids…")
        status = verify_in_qdrant(gold_ids)
        missing = {cid for cid, ok in status.items() if not ok}
        if missing:
            print(f"drop missing: {len(missing)}")
            items = [
                it
                for it in items
                if not it.get("gold_chunk_ids")
                or all(c not in missing for c in it["gold_chunk_ids"])
            ]
        else:
            print("all gold ids OK in payload['id']")

    # final support re-check against source chunks
    by_id = {c["chunk_id"]: c for c in chunks}
    kept: list[dict[str, Any]] = []
    for it in items:
        if not it.get("answerable", True):
            kept.append(it)
            continue
        cid = it["gold_chunk_ids"][0]
        ch = by_id.get(cid)
        if not ch or not answer_supported(it["gold_answer"], ch["text"]):
            continue
        kept.append(it)
    items = kept

    final = []
    for i, it in enumerate(items, start=1):
        row = {
            "id": f"eval_{i:04d}",
            "question": it["question"],
            "gold_chunk_ids": it.get("gold_chunk_ids") or [],
            "gold_answer": it["gold_answer"],
            "question_type": it["question_type"],
            "difficulty": it["difficulty"],
            "requires_year": it.get("requires_year", True),
            "answerable": it.get("answerable", True),
            "notes": it.get("notes", "curated"),
            "collection": "DRR_SNCF",
            "match_field": "payload.id",
        }
        for k in ("document_id", "section_path", "page_start", "page_end"):
            if k in it:
                row[k] = it[k]
        final.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(final)} -> {args.out}")
    print("types:", dict(Counter(r["question_type"] for r in final)))
    print(
        "answerable",
        sum(1 for r in final if r["answerable"]),
        "negative",
        sum(1 for r in final if not r["answerable"]),
        "seed",
        sum(1 for r in final if "seed" in r.get("notes", "")),
    )

    print("\n=== AUDIT SAMPLE ===")
    for row in RNG.sample([r for r in final if r["answerable"]], k=min(10, len(final))):
        print(f"\n[{row['id']}] {row['question_type']} | {row['gold_chunk_ids']}")
        print("Q:", row["question"])
        print("A:", row["gold_answer"][:170])


if __name__ == "__main__":
    main()
