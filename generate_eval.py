#!/usr/bin/env python3
"""Deterministic extractive French DRR RAG evaluation-set generator."""
import json, re, random, unicodedata
from pathlib import Path
from collections import Counter, defaultdict

INPUT_DIR = Path('/mnt/user-data/uploads')
OUTPUT_DIR = Path('/mnt/user-data/outputs')
SEED = 20260804
TARGET = 500

# Sentence-like units preserve exact source substrings for answer/support validation.
def norm(s):
    s = unicodedata.normalize('NFKC', s).replace('\u00a0', ' ')
    return re.sub(r'\s+', ' ', s).strip().casefold()

def units(text):
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    out=[]
    for x in raw:
        x=re.sub(r'\s+', ' ', x).strip(' \t\r\n')
        if len(x) >= 12 and not re.fullmatch(r'[\d\s./>-]+', x): out.append(x)
    return out

def short_support(s, answer=None):
    s=s.strip()
    if len(s)<=600: return s
    # Prefer a complete clause containing the answer, still an exact substring.
    parts=re.split(r'\s*[;:]\s*', s)
    if answer:
        for p in parts:
            if norm(answer) in norm(p) and len(p)<=600: return p.strip()
    for p in parts:
        if len(p)<=600: return p.strip()
    return s[:600].rsplit(' ',1)[0]

def extract_candidates(ch):
    text=ch['text']; ss=units(text); candidates=[]
    # Always-cover structural item. The section label is present in the chunk header.
    heading = ch['section_path'][-1] if ch.get('section_path') else text.splitlines()[0].strip()
    if heading and norm(heading) in norm(text):
        candidates.append(('locational/structural', 'Quelle section du DRR traite ce sujet ?', heading, heading))
    # Definitions / role statements. Full sentence is a faithful answer span.
    for s in ss:
        low=norm(s)
        if re.search(r'\b(est|sont|désigne|design[eé]|signifie|correspond à|constitue|se définit)\b', low) and len(s)<=600:
            candidates.append(('factual/definition', 'Que définit ou précise le document dans cet extrait ?', s, s))
            break
    # Legal references: answer is the exact legal citation phrase in its evidence sentence.
    legal_re = re.compile(r"(?:article\s+(?:L\.?|R\.?|D\.?|2111-9|8\.1)\s*[\w.-]+(?:\s+du\s+code\s+des\s+transports)?|décret\s+n[°º]?\s*[\w.-]+(?:\s+du\s+[^,;.]+)?|loi\s+n[°º]?\s*[\w.-]+(?:\s+du\s+[^,;.]+)?|directive\s+(?:européenne\s+)?[\w/.-]+|règlement\s+(?:UE|européen)[\w/() .-]*)", re.I)
    for s in ss:
        m=legal_re.search(s)
        if m:
            ans=m.group(0).strip(' ,;:.')
            candidates.append(('legal reference lookup', f'Quelle référence juridique est citée dans cet extrait ?', ans, s)); break
    # Numeric / quantitative facts.
    num_re=re.compile(r"(?:\b\d+(?:[,.]\d+)?\s*(?:€|euros?|%|ans?|mois|jours?|heures?|minutes?|km|m|trains?|fois|personnes?|pages?)\b|\bJ[- ]\d+\b|\b\d{1,2}[h:]\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b)", re.I)
    for s in ss:
        m=num_re.search(s)
        if m and not re.match(r'^\d{4}$', m.group(0)):
            ans=m.group(0)
            candidates.append(('numeric/quantitative', 'Quelle valeur chiffrée est indiquée dans cet extrait ?', ans, s)); break
    # Temporal/deadline when date/time/deadline wording appears.
    temp_re=re.compile(r"(?:avant le|au plus tard|à compter du|jusqu'au|date limite|délai|horaire|heures? de service|J[- ]\d+|\d{1,2}[h:]\d{2}|\d{1,2}/\d{1,2}/\d{2,4})", re.I)
    for s in ss:
        m=temp_re.search(s)
        if m:
            # answer a compact exact span, ideally the temporal phrase plus following value
            tail=s[m.start():]
            ans=re.split(r'[,;.]', tail)[0].strip()
            if len(ans)>180: ans=m.group(0)
            candidates.append(('temporal/deadline', 'Quelle échéance ou indication temporelle ressort de cet extrait ?', ans, s)); break
    # Procedural / enumerated action.
    for s in ss:
        if re.match(r'^(?:\d+[°.)]|[a-e][).]|[-–•])\s*', s) and re.search(r'\b(doit|peut|adresse|transmet|dépose|procédure|étape|demande|organise|coordonne)\b', norm(s)):
            candidates.append(('procedural', 'Quelle action ou procédure est décrite dans cet extrait ?', s, s)); break
    # Entity/role.
    for s in ss:
        if re.search(r'\b(SNCF Réseau|ART|candidat|entreprise ferroviaire|gestionnaire d.infrastructure|État|AOT|AOM)\b', s, re.I) and re.search(r'\b(doit|assure|est|organise|coordonne|responsable|incombe|peut|adresse|reçoit|gère)\b', norm(s)) and len(s)<=600:
            candidates.append(('entity/role', 'Quel acteur ou quelle entité joue le rôle indiqué dans cet extrait ?', s, s)); break
    # Fallback factual sentence, guaranteeing an answerable item even for sparse chunks.
    if ss:
        s=next((x for x in ss if not x.startswith(('CHAPITRE ', 'ANNEXE '))), ss[0])
        support=short_support(s)
        candidates.append(('factual/definition', 'Que précise le document dans cet extrait ?', support, support))
    # de-duplicate candidate signatures
    seen=set(); out=[]
    for qtype,q,a,sup in candidates:
        key=(qtype,norm(q),norm(a),norm(sup))
        if key not in seen and a and norm(a) in norm(text): seen.add(key); out.append((qtype,q,a,short_support(sup,a)))
    return out

def difficulty(ch, qtype):
    n=int(ch.get('token_count') or len(ch['text'].split()))
    if qtype in ('legal reference lookup','temporal/deadline','procedural') or n>700: return 'hard'
    if qtype in ('numeric/quantitative','entity/role') or n>350: return 'medium'
    return 'easy'

def main():
    random.seed(SEED)
    files=sorted(INPUT_DIR.glob('drr-*.jsonl'))
    assert len(files)==3, f'Expected 3 drr files, found {len(files)}'
    chunks=[]; bydoc=defaultdict(list); source_files=[]
    for fp in files:
        source_files.append(fp.name)
        with fp.open(encoding='utf-8') as f:
            for line_no,line in enumerate(f,1):
                ch=json.loads(line); assert {'chunk_id','document_id','text','section_path','page_start','page_end'} <= ch.keys(), (fp,line_no)
                chunks.append(ch); bydoc[ch['document_id']].append(ch)
    items=[]; used=set(); serial=0
    # One structural/fallback item per chunk, then one extra fact item on fact-rich chunks.
    for doc in sorted(bydoc):
        for idx,ch in enumerate(bydoc[doc]):
            cands=extract_candidates(ch); assert cands
            # deterministic selection: rotate among available candidates, favor non-structural for variety
            primary=cands[0]
            if len(cands)>1 and idx % 3 != 0: primary=cands[1 + (idx % (len(cands)-1))]
            selected=[primary]
            if len(cands)>2 and (idx % 4 == 0): selected.append(cands[2 + (idx % (len(cands)-2))])
            for qtype,q,a,sup in selected:
                key=(q,ch['chunk_id'])
                if key in used: continue
                used.add(key); serial+=1
                items.append(make_item(serial,q,a,qtype,ch,sup,True))
    # Approximately 5% adversarial items, separate from per-document answerable quotas.
    adv_n=max(1, round(len(items)*0.05)); adv_templates=[
        'Quel est le numéro de téléphone direct du responsable de ce service ?',
        'Quelle est la couleur du formulaire utilisé pour cette démarche ?',
        'Quel est le mot de passe de la plateforme mentionnée dans ce passage ?',
        'Quelle est la date exacte de naissance du directeur cité dans ce passage ?',
        'Quel est le montant de la prime individuelle versée à cet acteur ?',
    ]
    for j in range(adv_n):
        ch=chunks[(j*37+11)%len(chunks)]; q=adv_templates[j%len(adv_templates)]
        # Make each adversarial question unique per chunk pair deterministically.
        if (q,ch['chunk_id']) in used: q=q[:-1]+f' (référence {j}) ?'
        used.add((q,ch['chunk_id'])); serial+=1
        items.append(make_item(serial,q,'Information non disponible dans le document','unanswerable',ch,'',False))
    # Validation.
    source_map={c['chunk_id']:c['text'] for c in chunks}; ids=[x['id'] for x in items]
    assert len(ids)==len(set(ids))
    assert all(x['question'].strip() and x['answer'].strip() for x in items)
    ans_items=[x for x in items if x['is_answerable']]
    grounded=[x for x in ans_items if norm(x['answer']) in norm(source_map[x['chunk_id']])]
    assert len(grounded)==len(ans_items), 'Ungrounded answer found'
    assert all(x['chunk_id'] in source_map for x in items)
    assert len({(x['question'],x['chunk_id']) for x in items})==len(items)
    counts_doc=Counter(x['document_id'] for x in ans_items); assert all(v>=TARGET for v in counts_doc.values()), counts_doc
    meta={'created_at':'2026-08-04T20:54:00Z','description':'Jeu de données doré pour l’évaluation RAG sur les Documents de Référence du Réseau SNCF Réseau.','source_files':source_files,'language':'fr','counts_by_document':dict(sorted(Counter(x['document_id'] for x in items).items())),'answerable_by_document':dict(sorted(counts_doc.items())),'counts_by_question_type':dict(sorted(Counter(x['question_type'] for x in items).items())),'counts_by_difficulty':dict(sorted(Counter(x['difficulty'] for x in items).items())),'total_items':len(items),'generation_method':'Génération extractive déterministe par heuristiques regex, phrases et chemins de section; seed 20260804.','validation notes':'JSON valide; chaque réponse answerable est une sous-chaîne normalisée de son chunk source (100%); couverture de tous les chunks; identifiants uniques; questions non vides; les items unanswerable (~5%) sont comptés hors quotas answerable.'}
    out={'metadata':meta,'items':items}
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    with (OUTPUT_DIR/'eval.json').open('w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print('Final counts table')
    print('document_id\tanswerable\tunanswerable\ttotal')
    for d in sorted(bydoc): print(f"{d}\t{counts_doc[d]}\t{sum(x['document_id']==d and not x['is_answerable'] for x in items)}\t{sum(x['document_id']==d for x in items)}")
    print('total',len(items),'answerable',len(ans_items),'grounded_pct',100*len(grounded)/len(ans_items),'adversarial',adv_n)

def make_item(serial,q,a,qtype,ch,sup,answerable):
    return {'id':f"eval_{serial:06d}",'question':q,'answer':a,'answer_type':'extractive' if answerable else 'unanswerable','question_type':qtype,'difficulty':difficulty(ch,qtype),'document_id':ch['document_id'],'source_year':int(ch['document_id'].split('-')[-1]),'chunk_id':ch['chunk_id'],'section_path':ch['section_path'],'page_start':ch['page_start'],'page_end':ch['page_end'],'supporting_text':sup,'is_answerable':answerable,'multi_hop':False}

if __name__=='__main__': main()
