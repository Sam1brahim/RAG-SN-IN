"""
The needed hierarchy depends on the document itself, there is not universal method, only universal golden rules.

* Keep context linked even in different chunks:

for example: 
chunk 1
    Header1:
        subheader 1
        subheader 2
        subheader 3
chunk 2
    Header1:
        subheader 4
        subheader 5


* Better to chunk semantically related concepts together
* Can chunk together two different sections if first section's content is tiny
* Test token count with the tokenizer
* if small components of content isn't very different in size (paragraphs, articles), they can be used as blocks
for separation.

PSEUDO-CODE:

MAX_TOKENS = 1024
HARD_MAX_TOKENS = 1600

for each document:
    parse document into hierarchical sections

    for each main_header:
        context = [main_header]

        for each subsection under main_header:

            candidate = context + subsection
            candidate_tokens = count_tokens(candidate)

            if candidate_tokens <= MAX_TOKENS:
                context.append(subsection)

            else:
                emit(context)

                if tokens(subsection) <= HARD_MAX_TOKENS:
                    context = [main_header, subsection]
                else:
                    smaller_parts = split_by_paragraphs_tables(subsection)

                    for part in smaller_parts:
                        if tokens(context + part) <= MAX_TOKENS:
                            context.append(part)
                        else:
                            emit(context)
                            context = [main_header, part]

        if context contains content:
            emit(context)
            
"""

MAX_TOKENS = 700 # respecting "embeddinggemma" limits
TOP_K = 3 # provide sufficiently linked chunks for consistent context

from transformers import AutoTokenizer

MODEL_ID = "google/embeddinggemma-300m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = """
1.2 OBJECTIF
Le document de référence du réseau ferré national (DRR) contient les informations nécessaires aux entreprises ferroviaires et autres candidats qui souhaitent utiliser le réseau ferré national (RFN) pour y assurer des prestations de transport de voyageurs et de marchandises et plus généralement à toutes les parties intéressées par le transport ferroviaire.
Tout contrat ou accord commercial conclu avec SNCF Réseau conformément auChapitre 3 - Modalités d'accès au réseau ferré national est établi selon les règles définies dans le présent document.
1.3 ASPECTS LÉGAUX
1.3.1 CADRE JURIDIQUE
Le DRR est notamment basé sur les textes législatifs et réglementaires suivants :
- Règlement n° 913/2010 du 22 septembre 2010 relatif au réseau ferroviaire européen pour un fret compétitif ;
- Directive (UE) 2016/798 du Parlement européen et du Conseil du 11 mai 2016 relative à la sécurité ferroviaire ;
- Directive (UE) 2016/797 du Parlement européen et du Conseil du 11 mai 2016 relative à l'interopérabilité du système ferroviaire au sein de l'Union européenne ;
- Directive 2012/34/UE du 21 novembre 2012 établissant un espace ferroviaire unique européen, et directive 2016/2370/UE du 14 décembre 2016 la modifiant ;
- Code des transports, partie législative ;
- Loi n° 2014-872 du 4 août 2014 portant réforme ferroviaire ;
- Décret n° 97-444 du 5 mai 1997 modifié relatif aux missions et aux statuts de SNCF Réseau ;
- Décret n° 97-446 du 5 mai 1997 modifié relatif aux redevances d’utilisation du réseau ferré national perçues au profit de SNCF Réseau ;
- Décret n° 2003-194 du 7 mars 2003 modifié relatif à l'utilisation du réseau ferré national ;
- Décret n° 2019-525 du 27 mai 2019 modifié relatif à la sécurité des circulations ferroviaires et à l’interopérabilité du système ferroviaire ;
- Décret n° 2012-70 du 20 janvier 2012 relatif aux gares de voyageurs et autres infrastructures de services du réseau ferroviaire ;
- Arrêté du 9 décembre 2021 fixant les objectifs, les méthodes, les indicateurs de sécurité et la réglementation technique de sécurité et d'interopérabilité applicables sur le réseau ferré national ;
- Spécifications techniques d’interopérabilité (STI).
L’ensemble des textes applicables sont consultables sur les sites www.eur-lex.europa.eu (droit européen) et www.legifrance.gouv.fr (droit français).
"""

encoded = tokenizer(
    text,
    add_special_tokens=True,
    return_attention_mask=False,
    return_tensors=None,
)

token_ids = encoded["input_ids"]
token_count = len(token_ids)

print(f"Token count: {token_count}")
print(f"Fits under 2,000 tokens: {token_count <= 2000}")