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


"""

MAX_TOKENS = 700 # respecting "embeddinggemma" limits
TOP_K = 3 # provide sufficiently linked chunks for consistent context

from transformers import AutoTokenizer

MODEL_ID = "google/embeddinggemma-300m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = """
CHAPITRE 1 INFORMATIONS GÉNÉRALES
1.1 INTRODUCTION
Conformément à l'article 2111-9 du code des transports, « La société SNCF Réseau a pour mission d'assurer, de façon transparente et non discriminatoire, directement ou par l'intermédiaire de filiales, conformément aux principes du service public et dans le but de promouvoir le transport ferroviaire en France dans un objectif de développement durable, d'aménagement du territoire et d'efficacité économique et sociale :

1°) L'accès à l'infrastructure ferroviaire du réseau ferré national, comprenant la répartition des capacités et la tarification de cette infrastructure ;
2°) La gestion opérationnelle des circulations sur le réseau ferré national ;
3°) La maintenance, comprenant l'entretien et le renouvellement, de l'infrastructure du réseau ferré national ;
4°) Le développement, l'aménagement, la cohérence et la mise en valeur du réseau ferré national ;
5°) La gestion unifiée des gares de voyageurs, à travers une filiale dotée d'une autonomie organisationnelle, décisionnelle et financière ;
6°) La gestion et la mise en valeur d'installations de service ;
7°) Des missions transversales nécessaires au bon fonctionnement du système de transport ferroviaire national, au bénéfice de l'ensemble des acteurs de ce système, notamment en matière de gestion de crise et de coordination des acteurs pour la mise en accessibilité du système de transport ferroviaire national aux personnes handicapées ou à mobilité réduite ;

8°) Des missions répondant aux besoins de la défense dans le cadre de la stratégie de sécurité nationale. »

La transparence et la non-discrimination étant indispensables à la réalisation de l'objectif de développement du transport ferroviaire, SNCF Réseau a établi le présent document de référence du réseau (DRR) qui décrit les principes et procédures relatifs à l'utilisation de l'infrastructure ferroviaire, comme le prévoient le Code des transports et le décret n° 2003-194 du 7 mars 2003.

Conformément au 7° de l'article L.2111-9 du code des transports modifié par l'article 1er de la loi n° 2018-515 du 27 juin 2018 pour un nouveau pacte ferroviaire et conformément à l'article 8.1 du décret n° 97-444 en vigueur, SNCF Réseau assure des missions de coordination des acteurs pour la mise en accessibilité du système de transport ferroviaire aux personnes en situation de handicap ou à mobilité réduite.

La Direction Générale Clients et Territoires (DG C&T) est, depuis le 1er janvier 2020, dotée d'une Direction de l'accessibilité en charge de réaliser ces missions dans les conditions suivantes :

1° Pour les parties prenantes du système de transport ferroviaire national, elle est l'interlocuteur de référence pour toutes questions relatives à l'accessibilité ; elle organise, en tant que de besoin, des concertations avec les associations nationales représentatives des personnes handicapées et à mobilité réduite ;

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