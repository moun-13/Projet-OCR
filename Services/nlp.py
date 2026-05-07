"""
Extraction d'entités pour documents arabes marocains
- BERT CamelBERT NER avec découpage intelligent par tokens
- Extraction hybride : NLP + règles (regex) pour les organisations
- Filtrage par confiance
- Normalisation arabe pour la déduplication (OCR-resilient)
"""
import re
import logging
import time
from transformers import pipeline, AutoTokenizer

try:
    from Services.arabic_utils import (
        normalize_for_matching,
        normalize_for_display,
        build_flexible_pattern,
        AR_RANGE,
    )
except ImportError as e:
    logging.getLogger(__name__).warning(f"arabic_utils import failed: {e}")
    # Fallbacks minimaux
    AR_RANGE = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'
    def normalize_for_matching(t): return t
    def normalize_for_display(t): return t
    def build_flexible_pattern(kw): return re.escape(kw)

try:
    from Services.fuzzy_match import deduplicate_by_similarity
except ImportError as e:
    logging.getLogger(__name__).warning(f"fuzzy_match import failed: {e}")
    def deduplicate_by_similarity(items, threshold=0.85): return items

logger = logging.getLogger(__name__)

ner = None
tokenizer = None

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-ner"
MAX_TOKENS = 480
OVERLAP_TOKENS = 50
MIN_CONFIDENCE = 0.4

# Classe de caractères arabes pour les regex
_AR = AR_RANGE

# Séparateurs qui terminent le nom d'une organisation
# (ponctuation, retour à la ligne, certains mots-clés de coupure)
_ORG_STOP = r'[.،:\n\r]'

# Prefixes d'organisations marocaines
# On utilise build_flexible_pattern pour rendre chaque préfixe
# tolérant aux erreurs OCR (أ→ا, ة→ه, ي→ى, etc.)
_ORG_PREFIXES = [
    'وزارة',
    'المديرية', 'مديرية',
    'مؤسسة', 'المؤسسة',
    'جامعة', 'الجامعة',
    'وكالة', 'الوكالة',
    'المكتب', 'مكتب',
    'اللجنة', 'لجنة',
    'المحكمة', 'محكمة',
    'جماعة', 'الجماعة', 'بلدية',
    'عمالة', 'إقليم', 'ولاية', 'جهة',
    'بنك', 'البنك', 'صندوق', 'الصندوق',
    'المجلس', 'مجلس',
]

# Construire les patterns regex flexibles (tolérants OCR) pour chaque préfixe
# Chaque préfixe est rendu flexible: وزارة → [وٶؤ]ز[اأإآ]ر[ةه]
MOROCCAN_ORG_PATTERNS = []
_COMPILED_ORG_PATTERNS = []
try:
    for prefix in _ORG_PREFIXES:
        flex_prefix = build_flexible_pattern(prefix)
        pattern = flex_prefix + r'[\s\u200f]+((?:[' + _AR + r']+[\s\u200f]+){1,5}[' + _AR + r']+)'
        MOROCCAN_ORG_PATTERNS.append(pattern)
    _COMPILED_ORG_PATTERNS = [re.compile(p, re.UNICODE) for p in MOROCCAN_ORG_PATTERNS]
except Exception as e:
    logging.getLogger(__name__).warning(f"Flexible ORG patterns failed, using basic: {e}")
    # Fallback: patterns basiques sans flex
    _basic_prefixes = [
        'وزارة', 'المديرية', 'مديرية', 'مؤسسة', 'المؤسسة',
        'جامعة', 'الجامعة', 'وكالة', 'الوكالة', 'المكتب', 'مكتب',
        'اللجنة', 'لجنة', 'المحكمة', 'محكمة', 'جماعة', 'الجماعة', 'بلدية',
        'عمالة', 'إقليم', 'ولاية', 'جهة', 'بنك', 'البنك',
        'صندوق', 'الصندوق', 'المجلس', 'مجلس',
    ]
    MOROCCAN_ORG_PATTERNS = [
        rf'(?:{re.escape(p)})[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)'
        for p in _basic_prefixes
    ]
    _COMPILED_ORG_PATTERNS = [re.compile(p, re.UNICODE) for p in MOROCCAN_ORG_PATTERNS]


def get_ner():
    global ner, tokenizer
    if ner is None:
        logger.info(f"Chargement BERT NER: {MODEL_NAME}...")
        t = time.time()
        ner = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"BERT NER charge en {time.time() - t:.1f}s")
    return ner


def _split_text_by_tokens(text):
    global tokenizer
    if tokenizer is None:
        get_ner()
    words = text.split()
    if not words:
        return []
    chunks = []
    current_words = []
    current_token_count = 0
    for word in words:
        word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
        if current_token_count + word_tokens > MAX_TOKENS and current_words:
            chunks.append(" ".join(current_words))
            overlap_words = []
            overlap_count = 0
            for w in reversed(current_words):
                w_tokens = len(tokenizer.encode(w, add_special_tokens=False))
                if overlap_count + w_tokens > OVERLAP_TOKENS:
                    break
                overlap_words.insert(0, w)
                overlap_count += w_tokens
            current_words = overlap_words + [word]
            current_token_count = overlap_count + word_tokens
        else:
            current_words.append(word)
            current_token_count += word_tokens
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def _clean_bert_entity(word):
    cleaned = word.replace("##", "").strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Normaliser pour l'affichage (garder les taa marbuta etc.)
    cleaned = normalize_for_display(cleaned)
    return cleaned


def _extract_with_bert(text):
    ner_pipeline = get_ner()
    chunks = _split_text_by_tokens(text)
    all_entities = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            entities = ner_pipeline(chunk)
            for ent in entities:
                if ent["score"] >= MIN_CONFIDENCE:
                    cleaned_word = _clean_bert_entity(ent["word"])
                    if cleaned_word and len(cleaned_word) > 1:
                        all_entities.append({
                            "word": cleaned_word,
                            "entity_group": ent["entity_group"],
                            "score": round(ent["score"], 3),
                            "source": "bert",
                        })
        except Exception as e:
            logger.warning(f"Erreur BERT chunk {i+1}: {e}")
            continue
    return all_entities


def _extract_with_regex(text):
    """
    Extrait les organisations via regex.
    Utilise des patterns OCR-flexibles pour chaque préfixe.
    Chaque pattern capture le préfixe + la suite (max 6 mots arabes).
    """
    entities = []
    seen = set()

    for compiled_pattern in _COMPILED_ORG_PATTERNS:
        for match in compiled_pattern.finditer(text):
            # Le groupe 0 = match complet (préfixe + suite)
            full_match = match.group(0).strip()
            # Nettoyer : enlever la ponctuation en fin de match
            full_match = re.sub(r'[.،:\s]+$', '', full_match)

            # Vérifier la qualité du match
            if len(full_match) < 5:
                continue

            # Vérifier que le match contient au moins 2 mots arabes
            arabic_words = re.findall(rf'[{_AR}]+', full_match)
            if len(arabic_words) < 2:
                continue

            # Dédupliquer avec normalisation
            norm_key = normalize_for_matching(full_match)
            if norm_key in seen:
                continue
            seen.add(norm_key)

            entities.append({
                "word": normalize_for_display(full_match),
                "entity_group": "ORG",
                "score": 0.75,
                "source": "regex",
            })

    return entities


def _deduplicate_entities(entities):
    """
    Déduplique les entités en gardant celle avec le meilleur score.
    Utilise le fuzzy matching pour fusionner les variantes OCR.
    Gère aussi les cas où une entité est un sous-texte d'une autre.
    """
    if not entities:
        return []

    # D'abord, dédupliquer par texte normalisé (garder le meilleur score)
    seen = {}
    for ent in entities:
        key = normalize_for_matching(ent["word"])
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent

    # Ensuite, fusionner les variantes OCR proches (fuzzy dedup)
    result = list(seen.values())
    
    # Grouper par entity_group pour la déduplication fuzzy
    grouped = {}
    for ent in result:
        eg = ent["entity_group"]
        grouped.setdefault(eg, []).append(ent)
    
    filtered = []
    for eg, ents in grouped.items():
        # Trier par longueur de mot (plus long d'abord)
        ents_sorted = sorted(ents, key=lambda e: len(e["word"]), reverse=True)
        
        # Déduplication par similarité fuzzy
        words = [e["word"] for e in ents_sorted]
        unique_words = deduplicate_by_similarity(words, threshold=0.85)
        unique_words_set = set(unique_words)
        
        for ent in ents_sorted:
            if ent["word"] in unique_words_set:
                filtered.append(ent)
                unique_words_set.discard(ent["word"])

    return filtered


def extract_entities(text):
    if not text or not text.strip():
        logger.warning("Texte vide, pas d'entites a extraire")
        return []
    t = time.time()
    bert_entities = _extract_with_bert(text)
    logger.info(f"BERT: {len(bert_entities)} entite(s)")
    regex_entities = _extract_with_regex(text)
    logger.info(f"Regex: {len(regex_entities)} organisation(s)")
    all_entities = bert_entities + regex_entities
    unique_entities = _deduplicate_entities(all_entities)
    for ent in unique_entities:
        logger.info(f"  -> {ent['entity_group']}: '{ent['word']}' (score: {ent['score']}, source: {ent['source']})")
    logger.info(f"NLP termine: {len(unique_entities)} entite(s) uniques en {time.time() - t:.1f}s")
    return unique_entities