"""
Extraction d'entités pour documents arabes marocains
- BERT CamelBERT NER avec découpage intelligent par tokens
- Extraction hybride : NLP + règles (regex) pour les organisations
- Filtrage par confiance
"""
import re
import logging
import time
from transformers import pipeline, AutoTokenizer

logger = logging.getLogger(__name__)

ner = None
tokenizer = None

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-ner"
MAX_TOKENS = 480
OVERLAP_TOKENS = 50
MIN_CONFIDENCE = 0.4

# Classe de caractères arabes pour les regex
_AR = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'

# Séparateurs qui terminent le nom d'une organisation
# (ponctuation, retour à la ligne, certains mots-clés de coupure)
_ORG_STOP = r'[.،:\n\r]'

# Patterns regex pour les organisations marocaines
# Chaque pattern utilise un groupe de capture limité qui s'arrête
# à la ponctuation, fin de ligne ou après 6 mots arabes max.
MOROCCAN_ORG_PATTERNS = [
    # وزارة + suite (ex: وزارة التربية الوطنية)
    rf'وزارة[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # المديرية / مديرية + suite
    rf'(?:المديرية|مديرية)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # مؤسسة / المؤسسة + suite
    rf'(?:مؤسسة|المؤسسة)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # جامعة / الجامعة + suite
    rf'(?:جامعة|الجامعة)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # وكالة / الوكالة + suite
    rf'(?:وكالة|الوكالة)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # المكتب / مكتب + suite
    rf'(?:المكتب|مكتب)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # اللجنة / لجنة + suite
    rf'(?:اللجنة|لجنة)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # المحكمة / محكمة + suite
    rf'(?:المحكمة|محكمة)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # جماعة / بلدية + suite
    rf'(?:جماعة|الجماعة|بلدية)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # عمالة / إقليم / ولاية / جهة + suite
    rf'(?:عمالة|إقليم|ولاية|جهة)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # بنك / صندوق + suite
    rf'(?:بنك|البنك|صندوق|الصندوق)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
    # المجلس / مجلس + suite
    rf'(?:المجلس|مجلس)[\s\u200f]+((?:[{_AR}]+[\s\u200f]+){{1,5}}[{_AR}]+)',
]

# Pré-compiler les patterns pour de meilleures performances
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
    Chaque pattern capture le préfixe (وزارة, مديرية, etc.) + la suite,
    limitée à 6 mots arabes max pour éviter les phrases trop longues.
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

            # Dédupliquer à ce stade
            if full_match in seen:
                continue
            seen.add(full_match)

            entities.append({
                "word": full_match,
                "entity_group": "ORG",
                "score": 0.75,
                "source": "regex",
            })

    return entities


def _deduplicate_entities(entities):
    """
    Déduplique les entités en gardant celle avec le meilleur score.
    Gère aussi les cas où une entité est un sous-texte d'une autre.
    """
    if not entities:
        return []

    # D'abord, dédupliquer par texte exact (garder le meilleur score)
    seen = {}
    for ent in entities:
        key = ent["word"]
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent

    # Ensuite, retirer les sous-chaînes (si "مديرية" est contenu dans
    # "المديرية العامة للأمن" et les deux sont de type ORG, garder la plus longue)
    result = list(seen.values())
    filtered = []
    for ent in sorted(result, key=lambda e: len(e["word"]), reverse=True):
        is_substring = False
        for kept in filtered:
            if (ent["entity_group"] == kept["entity_group"]
                    and ent["word"] in kept["word"]
                    and ent["word"] != kept["word"]):
                is_substring = True
                break
        if not is_substring:
            filtered.append(ent)

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