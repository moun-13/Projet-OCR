"""
fuzzy_match.py — Matching intelligent tolérant aux erreurs OCR.

Utilise RapidFuzz pour le fuzzy matching et le module `regex` pour les
regex avec tolérance aux erreurs ({e<=N}).

Fournit:
- Recherche fuzzy de mots-clés dans du texte OCR
- Extraction de valeurs à proximité de mots-clés détectés
- Matching de labels/enums avec score de confiance
- Matching contextuel combinant position + similarité
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    import regex as re2
except ImportError:
    re2 = None
    logging.getLogger(__name__).warning(
        "Module 'regex' non installé. Fuzzy regex désactivé. "
        "Installer avec: pip install regex"
    )

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None
    logging.getLogger(__name__).warning(
        "Module 'rapidfuzz' non installé. Fuzzy matching désactivé. "
        "Installer avec: pip install rapidfuzz"
    )

from Services.arabic_utils import (
    normalize_for_matching,
    normalize_digits,
    clean_extracted_value,
    AR_RANGE,
    build_flexible_pattern,
)

logger = logging.getLogger(__name__)


# Data classes pour les résultats

@dataclass
class FuzzyMatch:
    """Résultat d'un match fuzzy de mot-clé."""
    keyword: str         # Mot-clé original cherché
    matched_text: str    # Texte qui a matché
    position: int        # Position dans le texte source
    score: float         # Score de similarité (0.0 - 1.0)


@dataclass
class ExtractionResult:
    """Résultat d'extraction d'un champ."""
    value: str = ""
    confidence: float = 0.0
    method: str = ""     # "regex_exact", "regex_fuzzy", "fuzzy_keyword", "enum", "ner"
    field_name: str = ""
    
    @property
    def is_empty(self) -> bool:
        return not self.value or not self.value.strip()
    
    def __repr__(self):
        if self.is_empty:
            return f"ExtractionResult(empty, field={self.field_name})"
        return (
            f"ExtractionResult('{self.value[:40]}...', "
            f"conf={self.confidence:.2f}, method={self.method})"
        )


# Fuzzy keyword search

def fuzzy_find_keyword(
    text: str,
    keywords: list[str],
    threshold: float = 0.70,
    window_factor: float = 1.5,
) -> Optional[FuzzyMatch]:
    """
    Cherche un mot-clé dans le texte OCR avec tolérance aux erreurs.
    
    Stratégie:
    1. D'abord recherche exacte (normalisée) — plus rapide
    2. Si pas trouvé, fenêtre glissante + RapidFuzz partial_ratio
    
    Args:
        text: Texte OCR source
        keywords: Liste de mots-clés à chercher
        threshold: Score minimum (0.0 - 1.0)
        window_factor: Multiplicateur de taille de fenêtre vs mot-clé
    
    Returns:
        FuzzyMatch ou None si aucun match >= threshold
    """
    text_norm = normalize_for_matching(text)
    best_match: Optional[FuzzyMatch] = None
    best_score = 0.0
    
    for kw in keywords:
        kw_norm = normalize_for_matching(kw)
        if not kw_norm:
            continue
        
        # --- Recherche exacte (normalisée) ---
        pos = text_norm.find(kw_norm)
        if pos >= 0:
            # Vérifier les limites de mot arabe
            before_ok = pos == 0 or not re.match(rf'[{AR_RANGE}]', text_norm[pos - 1])
            after_end = pos + len(kw_norm)
            after_ok = after_end >= len(text_norm) or not re.match(rf'[{AR_RANGE}]', text_norm[after_end])
            
            if before_ok and after_ok:
                return FuzzyMatch(
                    keyword=kw,
                    matched_text=kw_norm,
                    position=pos,
                    score=1.0,
                )
        
        # --- Fuzzy search avec RapidFuzz ---
        if fuzz is None:
            continue
        
        window_size = max(int(len(kw_norm) * window_factor), len(kw_norm) + 5)
        step = max(1, len(kw_norm) // 3)
        
        for i in range(0, max(1, len(text_norm) - len(kw_norm) + 1), step):
            window = text_norm[i:i + window_size]
            score = fuzz.partial_ratio(kw_norm, window) / 100.0
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = FuzzyMatch(
                    keyword=kw,
                    matched_text=window[:len(kw_norm)],
                    position=i,
                    score=score,
                )
                # Early exit si très bon match
                if score >= 0.95:
                    return best_match
    
    return best_match


def extract_value_after_keyword(
    text: str,
    keyword_match: FuzzyMatch,
    max_chars: int = 150,
    value_pattern: Optional[str] = None,
) -> str:
    """
    Extrait la valeur qui suit un mot-clé trouvé dans le texte.
    
    Args:
        text: Texte source original
        keyword_match: Résultat de fuzzy_find_keyword
        max_chars: Nombre max de caractères à extraire après le mot-clé
        value_pattern: Regex optionnelle pour valider/extraire la valeur
    
    Returns:
        Valeur nettoyée ou chaîne vide
    """
    if not keyword_match:
        return ""
    
    # Trouver la position dans le texte original (pas normalisé)
    text_norm = normalize_for_matching(text)
    pos = keyword_match.position
    kw_len = len(normalize_for_matching(keyword_match.keyword))
    
    # Zone après le mot-clé
    after_start = pos + kw_len
    after = text_norm[after_start:after_start + max_chars]
    
    if not after.strip():
        return ""
    
    # Si un pattern spécifique est fourni, l'utiliser
    if value_pattern:
        m = re.search(value_pattern, after, re.UNICODE)
        if m:
            return clean_extracted_value(m.group(1) if m.groups() else m.group(0))
    
    # Extraction générique: texte jusqu'au prochain séparateur
    m = re.match(
        r'\s*[:\s\-ـ]*\s*(.{1,' + str(max_chars) + r'}?)(?:\n|$|[.،؛](?:\s|$))',
        after, re.UNICODE
    )
    if m:
        val = clean_extracted_value(m.group(1))
        if val and len(val) > 1:
            return val
    
    return ""


# Fuzzy keyword + value extraction combinée


def fuzzy_extract_near_keyword(
    text: str,
    keywords: list[str],
    max_chars: int = 150,
    threshold: float = 0.70,
    value_pattern: Optional[str] = None,
) -> ExtractionResult:
    """
    Pipeline complet: cherche un mot-clé (fuzzy) puis extrait la valeur.
    
    Combine fuzzy_find_keyword + extract_value_after_keyword.
    Le score de confiance tient compte du match du mot-clé et
    de la qualité de la valeur extraite.
    """
    match = fuzzy_find_keyword(text, keywords, threshold)
    if not match:
        return ExtractionResult(confidence=0.0, method="fuzzy_keyword")
    
    value = extract_value_after_keyword(text, match, max_chars, value_pattern)
    
    if not value:
        return ExtractionResult(confidence=0.0, method="fuzzy_keyword")
    
    # Score de confiance = score du keyword match * bonus qualité valeur
    value_quality = min(1.0, len(value.strip()) / 3.0)  # Pénalise les valeurs très courtes
    confidence = match.score * 0.7 + value_quality * 0.3
    
    return ExtractionResult(
        value=value,
        confidence=round(confidence, 3),
        method="fuzzy_keyword",
    )



# Regex fuzzy avec le module `regex`

def fuzzy_regex_search(
    pattern: str,
    text: str,
    max_errors: int = 1,
    flags: int = 0,
):
    """
    Recherche regex avec tolérance aux erreurs via le module `regex`.
    
    Utilise la syntaxe {e<=N} pour accepter N substitutions/insertions/suppressions.
    
    Args:
        pattern: Pattern regex standard
        text: Texte à fouiller
        max_errors: Nombre max d'erreurs tolérées
        flags: Flags regex additionnels
    
    Returns:
        Match object ou None
    """
    if re2 is None:
        # Fallback: recherche exacte avec module `re` standard
        return re.search(pattern, text, flags)
    
    # Ajouter le paramètre fuzzy au pattern
    fuzzy_pattern = f'(?:{pattern}){{e<={max_errors}}}'
    
    try:
        return re2.search(fuzzy_pattern, text, flags=flags | re2.BESTMATCH)
    except Exception as e:
        logger.debug(f"Fuzzy regex error: {e}, falling back to exact")
        try:
            return re2.search(pattern, text, flags=flags)
        except Exception:
            return re.search(pattern, text, flags)



# Enum/label matching


def match_enum_fuzzy(
    text: str,
    enum_map: dict[str, str],
    threshold: float = 0.75,
) -> ExtractionResult:
    """
    Cherche la meilleure correspondance entre le texte OCR et
    un dictionnaire de valeurs connues (enum).
    
    Utile pour: نوع الاتفاقية, حالة الاتفاقية, etc.
    
    Args:
        text: Texte OCR source
        enum_map: {phrase_attendue: label_normalisé}
        threshold: Score minimum
    
    Returns:
        ExtractionResult avec la valeur matchée
    """
    text_norm = normalize_for_matching(text)
    
    best_label = ""
    best_score = 0.0
    
    for phrase, label in enum_map.items():
        phrase_norm = normalize_for_matching(phrase)
        
        # Recherche exacte d'abord
        if phrase_norm in text_norm:
            return ExtractionResult(
                value=label,
                confidence=0.95,
                method="enum_exact",
            )
        
        # Fuzzy matching
        if fuzz is not None:
            score = fuzz.partial_ratio(phrase_norm, text_norm) / 100.0
            if score > best_score and score >= threshold:
                best_score = score
                best_label = label
    
    if best_label:
        return ExtractionResult(
            value=best_label,
            confidence=round(best_score * 0.9, 3),  # Léger penalty car fuzzy
            method="enum_fuzzy",
        )
    
    return ExtractionResult(confidence=0.0, method="enum")


def deduplicate_by_similarity(
    items: list[str],
    threshold: float = 0.85,
) -> list[str]:
    """
    Déduplique une liste de chaînes en fusionnant les similarités.
    Garde la version la plus longue pour chaque cluster.
    
    Utile pour fusionner les variantes OCR d'une même entité NER.
    """
    if not items or fuzz is None:
        return items
    
    items_sorted = sorted(items, key=len, reverse=True)
    result = []
    
    for item in items_sorted:
        item_norm = normalize_for_matching(item)
        is_duplicate = False
        for kept in result:
            kept_norm = normalize_for_matching(kept)
            score = fuzz.ratio(item_norm, kept_norm) / 100.0
            if score >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            result.append(item)
    
    return result
