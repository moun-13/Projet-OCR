"""
arabic_utils.py — Utilitaires centralisés pour le traitement de texte arabe.

Fournit:
- Normalisation agressive (pour le matching interne)
- Normalisation légère (pour l'affichage)
- Table de caractères confusables OCR
- Construction de regex flexibles tolérant les erreurs OCR
- Conversion de chiffres arabes/persans → occidentaux
"""
from __future__ import annotations
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Tables de référence
# ═══════════════════════════════════════════════════════════════════

# Chiffres arabes orientaux → occidentaux
ARABIC_DIGIT_MAP = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

# Chiffres persans → occidentaux
PERSIAN_DIGIT_MAP = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

# Caractères confusables en OCR arabe — mapping pour le matching
# Clé = variante fréquente en OCR, Valeur = forme canonique
OCR_CONFUSABLES_MAP = {
    # Alef variants → ا
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ٲ': 'ا', 'ٳ': 'ا',
    # Taa marbuta → ه  (pour le matching uniquement)
    'ة': 'ه',
    # Alif maqsura → ي
    'ى': 'ي',
    # Hamza sur supports → lettre support
    'ؤ': 'و', 'ئ': 'ي',
    # Kaf persan/urdu → Kaf arabe
    'ک': 'ك', 'ڪ': 'ك',
    # Yaa persan → Yaa arabe
    'ی': 'ي', 'ے': 'ي',
    # Haa persan/urdu
    'ھ': 'ه',
    # Waw avec hamza isolée
    'ٶ': 'و',
}

# Mapping inverse : pour chaque caractère canonique, toutes ses variantes OCR
# Utilisé par build_flexible_pattern() pour générer des classes de caractères
_CANONICAL_TO_VARIANTS: dict[str, set[str]] = {}
for variant, canonical in OCR_CONFUSABLES_MAP.items():
    _CANONICAL_TO_VARIANTS.setdefault(canonical, {canonical}).add(variant)
# Ajouter les self-maps (le caractère canonique lui-même)
for canonical in list(_CANONICAL_TO_VARIANTS.keys()):
    _CANONICAL_TO_VARIANTS[canonical].add(canonical)

# Regex pré-compilée pour les diacritiques/tashkeel
_TASHKEEL_RE = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')

# Regex pour les marques directionnelles Unicode
_BIDI_RE = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069\u200b-\u200d\u00ad]')

# Regex pour les caractères de contrôle
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Plage Unicode arabe pour les regex
AR_RANGE = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'

# Mois arabes (dialecte marocain + MSA) avec toutes les variantes OCR connues
ARABIC_MONTHS = {
    # Marocain
    'يناير': '01', 'فبراير': '02', 'مارس': '03', 'أبريل': '04',
    'ابريل': '04', 'ماي': '05', 'يونيو': '06', 'يوليوز': '07',
    'غشت': '08', 'شتنبر': '09', 'أكتوبر': '10', 'اكتوبر': '10',
    'نونبر': '11', 'دجنبر': '12',
    # MSA
    'يوليو': '07', 'أغسطس': '08', 'اغسطس': '08',
    'سبتمبر': '09', 'نوفمبر': '11', 'ديسمبر': '12',
    # Variantes OCR fréquentes
    'نوفبر': '11', 'نوبر': '11', 'دسمبر': '12',
    'فبراير': '02', 'ابرايل': '04', 'ينایر': '01',
}


# ═══════════════════════════════════════════════════════════════════
# Fonctions de normalisation
# ═══════════════════════════════════════════════════════════════════

def remove_tashkeel(text: str) -> str:
    """Supprime les diacritiques (حركات) du texte arabe."""
    return _TASHKEEL_RE.sub('', text)


def normalize_digits(text: str) -> str:
    """Convertit les chiffres arabes orientaux et persans → occidentaux."""
    for ar, west in ARABIC_DIGIT_MAP.items():
        text = text.replace(ar, west)
    for pr, west in PERSIAN_DIGIT_MAP.items():
        text = text.replace(pr, west)
    return text


def normalize_for_matching(text: str) -> str:
    """
    Normalisation AGRESSIVE pour la comparaison/recherche.
    
    Applique:
    - NFC Unicode
    - Suppression tashkeel
    - Mapping confusables (ة→ه, أ→ا, etc.)
    - Suppression marques directionnelles
    - Suppression caractères de contrôle
    - Espaces multiples → un seul
    - Conversion chiffres
    
     NE PAS utiliser pour l'affichage final, seulement pour le matching.
    """
    if not text:
        return ""
    
    # NFC
    text = unicodedata.normalize("NFC", text)
    
    # Contrôle + bidi
    text = _CONTROL_RE.sub('', text)
    text = _BIDI_RE.sub('', text)
    
    # Tashkeel
    text = remove_tashkeel(text)
    
    # Confusables
    for variant, canonical in OCR_CONFUSABLES_MAP.items():
        text = text.replace(variant, canonical)
    
    # Chiffres
    text = normalize_digits(text)
    
    # Espaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_for_display(text: str) -> str:
    """
    Normalisation LÉGÈRE pour l'affichage.
    
    Garde les taa marbuta (ة), alef hamza (أ/إ), etc.
    Supprime seulement les artéfacts OCR (contrôle, bidi, tashkeel erroné).
    """
    if not text:
        return ""
    
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub('', text)
    text = _BIDI_RE.sub('', text)
    text = remove_tashkeel(text)
    
    # Normaliser les kaf/yaa persans (toujours des erreurs OCR dans les docs marocains)
    text = text.replace('ک', 'ك').replace('ڪ', 'ك')
    text = text.replace('ی', 'ي').replace('ے', 'ي')
    text = text.replace('ھ', 'ه')
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Construction de regex flexibles
# ═══════════════════════════════════════════════════════════════════

def build_flexible_pattern(keyword: str) -> str:
    """
    Construit un pattern regex tolérant aux erreurs OCR pour un mot arabe.
    
    Remplace chaque caractère arabe par une classe de caractères incluant
    toutes ses variantes OCR confusables.
    
    Exemples:
        'اتفاقية' → '[اأإآٱٲٳ]تف[اأإآٱٲٳ]ق[يیىے][ةه]'
        'وزارة'  → '[وٶؤ]ز[اأإآٱٲٳ]ر[ةه]'
    
    Ajoute aussi une tolérance aux espaces parasites entre les caractères
    (fréquent en OCR arabe) et aux tashkeel erronés.
    """
    parts = []
    tashkeel_opt = r'[\u064B-\u0652\u0670]?'  # Tashkeel optionnel entre lettres
    
    for char in keyword:
        if char in _CANONICAL_TO_VARIANTS:
            # Caractère canonique avec variantes connues
            variants = _CANONICAL_TO_VARIANTS[char]
            if len(variants) > 1:
                char_class = '[' + ''.join(sorted(variants)) + ']'
            else:
                char_class = re.escape(char)
            parts.append(char_class + tashkeel_opt)
        elif char in OCR_CONFUSABLES_MAP:
            # Caractère qui est lui-même une variante
            canonical = OCR_CONFUSABLES_MAP[char]
            variants = _CANONICAL_TO_VARIANTS.get(canonical, {char, canonical})
            char_class = '[' + ''.join(sorted(variants)) + ']'
            parts.append(char_class + tashkeel_opt)
        elif char == ' ':
            # Espace : tolérer espaces multiples, RLM, etc.
            parts.append(r'[\s\u200f\u200e]*')
        else:
            # Caractère non-arabe : escape et garder tel quel
            parts.append(re.escape(char) + tashkeel_opt)
    
    return ''.join(parts)


def build_flexible_keywords(keywords: list[str]) -> str:
    """
    Construit un pattern regex alternation pour plusieurs mots-clés.
    Chaque mot-clé est rendu flexible via build_flexible_pattern.
    
    Retourne un pattern '(?:pattern1|pattern2|...)' 
    """
    patterns = [build_flexible_pattern(kw) for kw in keywords]
    return '(?:' + '|'.join(patterns) + ')'


def clean_extracted_value(val: str) -> str:
    """Nettoie une valeur extraite: supprime ponctuation de début/fin, espaces."""
    if not val:
        return ""
    val = val.strip()
    val = re.sub(r'[\s\u200f]+', ' ', val)
    val = re.sub(r'^[:\s،.؛\-ـ]+|[:\s،.؛\-ـ]+$', '', val)
    return val.strip()
