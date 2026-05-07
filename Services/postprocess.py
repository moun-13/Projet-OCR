"""
Post-traitement pour conventions administratives marocaines.

Architecture déclarative:
- Chaque champ est configuré via FieldConfig (mots-clés, patterns, type...)
- Le moteur ExtractionEngine orchestre l'extraction hybride:
    1. Regex exactes (score 0.95)
    2. Regex flexibles / fuzzy regex (score 0.85)
    3. Fuzzy keyword + extraction contextuelle (score 0.70-0.85)
    4. Enum fuzzy matching (score 0.80-0.95)
    5. NER fallback (score du modèle)
- Score de confiance sur chaque résultat
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

try:
    from Services.arabic_utils import (
        normalize_for_matching,
        normalize_for_display,
        normalize_digits,
        clean_extracted_value,
        build_flexible_pattern,
        build_flexible_keywords,
        AR_RANGE,
        ARABIC_MONTHS,
        ARABIC_DIGIT_MAP,
    )
except ImportError as _e:
    logging.getLogger(__name__).error(f"arabic_utils import failed: {_e}")
    raise

try:
    from Services.fuzzy_match import (
        ExtractionResult,
        fuzzy_find_keyword,
        extract_value_after_keyword,
        fuzzy_extract_near_keyword,
        fuzzy_regex_search,
        match_enum_fuzzy,
    )
except ImportError as _e:
    logging.getLogger(__name__).error(f"fuzzy_match import failed: {_e}")
    raise

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration déclarative des champs
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FieldConfig:
    """Configuration d'un champ à extraire."""
    name: str                            # Nom arabe du champ (clé JSON)
    keywords: list[str]                  # Mots-clés de détection
    patterns: list[str] = field(default_factory=list)  # Regex spécifiques
    max_value_length: int = 100          # Longueur max de la valeur
    value_type: str = "text"             # text, number, date, enum
    enum_values: dict = field(default_factory=dict)    # Pour les champs enum
    ner_fallback: str = ""               # Type NER de fallback (ORG, PER)
    ner_index: int = 0                   # Index dans la liste NER
    fuzzy_threshold: float = 0.70        # Seuil de fuzzy matching
    max_extract_distance: int = 150      # Max caractères après le mot-clé


# --- Configurations de tous les champs ---

FIELD_CONFIGS = [
    FieldConfig(
        name="رقم_الاتفاقية",
        keywords=[
            'اتفاقية رقم', 'الاتفاقية رقم',
            'اتفاقية عدد', 'الاتفاقية عدد',
            'رقم الاتفاقية', 'عدد الاتفاقية',
        ],
        patterns=[
            # "اتفاقية رقم 1-151-83" ou "اتفاقية رقم: 1-151-83"
            r'(?:اتفاقية|الاتفاقية)\s*(?:رقم|عدد)\s*[:\s]*([^\n،.]{2,50})',
            r'(?:رقم|عدد)\s*(?:الاتفاقية|اتفاقية)\s*[:\s]*([^\n،.]{2,50})',
            # Numéro standalone type "1-151-83" ou "2024/123"
            r'(?:رقم|عدد)\s*[:\s]*([\d][\d/\-\s]*[\d])',
            # Pattern large: tout ce qui suit رقم/عدد avec des chiffres et tirets
            r'(?:رقم|عدد)\s*[:\s]*(\d[\d\-/\s\.]+\d)',
        ],
        max_value_length=50,
    ),
    FieldConfig(
        name="تاريخ_البداية",
        keywords=[
            'تاريخ البداية', 'تاريخ الانطلاق', 'تاريخ بداية', 'تاريخ السريان',
            'ابتداء من', 'بتاريخ',
        ],
        patterns=[
            r'(?:تاريخ\s*(?:البداية|الانطلاق|بداية|السريان))\s*[:\s]*([\d/\-.\s]+)',
            r'(?:ابتداء\s*من|بتاريخ|في)\s*[:\s]*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})',
        ],
        value_type="date",
        max_value_length=40,
    ),
    FieldConfig(
        name="الإطارية",
        keywords=['إطارية', 'اتفاقية إطار', 'اتفاق إطار', 'الإطارية'],
        patterns=[
            r'(?:اتفاقية|اتفاق)\s+(إطار(?:ية|ي)?[^،.\n]{0,60})',
            r'(إطارية|إطاري|اتفاقية\s+إطار)',
            r'(?:الإطار(?:ية|ي)?)\s*[:\s]*([^،.\n]{2,80})',
        ],
        max_value_length=80,
    ),
    FieldConfig(
        name="السنة",
        keywords=['سنة', 'السنة', 'عام'],
        patterns=[
            r'(?:سنة|السنة|عام)\s*[:\s]*(\d{4})',
        ],
        value_type="number",
        max_value_length=10,
    ),
    FieldConfig(
        name="الدورة",
        keywords=['الدورة', 'دورة'],
        max_value_length=80,
    ),
    FieldConfig(
        name="الكيان_الاحتصاري",
        keywords=[
            'الكيان', 'الجهة المعنية',
            'الطرف الأول', 'الطرف الاول',
        ],
        ner_fallback="ORG",
        ner_index=0,
        max_value_length=80,
    ),
    FieldConfig(
        name="مساهمة_الجهة",
        keywords=[
            'مساهمة الجهة', 'حصة الجهة', 'مساهمة المجلس',
        ],
        patterns=[
            r'(?:مساهمة\s*(?:الجهة)?)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
            r'(?:حصة|مساهمة)\s*(?:الجهة|المجلس)\s*[:\s]*([\d\s.,]+)',
            r'(?:مبلغ|المبلغ)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD))',
            r'(?:بمبلغ|قدره|يقدر\s*ب)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
            r'([\d.,]+\s*(?:درهم|DH|MAD))',
        ],
        value_type="number",
        max_value_length=40,
    ),
    FieldConfig(
        name="المبلغ_الإجمالي",
        keywords=[
            'المبلغ الإجمالي', 'المبلغ الاجمالي',
            'الغلاف المالي', 'التكلفة الإجمالية',
        ],
        patterns=[
            r'(?:المبلغ\s*الإجمالي|المبلغ\s*الاجمالي)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
            r'(?:الغلاف\s*المالي|التكلفة\s*الإجمالية)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
            r'(?:بمبلغ|قدره)\s*(?:إجمالي)?\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD))',
        ],
        value_type="number",
        max_value_length=40,
    ),
    FieldConfig(
        name="المجال",
        keywords=['المجال', 'مجال', 'القطاع', 'قطاع', 'الميدان'],
        max_value_length=60,
    ),
    FieldConfig(
        name="موضوع_الاتفاقية",
        keywords=[
            'موضوع الاتفاقية', 'موضوع', 'الموضوع',
            'الهدف من', 'تهدف إلى', 'تهدف الى',
        ],
        patterns=[
            r'(?:من\s+أجل|بهدف|بغرض)\s+(.{10,200}?)(?:\n|$|[.،؛](?:\s|$))',
        ],
        max_value_length=200,
        max_extract_distance=200,
    ),
    FieldConfig(
        name="صاحب_المشروع",
        keywords=[
            'صاحب المشروع', 'صاحب مشروع', 'المستفيد', 'الجهة المستفيدة',
            'الطرف الثاني', 'الطرف التاني',
        ],
        ner_fallback="PER",
        ner_index=0,
        max_value_length=80,
    ),
    FieldConfig(
        name="رقم_القرار",
        keywords=[
            'قرار رقم', 'القرار رقم',
            'رقم القرار', 'عدد القرار',
            'مقرر رقم', 'المقرر رقم',
        ],
        patterns=[
            r'(?:قرار|القرار)\s*(?:رقم|عدد)\s*[:\s]*([\d/\-A-Za-z]+)',
            r'(?:رقم|عدد)\s*(?:القرار|قرار)\s*[:\s]*([\d/\-A-Za-z]+)',
            r'(?:مقرر|المقرر)\s*(?:رقم|عدد)\s*[:\s]*([\d/\-A-Za-z]+)',
        ],
        max_value_length=40,
    ),
    FieldConfig(
        name="الشريك",
        keywords=[
            'الشريك', 'شريك', 'المتعاقد', 'المتعاقد معه',
            'الطرف الثاني', 'الطرف التاني',
        ],
        ner_fallback="ORG",
        ner_index=1,
        max_value_length=80,
    ),
    FieldConfig(
        name="الأطراف",
        keywords=[
            'الأطراف', 'أطراف الاتفاقية', 'الموقعون', 'الموقعين',
        ],
        ner_fallback="ORG",
        ner_index=-1,  # Toutes les ORG
        max_value_length=200,
        max_extract_distance=200,
    ),
    FieldConfig(
        name="الاختصاص",
        keywords=['الاختصاص', 'اختصاص', 'الصلاحية', 'الولاية'],
        max_value_length=100,
    ),
    FieldConfig(
        name="سريان_الاتفاقية",
        keywords=[
            'سريان', 'مدة الاتفاقية', 'مدة السريان',
            'مدة التنفيذ', 'صلاحية', 'المدة',
        ],
        patterns=[
            r'(?:لمدة|مدة)\s*[:\s]*(\d+\s*(?:سنة|سنوات|أشهر|شهر|شهرا))',
        ],
        max_value_length=60,
    ),
    FieldConfig(
        name="نوع_الاتفاقية",
        keywords=['نوع الاتفاقية', 'نوع الاتفاق', 'طبيعة'],
        value_type="enum",
        enum_values={
            'اتفاقية شراكة': 'شراكة',
            'اتفاقية تعاون': 'تعاون',
            'اتفاقية إطار': 'إطارية',
            'اتفاقية إطارية': 'إطارية',
            'اتفاقية انجاز': 'انجاز',
            'اتفاقية تمويل': 'تمويل',
            'اتفاقية برنامج': 'برنامج',
            'عقد': 'عقد',
            'ملحق': 'ملحق',
            'بروتوكول': 'بروتوكول',
        },
    ),
    FieldConfig(
        name="البرامج",
        keywords=['البرنامج', 'برنامج', 'البرامج', 'المشروع', 'مشروع'],
        max_value_length=100,
    ),
    FieldConfig(
        name="حالة_الاتفاقية",
        keywords=['حالة', 'الحالة', 'الوضعية'],
        value_type="enum",
        enum_values={
            'ساري المفعول': 'ساري المفعول',
            'سارية المفعول': 'سارية المفعول',
            'منتهية': 'منتهية',
            'منتهي': 'منتهي',
            'قيد التنفيذ': 'قيد التنفيذ',
            'قيد الانجاز': 'قيد الانجاز',
            'ملغاة': 'ملغاة',
            'معلقة': 'معلقة',
            'جديدة': 'جديدة',
        },
    ),
    FieldConfig(
        name="المرفقات",
        keywords=['المرفقات', 'مرفقات', 'الوثائق المرفقة', 'الملاحق'],
        max_value_length=150,
        max_extract_distance=150,
    ),
]


# ═══════════════════════════════════════════════════════════════════
# Moteur d'extraction hybride
# ═══════════════════════════════════════════════════════════════════

class ExtractionEngine:
    """
    Moteur d'extraction hybride pour les conventions administratives.
    
    Pipeline pour chaque champ:
    1. Regex exactes avec patterns flexibles (tolérant OCR)
    2. Regex fuzzy via module `regex` {e<=1}
    3. Fuzzy keyword search + extraction contextuelle
    4. Enum fuzzy matching (pour les champs à valeurs fixes)
    5. NER fallback (utilise les entités BERT)
    
    Garde le résultat avec le meilleur score de confiance.
    """
    
    def __init__(self):
        # Pré-compiler les patterns flexibles pour chaque config
        self._compiled_patterns: dict[str, list] = {}
        for config in FIELD_CONFIGS:
            compiled = []
            for pat in config.patterns:
                try:
                    compiled.append(re.compile(pat, re.UNICODE))
                except re.error as e:
                    logger.warning(f"Pattern invalide pour {config.name}: {e}")
            self._compiled_patterns[config.name] = compiled
    
    def _extract_with_regex(
        self, text: str, config: FieldConfig
    ) -> ExtractionResult:
        """Étape 1: Regex exactes sur texte normalisé pour les chiffres."""
        text_c = normalize_digits(text)
        
        for compiled in self._compiled_patterns.get(config.name, []):
            m = compiled.search(text_c)
            if m:
                val = clean_extracted_value(m.group(1) if m.groups() else m.group(0))
                if val and 1 < len(val) <= config.max_value_length:
                    return ExtractionResult(
                        value=val,
                        confidence=0.92,
                        method="regex_exact",
                        field_name=config.name,
                    )
        return ExtractionResult(field_name=config.name)
    
    def _extract_with_flexible_regex(
        self, text: str, config: FieldConfig
    ) -> ExtractionResult:
        """Étape 2: Regex avec patterns flexibles (caractères confusables)."""
        text_norm = normalize_for_matching(text)
        text_c = normalize_digits(text)
        
        # Construire des versions flexibles des patterns
        for pat_str in config.patterns:
            try:
                m = re.search(pat_str, text_norm, re.UNICODE)
                if m:
                    val = clean_extracted_value(m.group(1) if m.groups() else m.group(0))
                    if val and 1 < len(val) <= config.max_value_length:
                        return ExtractionResult(
                            value=val,
                            confidence=0.85,
                            method="regex_flexible",
                            field_name=config.name,
                        )
            except re.error:
                continue
        
        # Essayer fuzzy regex si le module regex est disponible
        if re2 is not None:
            for pat_str in config.patterns:
                try:
                    m = fuzzy_regex_search(pat_str, text_c, max_errors=1)
                    if m:
                        val = clean_extracted_value(
                            m.group(1) if m.groups() else m.group(0)
                        )
                        if val and 1 < len(val) <= config.max_value_length:
                            return ExtractionResult(
                                value=val,
                                confidence=0.82,
                                method="regex_fuzzy",
                                field_name=config.name,
                            )
                except Exception:
                    continue
        
        return ExtractionResult(field_name=config.name)
    
    def _extract_with_fuzzy_keyword(
        self, text: str, config: FieldConfig
    ) -> ExtractionResult:
        """Étape 3: Fuzzy keyword search + extraction contextuelle."""
        result = fuzzy_extract_near_keyword(
            text,
            config.keywords,
            max_chars=config.max_extract_distance,
            threshold=config.fuzzy_threshold,
        )
        
        if not result.is_empty:
            # Valider la longueur
            if len(result.value) <= config.max_value_length:
                result.field_name = config.name
                return result
        
        return ExtractionResult(field_name=config.name)
    
    def _extract_with_enum(
        self, text: str, config: FieldConfig
    ) -> ExtractionResult:
        """Étape 4: Matching fuzzy dans un ensemble de valeurs connues."""
        if not config.enum_values:
            return ExtractionResult(field_name=config.name)
        
        result = match_enum_fuzzy(text, config.enum_values, threshold=0.75)
        result.field_name = config.name
        return result
    
    def _extract_with_ner(
        self, config: FieldConfig, entities: list
    ) -> ExtractionResult:
        """Étape 5: Utilise les entités NER comme fallback."""
        if not config.ner_fallback or not entities:
            return ExtractionResult(field_name=config.name)
        
        matching = [
            e for e in entities
            if config.ner_fallback in e.get("entity_group", "")
        ]
        
        if not matching:
            return ExtractionResult(field_name=config.name)
        
        if config.ner_index == -1:
            # Toutes les entités du type → joindre
            values = [e["word"] for e in matching[:4]]
            return ExtractionResult(
                value=" / ".join(values),
                confidence=min(e.get("score", 0.5) for e in matching[:4]),
                method="ner_all",
                field_name=config.name,
            )
        
        if config.ner_index < len(matching):
            ent = matching[config.ner_index]
            return ExtractionResult(
                value=ent["word"],
                confidence=ent.get("score", 0.5),
                method="ner",
                field_name=config.name,
            )
        
        return ExtractionResult(field_name=config.name)
    
    def extract_field(
        self, text: str, config: FieldConfig, entities: list
    ) -> ExtractionResult:
        """
        Pipeline complet d'extraction pour un champ.
        
        Essaie chaque stratégie dans l'ordre et garde le meilleur score.
        """
        best = ExtractionResult(field_name=config.name)
        
        # Ordre des stratégies selon le type de champ
        strategies = []
        
        if config.value_type == "enum":
            strategies.append(self._extract_with_enum)
        
        if config.patterns:
            strategies.extend([
                self._extract_with_regex,
                self._extract_with_flexible_regex,
            ])
        
        strategies.append(self._extract_with_fuzzy_keyword)
        
        for strategy in strategies:
            try:
                if strategy in (self._extract_with_enum,):
                    result = strategy(text, config)
                else:
                    result = strategy(text, config)
                
                if not result.is_empty and result.confidence > best.confidence:
                    best = result
                    # Early exit si très haute confiance
                    if best.confidence >= 0.92:
                        break
            except Exception as e:
                logger.debug(
                    f"Erreur stratégie {strategy.__name__} "
                    f"pour {config.name}: {e}"
                )
                continue
        
        # NER fallback si rien trouvé ou faible confiance
        if best.confidence < 0.5 and config.ner_fallback:
            ner_result = self._extract_with_ner(config, entities)
            if not ner_result.is_empty and ner_result.confidence > best.confidence:
                best = ner_result
        
        return best
    
    def extract_all(
        self, text: str, entities: list
    ) -> dict:
        """
        Extrait tous les champs configurés.
        
        Returns:
            Dict avec les valeurs extraites et les métadonnées de confiance.
        """
        results = {}
        confidence_map = {}
        
        for config in FIELD_CONFIGS:
            result = self.extract_field(text, config, entities)
            results[config.name] = result.value if not result.is_empty else ""
            if not result.is_empty:
                confidence_map[config.name] = {
                    "confidence": result.confidence,
                    "method": result.method,
                }
        
        return results, confidence_map


# ═══════════════════════════════════════════════════════════════════
# Extracteurs spécialisés (pour les champs qui nécessitent
# une logique spécifique non couverte par le moteur générique)
# ═══════════════════════════════════════════════════════════════════

def _extract_date_from_text(text: str) -> str:
    """
    Extraction spécialisée de dates.
    Gère les formats numériques (dd/mm/yyyy) et textuels (23 فبراير 2010).
    """
    text_c = normalize_digits(text)
    
    # Format numérique
    m = re.search(r'(\d{1,2}[/\-.](?:0?[1-9]|1[0-2])[/\-.]\d{4})', text_c)
    if m:
        return m.group(1)
    
    # Format textuel avec mois arabe
    for month_name, month_num in ARABIC_MONTHS.items():
        # Pattern flexible pour le nom du mois
        month_flex = build_flexible_pattern(month_name)
        pattern = rf'(\d{{1,2}})\s+{month_flex}\s+(\d{{4}})'
        m = re.search(pattern, text_c, re.UNICODE)
        if m:
            return f"{m.group(1)} {month_name} {m.group(2)}"
    
    return ""


def _extract_year_fallback(text: str) -> str:
    """Extraction de l'année en fallback (cherche 20XX)."""
    text_c = normalize_digits(text)
    m = re.search(r'(20[12]\d)', text_c)
    return m.group(1) if m else ""


def _extract_framework_fallback(text: str) -> str:
    """Vérifie si le document mentionne 'إطارية' même sans structure."""
    text_norm = normalize_for_matching(text)
    if re.search(r'اطاريه|اتفاقيه\s+اطار', text_norm):
        return "إطارية"
    return ""


# ═══════════════════════════════════════════════════════════════════
# Fonction principale — API publique
# ═══════════════════════════════════════════════════════════════════

# Singleton du moteur d'extraction
_engine = ExtractionEngine()


def clean_output(text: str, entities: list) -> dict:
    """
    Post-traitement: extrait les paires clé-valeur d'une convention marocaine.
    
    Retourne un JSON structuré avec tous les champs et métadonnées.
    
    Args:
        text: Texte OCR complet
        entities: Entités NER extraites par nlp.py
    
    Returns:
        dict avec les champs extraits + raw_text + _confidence
    """
    logger.info("--- Post-traitement convention (v2 - hybrid engine) ---")
    
    if not text:
        logger.warning("Texte vide, pas de post-traitement")
        return {config.name: "" for config in FIELD_CONFIGS}
    
    # Pré-normaliser le texte pour l'affichage
    display_text = normalize_for_display(text)
    
    # Extraction hybride de tous les champs
    data, confidence_map = _engine.extract_all(display_text, entities)
    
    # --- Post-corrections spécialisées ---
    
    # Date: si pas trouvée par le moteur, fallback spécialisé
    if not data.get("تاريخ_البداية"):
        date_val = _extract_date_from_text(display_text)
        if date_val:
            data["تاريخ_البداية"] = date_val
            confidence_map["تاريخ_البداية"] = {
                "confidence": 0.65, "method": "date_fallback"
            }
    
    # Année: fallback cherche 20XX
    if not data.get("السنة"):
        year_val = _extract_year_fallback(display_text)
        if year_val:
            data["السنة"] = year_val
            confidence_map["السنة"] = {
                "confidence": 0.50, "method": "year_fallback"
            }
    
    # Cadre/framework: fallback simple
    if not data.get("الإطارية"):
        fw_val = _extract_framework_fallback(display_text)
        if fw_val:
            data["الإطارية"] = fw_val
            confidence_map["الإطارية"] = {
                "confidence": 0.60, "method": "framework_fallback"
            }
    
    # --- Logging ---
    filled = {k: v for k, v in data.items() if v}
    empty = [k for k, v in data.items() if not v]
    logger.info(f"Champs remplis ({len(filled)}): {list(filled.keys())}")
    logger.info(f"Champs vides ({len(empty)}): {empty}")
    
    # Log confiance
    for field_name, meta in confidence_map.items():
        logger.info(
            f"  → {field_name}: conf={meta['confidence']:.2f} "
            f"method={meta['method']} val='{data[field_name][:50]}'"
        )
    
    # Ajouter métadonnées
    data["raw_text"] = text[:3000] if text else ""
    data["_confidence"] = confidence_map
    
    return data