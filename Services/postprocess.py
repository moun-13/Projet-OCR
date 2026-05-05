"""
Post-traitement pour conventions administratives marocaines.
Extrait les paires clé-valeur spécifiques au format JSON structuré.
"""
import re
import logging

logger = logging.getLogger(__name__)

_AR = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'

ARABIC_DIGIT_MAP = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

ARABIC_MONTHS = {
    'يناير': '01', 'فبراير': '02', 'مارس': '03', 'أبريل': '04',
    'ماي': '05', 'يونيو': '06', 'يوليوز': '07', 'غشت': '08',
    'شتنبر': '09', 'أكتوبر': '10', 'نونبر': '11', 'دجنبر': '12',
    'ابريل': '04', 'يوليو': '07', 'اغسطس': '08',
    'سبتمبر': '09', 'اكتوبر': '10', 'نوفمبر': '11', 'ديسمبر': '12',
}


def _conv(text):
    """Convertit chiffres arabes → occidentaux."""
    for ar, west in ARABIC_DIGIT_MAP.items():
        text = text.replace(ar, west)
    return text


def _clean(val):
    """Nettoie une valeur extraite."""
    if not val:
        return ""
    val = val.strip()
    val = re.sub(r'[\s\u200f]+', ' ', val)
    val = re.sub(r'^[:\s،.]+|[:\s،.]+$', '', val)
    return val.strip()


def _search_near_keyword(text, keywords, max_chars=150):
    """
    Cherche une valeur après un mot-clé arabe.
    Utilise des limites de mots pour éviter les correspondances partielles.
    Ex: 'المجال' ne doit pas matcher dans 'المجالات' ou 'بالمجال'.
    """
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text):
            start, end = m.start(), m.end()
            # Limite de mot AVANT : pas de caractère arabe juste avant
            if start > 0 and re.match(rf'[{_AR}]', text[start - 1]):
                continue
            # Limite de mot APRÈS : pas de caractère arabe juste après
            if end < len(text) and re.match(rf'[{_AR}]', text[end]):
                continue
            # Extraire la valeur après le mot-clé
            after = text[end:]
            val_match = re.match(
                rf'\s*[:\s\-ـ]*\s*(.{{1,{max_chars}}}?)(?:\n|$|[.،؛](?:\s|$))',
                after, re.UNICODE
            )
            if val_match:
                val = _clean(val_match.group(1))
                if val and len(val) > 1:
                    return val
    return ""


def _extract_agreement_number(text):
    """رقم الاتفاقية"""
    patterns = [
        r'(?:اتفاقية|الاتفاقية)\s*(?:رقم|عدد)\s*[:\s]*([^\n،.]{2,40})',
        r'(?:رقم|عدد)\s*(?:الاتفاقية|اتفاقية)\s*[:\s]*([^\n،.]{2,40})',
        r'(?:رقم|عدد)\s*[:\s]*([\d/\-]+(?:\s*/\s*\d+)*)',
    ]
    text_c = _conv(text)
    for p in patterns:
        m = re.search(p, text_c, re.UNICODE)
        if m:
            return _clean(m.group(1))
    return ""


def _extract_start_date(text):
    """تاريخ البداية"""
    text_c = _conv(text)
    # Chercher date près de mots-clés
    kw_patterns = [
        r'(?:تاريخ\s*(?:البداية|الانطلاق|بداية|السريان))\s*[:\s]*([\d/\-.\s]+)',
        r'(?:ابتداء\s*من|بتاريخ|في)\s*[:\s]*([\d]{1,2}[/\-.][\d]{1,2}[/\-.][\d]{4})',
        r'(?:بتاريخ|في)\s+(\d{1,2}\s+[' + _AR + r']+\s+\d{4})',
    ]
    for p in kw_patterns:
        m = re.search(p, text_c, re.UNICODE)
        if m:
            return _clean(m.group(1))
    # Fallback: première date trouvée
    m = re.search(r'(\d{1,2}[/\-.](?:0?[1-9]|1[0-2])[/\-.]\d{4})', text_c)
    if m:
        return m.group(1)
    for month_name in ARABIC_MONTHS:
        m = re.search(rf'(\d{{1,2}})\s+{month_name}\s+(\d{{4}})', text_c)
        if m:
            return f"{m.group(1)} {month_name} {m.group(2)}"
    return ""


def _extract_framework(text):
    """الإطارية - type cadre de l'accord"""
    patterns = [
        r'(?:اتفاقية|اتفاق)\s+(إطار(?:ية|ي)?[^،.\n]{0,60})',
        r'(إطارية|إطاري|اتفاقية\s+إطار)',
        r'(?:الإطار(?:ية|ي)?)\s*[:\s]*([^،.\n]{2,80})',
    ]
    for p in patterns:
        m = re.search(p, text, re.UNICODE)
        if m:
            return _clean(m.group(1))
    if re.search(r'إطارية|اتفاقية\s+إطار', text):
        return "إطارية"
    return ""


def _extract_year(text):
    """السنة"""
    text_c = _conv(text)
    kw = [r'(?:سنة|السنة|عام)\s*[:\s]*(\d{4})']
    for p in kw:
        m = re.search(p, text_c)
        if m:
            return m.group(1)
    m = re.search(r'(20[12]\d)', text_c)
    return m.group(1) if m else ""


def _extract_session(text):
    """الدورة"""
    return _search_near_keyword(text, ['الدورة', 'دورة'], 80)


def _extract_entity(text, entities):
    """الكيان الاحتصاري - entité administrative"""
    val = _search_near_keyword(text, [
        'الكيان', 'الجهة المعنية',
        'الطرف الأول', 'الطرف الاول'
    ], 100)
    # Valider : un nom d'entité ne doit pas être trop long
    if val and len(val) < 80:
        return val
    # Fallback: première organisation détectée par NER
    orgs = [e["word"] for e in entities if "ORG" in e.get("entity_group", "")]
    return orgs[0] if orgs else ""


def _extract_contribution(text):
    """مساهمة الجهة"""
    text_c = _conv(text)
    patterns = [
        r'(?:مساهمة\s*(?:الجهة)?)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
        r'(?:حصة|مساهمة)\s*(?:الجهة|المجلس)\s*[:\s]*([\d\s.,]+)',
        r'(?:مبلغ|المبلغ)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD))',
        r'(?:بمبلغ|قدره|يقدر\s*ب)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
        r'([\d.,]+\s*(?:درهم|DH|MAD))',
    ]
    for p in patterns:
        m = re.search(p, text_c, re.UNICODE)
        if m:
            return _clean(m.group(1))
    return ""


def _extract_domain(text):
    """المجال"""
    val = _search_near_keyword(text, [
        'المجال', 'مجال', 'القطاع', 'قطاع', 'الميدان'
    ], 60)
    # Valider : le domaine doit être court
    if val and len(val) < 60:
        return val
    return ""


def _extract_project_owner(text, entities):
    """صاحب المشروع"""
    val = _search_near_keyword(text, [
        'صاحب المشروع', 'صاحب مشروع', 'المستفيد', 'الجهة المستفيدة',
        'الطرف الثاني', 'الطرف التاني'
    ], 100)
    if val and len(val) < 80:
        return val
    # Fallback: première personne détectée par NER
    persons = [e["word"] for e in entities if "PER" in e.get("entity_group", "")]
    return persons[0] if persons else ""


def _extract_decision_number(text):
    """رقم القرار"""
    text_c = _conv(text)
    patterns = [
        r'(?:قرار|القرار)\s*(?:رقم|عدد)\s*[:\s]*([\d/\-A-Za-z]+)',
        r'(?:رقم|عدد)\s*(?:القرار|قرار)\s*[:\s]*([\d/\-A-Za-z]+)',
        r'(?:مقرر|المقرر)\s*(?:رقم|عدد)\s*[:\s]*([\d/\-A-Za-z]+)',
    ]
    for p in patterns:
        m = re.search(p, text_c, re.UNICODE)
        if m:
            return _clean(m.group(1))
    return ""


def _extract_partner(text, entities):
    """الشريك"""
    val = _search_near_keyword(text, [
        'الشريك', 'شريك', 'المتعاقد', 'المتعاقد معه',
        'الطرف الثاني', 'الطرف التاني'
    ], 100)
    # Valider : un nom de partenaire ne doit pas être trop long
    if val and len(val) < 80:
        return val
    # Fallback: deuxième organisation détectée par NER
    orgs = [e["word"] for e in entities if "ORG" in e.get("entity_group", "")]
    return orgs[1] if len(orgs) > 1 else ""


def _extract_jurisdiction(text):
    """الاختصاص"""
    return _search_near_keyword(text, [
        'الاختصاص', 'اختصاص', 'الصلاحية', 'الولاية'
    ], 100)


def _extract_validity(text):
    """سريان الاتفاقية"""
    val = _search_near_keyword(text, [
        'سريان', 'مدة الاتفاقية', 'مدة السريان',
        'مدة التنفيذ', 'صلاحية', 'المدة'
    ], 100)
    if val:
        return val
    text_c = _conv(text)
    m = re.search(r'(?:لمدة|مدة)\s*[:\s]*(\d+\s*(?:سنة|سنوات|أشهر|شهر|شهرا))', text_c, re.UNICODE)
    return _clean(m.group(0)) if m else ""


def _extract_agreement_type(text):
    """نوع الاتفاقية"""
    types_map = {
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
    }
    for phrase, label in types_map.items():
        if phrase in text:
            return label
    val = _search_near_keyword(text, ['نوع الاتفاقية', 'نوع الاتفاق', 'طبيعة'], 60)
    return val if val else ""


def _extract_programs(text):
    """البرامج"""
    val = _search_near_keyword(text, [
        'البرنامج', 'برنامج', 'البرامج', 'المشروع', 'مشروع'
    ], 100)
    if val and len(val) < 100:
        return val
    return ""


def _extract_status(text):
    """حالة الاتفاقية"""
    statuses = {
        'ساري المفعول': 'ساري المفعول',
        'سارية المفعول': 'سارية المفعول',
        'منتهية': 'منتهية',
        'منتهي': 'منتهي',
        'قيد التنفيذ': 'قيد التنفيذ',
        'قيد الانجاز': 'قيد الانجاز',
        'ملغاة': 'ملغاة',
        'معلقة': 'معلقة',
        'جديدة': 'جديدة',
    }
    for phrase, label in statuses.items():
        if phrase in text:
            return label
    val = _search_near_keyword(text, ['حالة', 'الحالة', 'الوضعية'], 60)
    return val if val else ""


def _extract_attachments(text):
    """المرفقات"""
    val = _search_near_keyword(text, ['المرفقات', 'مرفقات', 'الوثائق المرفقة', 'الملاحق'], 150)
    return val


def _extract_subject(text):
    """موضوع الاتفاقية"""
    val = _search_near_keyword(text, [
        'موضوع الاتفاقية', 'موضوع', 'الموضوع',
        'الهدف من', 'تهدف إلى', 'تهدف الى'
    ], 200)
    if val:
        return val
    # Chercher les patterns "من أجل" / "بهدف" / "بغرض"
    patterns = [
        rf'(?:من\s+أجل|بهدف|بغرض)\s+(.{{10,200}}?)(?:\n|$|[.،؛](?:\s|$))',
    ]
    for p in patterns:
        m = re.search(p, text, re.UNICODE)
        if m:
            return _clean(m.group(1))
    return ""


def _extract_total_amount(text):
    """المبلغ الإجمالي"""
    text_c = _conv(text)
    patterns = [
        r'(?:المبلغ\s*الإجمالي|المبلغ\s*الاجمالي)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
        r'(?:الغلاف\s*المالي|التكلفة\s*الإجمالية)\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD)?)',
        r'(?:بمبلغ|قدره)\s*(?:إجمالي)?\s*[:\s]*([\d\s.,]+\s*(?:درهم|DH|MAD))',
    ]
    for p in patterns:
        m = re.search(p, text_c, re.UNICODE)
        if m:
            return _clean(m.group(1))
    return ""


def _extract_parties(text, entities):
    """أطراف الاتفاقية"""
    # Chercher après les mots-clés
    val = _search_near_keyword(text, [
        'الأطراف', 'أطراف الاتفاقية', 'الموقعون', 'الموقعين'
    ], 200)
    if val:
        return val
    # Fallback: toutes les organisations NER
    orgs = [e["word"] for e in entities if "ORG" in e.get("entity_group", "")]
    if orgs:
        return " / ".join(orgs[:4])
    return ""


def clean_output(text, entities):
    """
    Post-traitement: extrait les paires clé-valeur d'une convention marocaine.
    Retourne un JSON structuré avec tous les champs.
    """
    logger.info("--- Post-traitement convention ---")

    data = {
        "رقم_الاتفاقية": _extract_agreement_number(text),
        "تاريخ_البداية": _extract_start_date(text),
        "الإطارية": _extract_framework(text),
        "السنة": _extract_year(text),
        "الدورة": _extract_session(text),
        "الكيان_الاحتصاري": _extract_entity(text, entities),
        "مساهمة_الجهة": _extract_contribution(text),
        "المبلغ_الإجمالي": _extract_total_amount(text),
        "المجال": _extract_domain(text),
        "موضوع_الاتفاقية": _extract_subject(text),
        "صاحب_المشروع": _extract_project_owner(text, entities),
        "رقم_القرار": _extract_decision_number(text),
        "الشريك": _extract_partner(text, entities),
        "الأطراف": _extract_parties(text, entities),
        "الاختصاص": _extract_jurisdiction(text),
        "سريان_الاتفاقية": _extract_validity(text),
        "نوع_الاتفاقية": _extract_agreement_type(text),
        "البرامج": _extract_programs(text),
        "حالة_الاتفاقية": _extract_status(text),
        "المرفقات": _extract_attachments(text),
    }

    # Log les champs trouvés
    filled = {k: v for k, v in data.items() if v}
    empty = [k for k, v in data.items() if not v]
    logger.info(f"Champs remplis ({len(filled)}): {list(filled.keys())}")
    logger.info(f"Champs vides ({len(empty)}): {empty}")

    # Ajouter le texte brut pour débogage
    data["raw_text"] = text[:3000] if text else ""

    return data