"""
OCR optimisé pour documents arabes scannés (EasyOCR).

Optimisations clés :
- decoder='beamsearch' : plus précis que le décodeur glouton par défaut
- beamWidth=10 : explore plus de possibilités pour les mots arabes
- paragraph=False : on gère l'ordre nous-mêmes (RTL)
- Test sur 3 versions d'image, garde la meilleure
- Nettoyage Unicode du texte arabe (normalisation via arabic_utils)
"""
import re
import logging
import time
import os

from Services.arabic_utils import normalize_for_display

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

reader = None
paddle_reader = None


def get_ocr_engine_name() -> str:
    """Retourne le moteur OCR demandé via OCR_ENGINE."""
    return os.getenv("OCR_ENGINE", "easyocr").strip().lower()


def get_reader():
    """Charge le modèle EasyOCR une seule fois (arabe + anglais)."""
    global reader
    if reader is None:
        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError(
                "EasyOCR n'est pas installe. Installer easyocr "
                "ou utiliser OCR_ENGINE=paddle avec PaddleOCR installe."
            ) from e

        logger.info("Chargement EasyOCR (ar + en)...")
        t = time.time()
        reader = easyocr.Reader(
            ['ar', 'en'],  # Arabe + Anglais (fr non compatible)
            gpu=False,
            verbose=False,
        )
        logger.info(f"EasyOCR charge en {time.time() - t:.1f}s")
    return reader


def get_paddle_reader():
    """
    Charge PaddleOCR si disponible.
    Installation optionnelle:
        pip install paddleocr paddlepaddle
    """
    global paddle_reader
    if paddle_reader is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                "PaddleOCR n'est pas installe. Installer paddleocr/paddlepaddle "
                "ou utiliser OCR_ENGINE=easyocr."
            ) from e

        logger.info("Chargement PaddleOCR (ar + latin)...")
        t = time.time()
        paddle_reader = PaddleOCR(
            lang="arabic",
            use_angle_cls=True,
            show_log=False,
        )
        logger.info(f"PaddleOCR charge en {time.time() - t:.1f}s")
    return paddle_reader


# Nettoyage du texte arabe post-OCR

def _normalize_arabic(text: str) -> str:
    """
    Normalise le texte arabe pour corriger les erreurs OCR courantes.
    Délègue à arabic_utils.normalize_for_display pour la logique centralisée.
    """
    return normalize_for_display(text)


def _clean_ocr_line(text: str) -> str:
    """
    Nettoie une ligne de texte OCR.
    Supprime les caractères isolés incohérents et les artefacts.
    """
    if not text or len(text.strip()) < 2:
        return ""

    # Supprimer les caractères isolés non-arabes/non-numériques
    # qui sont souvent des erreurs OCR (ex: "ا ل م ك ت ب" → garder)
    cleaned = text.strip()

    # Si le texte est majoritairement des symboles/ponctuation, l'ignorer
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', cleaned))
    latin_chars = len(re.findall(r'[a-zA-Z]', cleaned))
    digit_chars = len(re.findall(r'[0-9٠-٩]', cleaned))
    useful_chars = arabic_chars + latin_chars + digit_chars

    if len(cleaned) > 3 and useful_chars < len(cleaned) * 0.3:
        return ""  # Trop peu de caractères utiles

    return cleaned


# OCR principal

def _ocr_single_image(ocr_reader, image, min_confidence: float = 0.25) -> list:
    """
    Exécute l'OCR sur une seule image avec des paramètres optimisés
    pour l'arabe.
    """
    try:
        results = ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,     # Pas de fusion en paragraphes (on gère RTL)
            decoder='beamsearch', # Plus précis que 'greedy' pour l'arabe
            beamWidth=10,        # Largeur du faisceau (plus = plus précis mais plus lent)
            batch_size=1,        # CPU: batch_size=1 est optimal
            contrast_ths=0.1,    # Seuil de contraste bas (scans peu contrastés)
            adjust_contrast=0.5, # Ajustement automatique du contraste
            text_threshold=0.6,  # Confiance min pour détecter du texte
            low_text=0.3,        # Score bas pour le texte
            link_threshold=0.3,  # Seuil de liaison entre caractères
            mag_ratio=1.5,       # Ratio d'agrandissement interne
            slope_ths=0.2,       # Tolérance pour les lignes inclinées
            width_ths=0.8,       # Fusion horizontale des boîtes
        )

        filtered = []
        skipped = 0
        for bbox, text, confidence in results:
            cleaned = _clean_ocr_line(text)
            if confidence >= min_confidence and cleaned:
                normalized = _normalize_arabic(cleaned)
                if normalized:
                    filtered.append((normalized, confidence))
            else:
                skipped += 1

        if skipped > 0:
            logger.debug(f"  {skipped} resultat(s) ignore(s) (confiance < {min_confidence})")

        return filtered

    except Exception as e:
        logger.error(f"Erreur OCR sur image: {e}")
        return []


def _flatten_paddle_results(results) -> list:
    """Gere les variantes de format retournees par PaddleOCR."""
    if not results:
        return []

    # PaddleOCR recent: [ [ [bbox, (text, score)], ... ] ]
    if len(results) == 1 and isinstance(results[0], list):
        first = results[0]
        if not first:
            return []
        if isinstance(first[0], list) and len(first[0]) >= 2:
            return first

    # Format plus ancien: [ [bbox, (text, score)], ... ]
    return results


def _ocr_single_image_paddle(paddle_ocr, image, min_confidence: float = 0.25) -> list:
    """Execute PaddleOCR sur une image."""
    try:
        raw_results = paddle_ocr.ocr(image, cls=True)
        filtered = []

        for item in _flatten_paddle_results(raw_results):
            if not item or len(item) < 2:
                continue
            payload = item[1]
            if not isinstance(payload, (tuple, list)) or len(payload) < 2:
                continue

            text, confidence = payload[0], float(payload[1])
            cleaned = _clean_ocr_line(str(text))
            if confidence >= min_confidence and cleaned:
                normalized = _normalize_arabic(cleaned)
                if normalized:
                    filtered.append((normalized, confidence))

        return filtered

    except Exception as e:
        logger.error(f"Erreur PaddleOCR sur image: {e}")
        return []


def _run_ocr_on_image(ocr_reader, preprocessed: dict, page_num: int, engine: str) -> str:
    """
    Exécute l'OCR sur toutes les versions d'une image prétraitée.
    Compare les résultats et garde la version avec le meilleur score.

    Le score combine :
    - La confiance moyenne de l'OCR
    - Le nombre de caractères arabes détectés (plus = mieux)
    """
    results = {}

    for version_name, image in preprocessed.items():
        if engine == "paddle":
            ocr_results = _ocr_single_image_paddle(ocr_reader, image)
        else:
            ocr_results = _ocr_single_image(ocr_reader, image)
        if ocr_results:
            avg_conf = sum(c for _, c in ocr_results) / len(ocr_results)
            total_text = " ".join(t for t, _ in ocr_results)

            # Compter les caractères arabes (indicateur de qualité)
            arabic_count = len(re.findall(r'[\u0600-\u06FF]', total_text))

            # Score combiné: confiance * log(nombre de caractères arabes + 1)
            import math
            quality_score = avg_conf * math.log(arabic_count + 1, 10) if arabic_count > 0 else 0

            results[version_name] = {
                "text": total_text,
                "avg_confidence": avg_conf,
                "arabic_chars": arabic_count,
                "quality_score": quality_score,
                "count": len(ocr_results),
            }
            logger.debug(
                f"  Page {page_num} [{version_name}]: "
                f"{len(ocr_results)} blocs, conf={avg_conf:.2f}, "
                f"arabe={arabic_count} chars, score={quality_score:.2f}"
            )

    if not results:
        logger.warning(f"Page {page_num}: aucun texte detecte")
        return ""

    # Choisir la version avec le meilleur score de qualité
    best_version = max(results.keys(), key=lambda k: results[k]["quality_score"])
    best = results[best_version]

    logger.info(
        f"Page {page_num}: '{best_version}' choisie — "
        f"{best['count']} blocs, confiance {best['avg_confidence']:.2f}, "
        f"{best['arabic_chars']} caracteres arabes"
    )

    # Aperçu du texte
    preview = best["text"][:200].replace("\n", " ")
    logger.info(f"  Apercu: {preview}")

    return best["text"]


def run_ocr(preprocessed_images: list) -> str:
    """
    Exécute l'OCR sur toutes les images et retourne le texte complet.
    """
    engine = get_ocr_engine_name()
    if engine == "paddle":
        try:
            ocr_reader = get_paddle_reader()
        except RuntimeError as e:
            logger.warning(f"{e} Retour a EasyOCR.")
            engine = "easyocr"
            ocr_reader = get_reader()
    else:
        engine = "easyocr"
        ocr_reader = get_reader()
    all_text = []

    t = time.time()
    for i, preprocessed in enumerate(preprocessed_images):
        page_text = _run_ocr_on_image(
            ocr_reader,
            preprocessed,
            page_num=i + 1,
            engine=engine,
        )
        if page_text:
            all_text.append(page_text)

    full_text = "\n".join(all_text)

    # Nettoyage final du texte complet
    full_text = _normalize_arabic(full_text)

    logger.info(
        f"OCR termine ({engine}): {len(all_text)} page(s), "
        f"{len(full_text)} caracteres en {time.time() - t:.1f}s"
    )

    return full_text
