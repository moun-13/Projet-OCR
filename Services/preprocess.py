"""
Prétraitement d'images optimisé pour l'OCR de documents arabes scannés.

Principes clés pour l'arabe :
- MOINS de traitement = MIEUX. L'arabe est cursif avec des traits fins
  qui relient les lettres → le seuillage agressif les détruit.
- CLAHE pour le contraste (au lieu de binarisation directe)
- Morphologie pour reconnecter les lettres cassées
- L'image en niveaux de gris est souvent meilleure que la binarisée
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def deskew(image: np.ndarray) -> np.ndarray:
    """Corrige l'inclinaison d'un scan penché."""
    try:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        if lines is None or len(lines) == 0:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        logger.info(f"Inclinaison corrigee: {median_angle:.1f} degres")
        return corrected
    except Exception as e:
        logger.warning(f"Deskew echoue: {e}")
        return image


def upscale_if_small(image: np.ndarray, min_height: int = 2500) -> np.ndarray:
    """
    Agrandit les petites images. Essentiel pour l'arabe car les traits
    fins deviennent invisibles en basse résolution.
    """
    h, w = image.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Image agrandie: {w}x{h} -> {new_w}x{new_h}")
    return image


def enhance_contrast_clahe(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.
    Bien meilleur que l'égalisation globale pour les scans inégaux.
    Améliore le contraste localement sans détruire les détails fins.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Accentuation légère pour rendre les lettres arabes plus nettes.
    Important pour les scans flous où les liaisons entre lettres sont vagues.
    """
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ])
    return cv2.filter2D(image, -1, kernel)


def remove_noise_gentle(image: np.ndarray) -> np.ndarray:
    """
    Débruitage doux qui préserve les bords.
    h=6 est doux (h=10+ est trop agressif pour l'arabe).
    """
    return cv2.fastNlMeansDenoising(image, h=6, templateWindowSize=7, searchWindowSize=21)


def fix_broken_characters(binary: np.ndarray) -> np.ndarray:
    """
    Morphologie pour reconnecter les lettres arabes cassées par le scan.
    Utilise un kernel horizontal car l'arabe se lit horizontalement.
    """
    # Kernel horizontal léger pour reconnecter les traits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def remove_borders_and_lines(image: np.ndarray) -> np.ndarray:
    """
    Supprime les bordures et lignes horizontales/verticales des tableaux
    qui perturbent l'OCR. Courant dans les documents administratifs.
    """
    h, w = image.shape[:2]

    # Détecter et supprimer les lignes horizontales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    horizontal_lines = cv2.morphologyEx(
        cv2.bitwise_not(image), cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    # Détecter et supprimer les lignes verticales
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8))
    vertical_lines = cv2.morphologyEx(
        cv2.bitwise_not(image), cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Combiner les lignes détectées
    lines_mask = cv2.add(horizontal_lines, vertical_lines)
    # Supprimer les lignes de l'image
    cleaned = cv2.add(image, lines_mask)

    return cleaned


def preprocess(image: np.ndarray) -> dict:
    """
    Prétraite une image pour l'OCR arabe.

    Stratégie : produire TROIS versions de l'image et laisser l'OCR
    choisir la meilleure (celle avec la confiance la plus haute).

    Versions:
    - 'enhanced': CLAHE + accentuation (meilleur pour la plupart des scans)
    - 'clean': débruitage + suppression des lignes (pour les tableaux)
    - 'gray': niveaux de gris simple (fallback, parfois le meilleur)
    """
    try:
        # 1. Agrandir si nécessaire
        image = upscale_if_small(image)

        # 2. Niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 3. Correction d'inclinaison
        gray = deskew(gray)

        # === Version 'enhanced' : contraste + netteté ===
        enhanced = enhance_contrast_clahe(gray)
        enhanced = sharpen(enhanced)

        # === Version 'clean' : débruitée + sans lignes ===
        denoised = remove_noise_gentle(gray)
        clean = remove_borders_and_lines(denoised)

        logger.info(f"Pretraitement: image {gray.shape[1]}x{gray.shape[0]}, 3 versions generees")

        return {
            "enhanced": enhanced,
            "clean": clean,
            "gray": gray,
        }

    except Exception as e:
        logger.error(f"Erreur pretraitement: {e}")
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return {"gray": gray}