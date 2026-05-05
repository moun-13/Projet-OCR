"""
Conversion PDF → Images
- DPI 300 pour une meilleure qualité OCR sur les documents arabes
- Gestion d'erreurs pour les PDF corrompus
- Chemin Poppler configurable via variable d'environnement
"""
import os
import logging
import time
import numpy as np
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)

# Chemin Poppler configurable via env ou valeur par défaut
POPPLER_PATH = os.environ.get(
    "POPPLER_PATH",
    r"C:\Users\pc\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
)

# DPI élevé = meilleure résolution pour l'OCR arabe
PDF_DPI = 300


def convert_pdf_to_images(pdf_bytes: bytes) -> list:
    """
    Convertit un PDF (bytes) en liste d'images numpy (BGR).
    
    Args:
        pdf_bytes: Contenu du fichier PDF en bytes
        
    Returns:
        Liste d'images numpy arrays (format BGR pour OpenCV)
        
    Raises:
        ValueError: Si le PDF est vide ou invalide
    """
    if not pdf_bytes:
        raise ValueError("Le fichier PDF est vide")

    try:
        t = time.time()
        pil_images = convert_from_bytes(
            pdf_bytes,
            dpi=PDF_DPI,
            poppler_path=POPPLER_PATH,
            fmt="png",  # PNG = sans perte, meilleur pour l'OCR
        )

        # Convertir PIL → numpy (RGB → BGR pour OpenCV)
        images = []
        for img in pil_images:
            np_img = np.array(img)
            images.append(np_img)

        logger.info(
            f"📄 PDF converti: {len(images)} page(s) à {PDF_DPI} DPI "
            f"en {time.time() - t:.1f}s"
        )
        return images

    except Exception as e:
        logger.error(f"❌ Erreur conversion PDF: {e}")
        raise ValueError(f"Impossible de convertir le PDF: {e}")