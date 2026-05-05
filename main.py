"""
Backend FastAPI pour le traitement de documents administratifs marocains
- Pipeline: PDF → Images → Prétraitement → OCR → NLP → Post-traitement
- Gestion d'erreurs complète
- Validation des fichiers uploadés
- Logs structurés pour le débogage
"""
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.pdf_to_image import convert_pdf_to_images
from Services.preprocess import preprocess
from Services.ocr import run_ocr, get_reader
from Services.nlp import extract_entities, get_ner
from Services.postprocess import clean_output

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Taille max du fichier (20 MB)
MAX_FILE_SIZE = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-charger les modeles au demarrage."""
    logger.info("--- Chargement des modeles (2-3 min) ---")
    total = time.time()

    t = time.time()
    get_reader()
    logger.info(f"EasyOCR charge en {time.time() - t:.1f}s")

    t = time.time()
    get_ner()
    logger.info(f"BERT NER charge en {time.time() - t:.1f}s")

    logger.info(f"Serveur pret ! (total: {time.time() - total:.1f}s)")
    yield
    logger.info("Arret du serveur")


app = FastAPI(
    title="API Extraction Documents Marocains",
    description="OCR + NLP pour documents administratifs arabes/francais",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Endpoint principal: extrait les informations d'un PDF arabe.
    
    Retourne: montant, date, organisations, personnes, lieux, reference, raw_text
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] === Nouvelle requete: {file.filename} ===")
    total_start = time.time()

    # --- Validation ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporte: {file.filename}. Seuls les PDF sont acceptes."
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        logger.error(f"[{request_id}] Erreur lecture fichier: {e}")
        raise HTTPException(status_code=400, detail="Impossible de lire le fichier")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Le fichier est vide")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux ({len(pdf_bytes) // 1024 // 1024} MB). Max: {MAX_FILE_SIZE // 1024 // 1024} MB"
        )

    logger.info(f"[{request_id}] Fichier valide: {len(pdf_bytes) // 1024} KB")

    try:
        # 1. PDF → Images
        t = time.time()
        images = convert_pdf_to_images(pdf_bytes)
        logger.info(f"[{request_id}] 1/5 PDF -> {len(images)} images en {time.time() - t:.1f}s")

        # 2. Pretraitement
        t = time.time()
        processed_images = [preprocess(img) for img in images]
        logger.info(f"[{request_id}] 2/5 Pretraitement en {time.time() - t:.1f}s")

        # 3. OCR
        t = time.time()
        text = run_ocr(processed_images)
        logger.info(f"[{request_id}] 3/5 OCR en {time.time() - t:.1f}s — {len(text)} caracteres")

        if not text.strip():
            logger.warning(f"[{request_id}] OCR n'a detecte aucun texte!")
            return {
                "warning": "Aucun texte detecte dans le document",
                "raw_text": "",
                "processing_time": round(time.time() - total_start, 1),
            }

        # 4. NLP (BERT arabe)
        t = time.time()
        entities = extract_entities(text)
        logger.info(f"[{request_id}] 4/5 NLP en {time.time() - t:.1f}s — {len(entities)} entites")

        # 5. Post-traitement
        t = time.time()
        result = clean_output(text, entities)
        logger.info(f"[{request_id}] 5/5 Post-traitement en {time.time() - t:.1f}s")

        # Ajouter les metadonnees
        result["processing_time"] = round(time.time() - total_start, 1)
        result["pages"] = len(images)

        logger.info(
            f"[{request_id}] === Termine en {result['processing_time']}s === "
            f"Champs: {[k for k in result.keys() if k != 'raw_text']}"
        )

        return result

    except ValueError as e:
        logger.error(f"[{request_id}] Erreur de valeur: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] Erreur interne: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )