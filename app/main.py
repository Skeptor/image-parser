import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile

from app.models import ParseResponse
from app.services.column_detector import crop_column, detect_columns, get_column_text_from_words
from app.services.image_loader import ImageLoadError, load_from_upload, load_from_url
from app.services.ocr import OCRError, run_ocr, run_ocr_column
from app.services.schedule_extractor import extract_schedule
from app.utils import compute_confidence

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/parse", response_model=ParseResponse)
async def parse_concert_schedule(
    image: UploadFile | None = None,
    url: str | None = Form(None),
    lang: str = Query("eng", description="Tesseract language code"),
    psm: int = Query(6, description="Tesseract page segmentation mode"),
):
    if not image and not url:
        raise HTTPException(
            status_code=400,
            detail="Either 'image' (file upload) or 'url' must be provided",
        )

    try:
        pil_image = (
            load_from_upload(image.file, image.filename)
            if image
            else await load_from_url(url)
        )
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    try:
        columns = detect_columns(pil_image)

        if len(columns) > 1:
            logger.info("Multi-column layout detected, using word boxes for %d columns", len(columns))

            # Find header y position to skip header words in content
            header_y_max = 0
            for col in columns:
                for w in col.words:
                    if w["top"] < pil_image.height * 0.15:  # Words in top 15%
                        header_y_max = max(header_y_max, w["top"] + w["height"])

            column_data = []
            column_texts = []
            for col in columns:
                # Reconstruct text from word boxes (skipping header region)
                text = get_column_text_from_words(col, skip_header_y=header_y_max)
                column_texts.append(text)
                # Filter words to exclude header region
                content_words = [w for w in col.words if w["top"] > header_y_max]
                column_data.append({"text": text, "header": col.header, "words": content_words})
                logger.info("Column %d (header=%s): %d words, text preview: %s",
                           col.index, col.header, len(col.words), text[:80].replace("\n", " ") + "...")

            raw_text = "\n---COLUMN BREAK---\n".join(column_texts)
            schedule = extract_schedule(raw_text, column_data=column_data)
        else:
            raw_text = run_ocr(pil_image, lang=lang, psm=psm)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir)
                from app.services.table_parser import detect_tables

                table_data = detect_tables(pil_image, output_path)
            schedule = extract_schedule(raw_text, table_data=table_data)

    except OCRError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Parse pipeline failed")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

    if not raw_text:
        return ParseResponse(
            success=False,
            error="No text detected in the image",
            confidence=0.0,
        )

    try:
        confidence = compute_confidence(schedule)

        return ParseResponse(
            success=True,
            data=schedule,
            confidence=confidence,
            raw_text=raw_text,
        )

    except Exception as e:
        logger.exception("Schedule extraction failed")
        return ParseResponse(
            success=False,
            error=f"Failed to parse schedule: {e}",
            confidence=0.0,
            raw_text=raw_text,
        )
