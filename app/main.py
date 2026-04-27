import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile

from app.models import ParseResponse
from app.services.image_loader import ImageLoadError, load_from_upload, load_from_url
from app.services.ocr import OCRError, run_ocr
from app.services.schedule_extractor import extract_schedule
from app.services.table_parser import detect_tables
from app.utils import clean_group_name, clean_stage_name, compute_confidence

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
        raw_text = run_ocr(pil_image, lang=lang, psm=psm)
    except OCRError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

    if not raw_text:
        return ParseResponse(
            success=False,
            error="No text detected in the image",
            confidence=0.0,
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            table_data = detect_tables(pil_image, output_path)

        schedule = extract_schedule(raw_text, table_data)

        for stage_name in list(schedule.stages.keys()):
            cleaned = clean_stage_name(stage_name)
            schedule.stages[cleaned] = schedule.stages.pop(stage_name)
            for slot in schedule.stages[cleaned]:
                slot.group = clean_group_name(slot.group)

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
