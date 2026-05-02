import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
from PIL import Image

from app.models import DaySchedule, ParseResponse, StageSchedule
from app.services.column_detector import (
    crop_column,
    detect_columns,
    detect_day_regions,
    get_column_text_from_words,
)
from app.services.image_loader import ImageLoadError, load_from_upload, load_from_url
from app.services.ocr import OCRError, run_ocr, run_ocr_column
from app.services.schedule_extractor import extract_schedule
from app.utils import compute_confidence

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


def _parse_image_section(pil_image: Image.Image, lang: str, psm: int) -> tuple[StageSchedule, str]:
    """Run the full parse pipeline on one image (or one day's crop).

    Returns (schedule, raw_text).
    """
    columns = detect_columns(pil_image)

    if len(columns) > 1:
        logger.info("Multi-column layout detected (%d columns)", len(columns))

        header_y_max = 0
        for col in columns:
            for w in col.words:
                if w["top"] < pil_image.height * 0.15:
                    header_y_max = max(header_y_max, w["top"] + w["height"])

        column_data = []
        column_texts = []
        for col in columns:
            text = get_column_text_from_words(col, skip_header_y=header_y_max)
            column_texts.append(text)
            content_words = [w for w in col.words if w["top"] > header_y_max]
            column_data.append({"text": text, "header": col.header, "words": content_words})
            logger.info(
                "Column %d (header=%s): %d words, preview: %s",
                col.index, col.header, len(col.words),
                text[:80].replace("\n", " ") + "...",
            )

        raw_text = "\n---COLUMN BREAK---\n".join(column_texts)
        schedule = extract_schedule(raw_text, column_data=column_data)
    else:
        # Single-column: reuse the words already collected by detect_columns (multi-pass OCR)
        # instead of calling run_ocr() again with just one pass.
        col = columns[0]
        if col.words:
            raw_text = get_column_text_from_words(col, skip_header_y=0)
            column_data = [{"text": raw_text, "header": col.header, "words": col.words}]
            schedule = extract_schedule(raw_text, column_data=column_data)
        else:
            raw_text = run_ocr(pil_image, lang=lang, psm=psm)
            with tempfile.TemporaryDirectory() as tmpdir:
                from app.services.table_parser import detect_tables
                table_data = detect_tables(pil_image, Path(tmpdir))
            schedule = extract_schedule(raw_text, table_data=table_data)

    return schedule, raw_text


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
        day_regions = detect_day_regions(pil_image)

        if day_regions and len(day_regions) >= 2:
            logger.info("Multi-day layout detected: %s", [r.day_name for r in day_regions])

            day_schedules: list[DaySchedule] = []
            all_raw_texts: list[str] = []
            merged_stages: dict = {}

            for region in day_regions:
                day_image = pil_image.crop(region.crop)
                day_sched, day_raw = _parse_image_section(day_image, lang, psm)
                all_raw_texts.append(f"=== {region.day_name} ===\n{day_raw}")

                day_schedules.append(DaySchedule(
                    day_name=region.day_name,
                    date=day_sched.date,
                    stages=day_sched.stages,
                ))

                # Flat merge: suffix stage names with day to avoid key collisions
                for stage_name, slots in day_sched.stages.items():
                    key = f"{stage_name} ({region.day_name.capitalize()})"
                    merged_stages[key] = slots

            raw_text = "\n\n".join(all_raw_texts)
            schedule = StageSchedule(stages=merged_stages, days=day_schedules)

        else:
            schedule, raw_text = _parse_image_section(pil_image, lang, psm)

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
