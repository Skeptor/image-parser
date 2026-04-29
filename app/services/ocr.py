import logging
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

import pytesseract

logger = logging.getLogger(__name__)


class OCRError(Exception):
    pass


def check_tesseract() -> bool:
    """Check if Tesseract is available."""
    try:
        subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_ocr(
    image: Image.Image,
    lang: str = "eng",
    psm: int = 6,
) -> str:
    """Run Tesseract OCR on an image and return raw text."""
    if not check_tesseract():
        raise OCRError("Tesseract OCR is not installed or not in PATH")

    logger.info("Running OCR with lang=%s psm=%d", lang, psm)
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text.strip()


def run_ocr_with_boxes(
    image: Image.Image,
    lang: str = "eng",
    psm: int = 6,
) -> list[dict]:
    """Run OCR and return word-level data with bounding boxes."""
    if not check_tesseract():
        raise OCRError("Tesseract OCR is not installed or not in PATH")

    logger.info("Running OCR with word boxes, lang=%s psm=%d", lang, psm)
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or data["conf"][i] < 20:
            continue
        words.append(
            {
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "confidence": data["conf"][i],
            }
        )
    return words


def run_ocr_column(
    column_image: Image.Image,
    lang: str = "eng",
    psm: int = 6,
) -> str:
    """Run OCR on a single cropped column. Uses PSM 6 for a single column of text."""
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(column_image, lang=lang, config=config)
    return text.strip()
