import logging
import tempfile
from pathlib import Path
from typing import BinaryIO

import httpx
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    pass


def validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file type '{ext}'. Allowed: {SUPPORTED_EXTENSIONS}"
        )


async def load_from_url(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image."""
    logger.info("Fetching image from URL: %s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type:
        logger.warning(
            "Content-Type '%s' is not an image type, attempting anyway",
            content_type,
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    if Path(tmp_path).stat().st_size > MAX_FILE_SIZE:
        raise ImageLoadError(f"Image exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024} MB")

    try:
        img = Image.open(tmp_path)
        img.load()
        return img
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_from_upload(file: BinaryIO, filename: str) -> Image.Image:
    """Read an uploaded file and return a PIL Image."""
    validate_extension(filename)
    logger.info("Loading uploaded image: %s", filename)

    data = file.read()

    if len(data) > MAX_FILE_SIZE:
        raise ImageLoadError(f"Image exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024} MB")

    img = Image.open(file)
    img.load()
    return img
