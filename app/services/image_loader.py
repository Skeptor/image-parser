import asyncio
import logging
import tempfile
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

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

    parsed = urlparse(url)
    site_origin = f"{parsed.scheme}://{parsed.netloc}/"

    base_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Each attempt is (headers_override, description). We try progressively
    # simpler header sets — some servers block Chrome UAs without the full
    # browser header suite (sec-fetch-*, etc.), so stripping down often works.
    header_attempts = [
        ({**base_headers, "Referer": site_origin}, "chrome-ua+site-referer"),
        ({**base_headers}, "chrome-ua+no-referer"),
        ({"Accept": base_headers["Accept"]}, "no-ua+no-referer"),
    ]

    content: bytes | None = None

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for headers, desc in header_attempts:
            response = await client.get(url, headers=headers)
            if response.status_code != 403:
                response.raise_for_status()
                content = response.content
                break
            logger.info("Got 403 with %s, retrying", desc)

    if content is None:
        logger.info("httpx failed with 403, falling back to curl")
        content = await _fetch_with_curl(url)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    if Path(tmp_path).stat().st_size > MAX_FILE_SIZE:
        raise ImageLoadError(f"Image exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024} MB")

    try:
        img = Image.open(tmp_path)
        img.load()
        return img
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _fetch_with_curl(url: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sL", "--max-time", "30",
            "-o", tmp_path,
            "-w", "%{http_code}",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        status_code = int(stdout.strip()) if stdout.strip().isdigit() else 0
        if status_code >= 400:
            raise ImageLoadError(f"Failed to fetch image: HTTP {status_code}")
        return Path(tmp_path).read_bytes()
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
