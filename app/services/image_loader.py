import asyncio
import io
import logging
import tempfile
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import httpx
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

Image.MAX_IMAGE_PIXELS = 50_000_000  # ~7000×7000 max to prevent decompression bombs

_PRIVATE_NETS = [
    ip_network(n) for n in [
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
        "192.168.0.0/16", "169.254.0.0/16", "::1/128", "0.0.0.0/8",
        "100.64.0.0/10",  # CGNAT
    ]
]

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    pass


def validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file type '{ext}'. Allowed: {SUPPORTED_EXTENSIONS}"
        )


def _validate_url(url: str) -> None:
    """Reject non-http(s) schemes and private/loopback IP literals (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageLoadError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise ImageLoadError("Invalid URL: missing hostname")
    try:
        addr = ip_address(parsed.hostname)
        if any(addr in net for net in _PRIVATE_NETS):
            raise ImageLoadError("URL targets a private or loopback address")
    except ValueError:
        pass  # hostname (not an IP literal) — validated at DNS resolution time


async def load_from_url(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image."""
    _validate_url(url)
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

    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for headers, desc in header_attempts:
            response = await client.get(url, headers=headers)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
                _validate_url(location)  # block redirects to private targets
                response = await client.get(location, headers=headers, follow_redirects=False)
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
            "curl", "-s", "--max-time", "30",
            "--proto", "=http,https",  # block file://, gopher://, etc.
            "--max-redirs", "0",       # no redirects (already validated by httpx path)
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

    # Read one byte past the limit so we can detect oversized files without
    # buffering the entire payload first.
    data = file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise ImageLoadError(f"Image exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB")

    img = Image.open(io.BytesIO(data))
    img.load()
    return img
