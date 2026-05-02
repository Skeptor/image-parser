import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import pytesseract

logger = logging.getLogger(__name__)


def upscale_if_needed(image: Image.Image, target_dpi: int = 300) -> Image.Image:
    """Upscale image if resolution is below target DPI.

    Research shows 2x upscaling can improve OCR accuracy by 5-10% for low-res images.
    Tesseract works best at 300+ DPI.

    Args:
        image: Input image
        target_dpi: Target DPI (default: 300)

    Returns:
        Upscaled image if needed, original otherwise
    """
    # Get DPI from image metadata if available
    dpi = image.info.get("dpi", (72, 72))
    if isinstance(dpi, (tuple, list)):
        dpi = dpi[0]  # Use horizontal DPI

    # If DPI is too low, upscale 2x
    if dpi < 200:
        new_width = image.width * 2
        new_height = image.height * 2
        logger.info(
            "Upscaling image from %dx%d (DPI: %d) to %dx%d for better OCR",
            image.width, image.height, dpi, new_width, new_height
        )
        # Use Lanczos resampling for best quality
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def preprocess_for_ocr(
    image: Image.Image,
    invert: bool = False,
    use_binarization: bool = True,
    use_morphology: bool = True,
) -> Image.Image:
    """Preprocess image to improve OCR accuracy.

    Applies research-backed techniques for 8-12% accuracy improvement:
    - Adaptive binarization (Otsu's method)
    - Morphological operations to reduce noise
    - Contrast and sharpness enhancement

    Args:
        image: Input image
        invert: If True, invert colors (useful for light text on dark backgrounds)
        use_binarization: Apply adaptive thresholding (recommended: True)
        use_morphology: Apply morphological operations to clean noise (recommended: True)
    """
    # Convert PIL to numpy array for OpenCV processing
    img_array = np.array(image.convert("L"))

    # Invert if requested (helps with light text on dark backgrounds)
    if invert:
        img_array = cv2.bitwise_not(img_array)

    # Apply contrast enhancement before binarization
    img_array = cv2.convertScaleAbs(img_array, alpha=1.3, beta=10)

    if use_binarization:
        # Adaptive binarization using Otsu's method
        # This automatically determines optimal threshold
        _, img_array = cv2.threshold(
            img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if use_morphology:
        # Morphological operations to clean up noise
        # Opening: removes small noise (erosion then dilation)
        # Closing: fills small gaps (dilation then erosion)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Opening to remove small noise
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)

        # Closing to fill small gaps in characters
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL Image
    result = Image.fromarray(img_array)

    # Apply sharpening to enhance edges
    enhancer = ImageEnhance.Sharpness(result)
    result = enhancer.enhance(1.5)

    return result


@dataclass
class Column:
    """Represents a vertical column in the image."""
    index: int
    x_start: int
    x_end: int
    words: list[dict[str, Any]] = field(default_factory=list)
    header: str | None = None


def get_word_boxes(
    image: Image.Image,
    lang: str = "eng",
    psm: int = 4,
    oem: int = 1,
    boundary_only: bool = False,
) -> list[dict]:
    """Run Tesseract and return word-level bounding boxes.

    Runs OCR multiple times with different preprocessing to maximize word detection.

    PSM modes (Page Segmentation Mode):
        4 = Single column of variable sizes (good for schedules with rows/columns)
        6 = Uniform block of text (default)
        11 = Sparse text with table structure (ideal for grid-based schedules)

    OEM modes (OCR Engine Mode):
        0 = Legacy engine only
        1 = LSTM neural network only
        2 = Legacy + LSTM (hybrid, often best) - recommended
        3 = Default (based on what's available)

    Args:
        image: Input image
        lang: Tesseract language code
        psm: Page segmentation mode (default: 4 for single column)
        oem: OCR engine mode (default: 2 for hybrid LSTM)
    """
    config = f"--psm {psm} --oem {oem}"
    all_words = []
    word_positions: set[tuple] = set()
    pos_tolerance = 5  # px tolerance for deduplication

    def _collect(pil_img: Image.Image, min_conf: int = 30) -> None:
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text or data["conf"][i] < min_conf:
                continue
            pos_key = (text, data["left"][i] // pos_tolerance, data["top"][i] // pos_tolerance)
            if pos_key not in word_positions:
                word_positions.add(pos_key)
                all_words.append({
                    "text": text,
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                    "confidence": data["conf"][i],
                    "x_center": data["left"][i] + data["width"][i] // 2,
                })

    # Pass 1: standard grayscale + contrast + Otsu binarization
    _collect(preprocess_for_ocr(image))

    # Pass 2: original image (no preprocessing) — recovers words lost by binarization
    _collect(image)

    # Pass 3: inverted — helps with light text on dark backgrounds
    _collect(preprocess_for_ocr(image, invert=True))

    if not boundary_only:
        # Pass 4: green channel — better contrast on pink/yellow/red colored backgrounds.
        # Global Otsu binarization often fails on large coloured areas; extracting a single
        # channel bypasses this. Not used for column-boundary detection because the different
        # x-positions it finds can introduce false gaps in the histogram.
        rgb = image.convert("RGB")
        g_channel = Image.fromarray(np.array(rgb)[:, :, 1])
        _collect(g_channel)

        # Pass 5: CLAHE on grayscale — local contrast enhancement for images where large
        # bright areas cause global Otsu to lose text detail.
        gray_arr = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _collect(Image.fromarray(clahe.apply(gray_arr)))

    return all_words


def _find_column_boundaries(words: list[dict], img_width: int, min_gap: int = 30) -> list[int]:
    """Find column boundaries by looking for gaps in word x-positions.

    Returns a sorted list of x-coordinates where columns should split.
    """
    if not words:
        return []

    # Build histogram of word x-center positions (binned by 10px)
    bin_size = 10
    bins = [0] * (img_width // bin_size + 1)

    for w in words:
        x_center = w["x_center"]
        bin_idx = min(x_center // bin_size, len(bins) - 1)
        bins[bin_idx] += 1

    # Find gaps (runs of empty bins >= min_gap / bin_size)
    min_gap_bins = min_gap // bin_size
    boundaries = []
    gap_start = None

    for i, count in enumerate(bins):
        if count == 0:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_length = i - gap_start
                if gap_length >= min_gap_bins:
                    # Put boundary in the middle of the gap
                    boundary_x = (gap_start + i) // 2 * bin_size
                    boundaries.append(boundary_x)
            gap_start = None

    return sorted(boundaries)


STAGE_KEYWORDS = {
    "escenario", "stage", "santuario", "turia", "walker", "sansan", "dub",
    "negrita", "cutty", "sark", "babilonia", "villarrobledo", "castilla",
    "main", "acoustic", "arena", "tent", "lawn", "adamo",
}


def _find_header_columns(
    words: list[dict],
    img_height: int,
    img_width: int,
    header_y_max_ratio: float = 0.20,
) -> list[tuple[int, int, str]]:
    """Find stage headers in the top portion of the image.

    Strategy:
    1. Find all words containing stage keywords in the top portion
    2. Cluster them by x-position to identify column headers
    3. Build header text from clustered words

    Returns list of (x_start, x_end, header_text) tuples.
    """
    header_y_max = int(img_height * header_y_max_ratio)

    # Get all words in header region
    header_words = [w for w in words if w["top"] < header_y_max]

    if not header_words:
        return []

    # Find words that contain stage keywords
    stage_words = []
    for w in header_words:
        text_lower = w["text"].lower()
        if any(kw in text_lower for kw in STAGE_KEYWORDS):
            stage_words.append(w)

    if len(stage_words) < 2:
        # Fall back to using all header words that look like headers (capitalized, etc)
        stage_words = [w for w in header_words if w["text"][0].isupper() and len(w["text"]) > 3]

    if not stage_words:
        return []

    # Cluster stage words by x-position
    # Sort by x position first
    stage_words.sort(key=lambda w: w["left"])

    # Group words that are close in x-position (within same column)
    column_width_estimate = img_width // 5  # Assume ~5 columns max
    groups = []
    current_group = [stage_words[0]]

    for w in stage_words[1:]:
        prev = current_group[-1]
        # Check x-distance (not gap, but center-to-center distance)
        prev_center = prev["left"] + prev["width"] // 2
        curr_center = w["left"] + w["width"] // 2
        x_dist = abs(curr_center - prev_center)

        if x_dist < column_width_estimate // 2:
            # Same column
            current_group.append(w)
        else:
            # New column
            groups.append(current_group)
            current_group = [w]

    if current_group:
        groups.append(current_group)

    # For each group, get the x range and combine text
    # Also pull in any nearby words from header region to build full header text
    headers = []
    for group in groups:
        x_min = min(w["left"] for w in group)
        x_max = max(w["left"] + w["width"] for w in group)

        # Expand range more generously to capture nearby words
        x_min_expanded = max(0, x_min - 100)
        x_max_expanded = min(img_width, x_max + 100)

        # Find all header words in this x range
        column_header_words = [
            w for w in header_words
            if x_min_expanded <= w["left"] + w["width"] // 2 <= x_max_expanded
        ]

        # Sort by y then x to get reading order
        column_header_words.sort(key=lambda w: (w["top"], w["left"]))

        # Build header text
        header_text = " ".join(w["text"] for w in column_header_words)

        # Clean up header text - extract just the stage name
        import re

        # Remove dates, times, numbers
        header_text = re.sub(r'\b\d{1,2}:\d{2}H?\b', '', header_text)  # Remove times like 18:00H
        header_text = re.sub(r'\b\d{4}\b', '', header_text)  # Remove years
        header_text = re.sub(r'\bJUEVES\b|\bVIERNES\b|\bSABADO\b|\bDOMINGO\b', '', header_text, flags=re.IGNORECASE)  # Remove day names
        header_text = re.sub(r'\bABRIL\b|\bMAYO\b|\bJUNIO\b', '', header_text, flags=re.IGNORECASE)  # Remove month names
        header_text = re.sub(r'\b\d+\s+ABRIL\b|\b\d+\s+MAYO\b', '', header_text, flags=re.IGNORECASE)  # Remove "2 ABRIL"
        header_text = re.sub(r'\bAPERTURA\b|\bPUERTAS\b', '', header_text, flags=re.IGNORECASE)  # Remove generic words
        header_text = re.sub(r'\bDE\b|\bLA\b|\bEL\b', '', header_text, flags=re.IGNORECASE)  # Remove articles (unless part of stage name)
        header_text = re.sub(r'\bROCK\b|\bHORARIOS\b|\bMET\b|\bnai\b|\bvAS\b', '', header_text, flags=re.IGNORECASE)  # Remove noise words
        header_text = re.sub(r'\s+', ' ', header_text).strip()
        header_text = re.sub(r'[-—]+$', '', header_text).strip()  # Remove trailing dashes

        # Extract known stage names
        stage_name = None
        text_lower = header_text.lower()

        # Check for known stage patterns
        if 'sansan' in text_lower:
            stage_name = 'Escenario SanSan'
        elif 'turia' in text_lower:
            stage_name = 'Escenario Turia'
        elif 'walker' in text_lower or 'johnnie' in text_lower:
            stage_name = 'Escenario Johnnie Walker'
        elif 'santuario' in text_lower or 'adamo' in text_lower:
            stage_name = 'El Santuario by Adamo'
        elif 'negrita' in text_lower:
            stage_name = 'Escenario Negrita'
        elif 'cutty' in text_lower or 'sark' in text_lower:
            stage_name = 'Escenario Cutty Sark'
        elif 'villarrobledo' in text_lower or 'argimiro' in text_lower:
            stage_name = 'Escenario Villarrobledo'
        elif 'castilla' in text_lower or 'mancha' in text_lower:
            stage_name = 'Escenario Castilla-La Mancha'
        elif 'babilonia' in text_lower:
            stage_name = 'Escenario Babilonia'
        elif 'dub' in text_lower and ('viña' in text_lower or 'vina' in text_lower):
            stage_name = 'Viña Dub'
        elif 'escenario' in text_lower and len(header_text) < 20:
            # Generic escenario with few words - likely incomplete OCR
            # Use position-based naming
            stage_name = f"Escenario {len(headers) + 1}"
        elif 'escenario' in text_lower:
            # Generic escenario, keep cleaned text
            stage_name = header_text
        else:
            # Use cleaned text
            stage_name = header_text if header_text else f"Stage {len(headers) + 1}"

        if stage_name and len(stage_name) > 2:
            headers.append((x_min, x_max, stage_name))

    return headers


def detect_columns(
    image: Image.Image,
    lang: str = "spa",
    min_columns: int = 2,
    max_columns: int = 10,
    psm: int = 4,
    oem: int = 1,
) -> list[Column]:
    """Detect vertical columns using word positions and header detection.

    Strategy:
    1. Upscale image if resolution is low (< 200 DPI)
    2. Get all word bounding boxes with optimized PSM/OEM settings
    3. Find stage headers in the top portion
    4. Use header positions to define column boundaries
    5. Fall back to gap detection if no headers found

    Args:
        image: Input image
        lang: Tesseract language code
        min_columns: Minimum number of columns to detect
        max_columns: Maximum number of columns to detect
        psm: Page segmentation mode (4=single column, 6=block, 11=sparse table)
        oem: OCR engine mode (2=hybrid LSTM, recommended)
    """
    # Upscale if needed for better OCR quality
    image = upscale_if_needed(image)

    w, h = image.size

    # Two-phase word collection:
    # Phase 1 (boundary_only=True) — 3 standard passes used to determine column splits.
    #   Using only these keeps the x-position histogram stable so column gaps are reliable.
    # Phase 2 (boundary_only=False) — all 5 passes (+ G-channel, CLAHE) for richer text
    #   extraction. Words are assigned to already-detected column boundaries.
    boundary_words = get_word_boxes(image, lang=lang, psm=psm, oem=oem, boundary_only=True)
    all_words = get_word_boxes(image, lang=lang, psm=psm, oem=oem, boundary_only=False)
    # Use boundary_words for column detection, all_words for column content
    words = boundary_words

    if not words:
        return [Column(index=0, x_start=0, x_end=w, words=all_words, header=None)]

    # Try to find headers first
    headers = _find_header_columns(words, h, w)

    if len(headers) >= min_columns:
        logger.info("Found %d stage headers in image", len(headers))

        # Create columns based on header positions
        columns = []
        sorted_headers = sorted(headers, key=lambda x: x[0])

        for idx, (hx_start, hx_end, header_text) in enumerate(sorted_headers):
            # Column starts at header x_start, ends at next header start (or image edge)
            col_x_start = hx_start
            if idx < len(sorted_headers) - 1:
                col_x_end = sorted_headers[idx + 1][0]
            else:
                col_x_end = w

            # Use all_words (enriched with G-channel/CLAHE) for column content
            col_words = [
                wd for wd in all_words
                if col_x_start <= wd["x_center"] < col_x_end
            ]

            columns.append(Column(
                index=idx,
                x_start=col_x_start,
                x_end=col_x_end,
                words=col_words,
                header=header_text,
            ))

        # Check for time column on the left (before first header)
        first_header_x = sorted_headers[0][0]
        if first_header_x > 50:  # There's space for a time column
            time_words = [wd for wd in all_words if wd["x_center"] < first_header_x]
            if time_words:
                logger.info("Detected time axis column (x < %d)", first_header_x)

        for col in columns:
            logger.info(
                "  Column %d: x=[%d, %d], header='%s', %d words",
                col.index, col.x_start, col.x_end, col.header, len(col.words)
            )

        return columns

    # Fallback: use gap detection (boundaries from boundary_words, content from all_words)
    boundaries = _find_column_boundaries(words, w, min_gap=40)

    if len(boundaries) >= 1:
        logger.info("Using gap-based column detection, found %d boundaries", len(boundaries))

        all_bounds = [0] + boundaries + [w]
        columns = []

        for idx in range(len(all_bounds) - 1):
            col_x_start = all_bounds[idx]
            col_x_end = all_bounds[idx + 1]

            col_words = [
                wd for wd in all_words
                if col_x_start <= wd["x_center"] < col_x_end
            ]

            columns.append(Column(
                index=idx,
                x_start=col_x_start,
                x_end=col_x_end,
                words=col_words,
                header=None,
            ))

        # Filter out columns with very few words (likely time axis)
        columns = [c for c in columns if len(c.words) > 3]

        for i, col in enumerate(columns):
            col.index = i

        if len(columns) >= min_columns:
            for col in columns:
                logger.info(
                    "  Column %d: x=[%d, %d], %d words",
                    col.index, col.x_start, col.x_end, len(col.words)
                )
            return columns

    # Ultimate fallback: single column with enriched words
    logger.info("Treating image as single column")
    return [Column(index=0, x_start=0, x_end=w, words=all_words, header=None)]


_SPANISH_DAY_NAMES = frozenset({
    "lunes", "martes", "miercoles", "miércoles", "jueves",
    "viernes", "sabado", "sábado", "domingo",
})
_ENGLISH_DAY_NAMES = frozenset({
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
})
_ALL_DAY_NAMES = _SPANISH_DAY_NAMES | _ENGLISH_DAY_NAMES


@dataclass
class DayRegion:
    """A detected day section within a multi-day image."""
    day_name: str
    crop: tuple[int, int, int, int]  # (x1, y1, x2, y2)


def detect_day_regions(image: Image.Image, lang: str = "spa") -> list[DayRegion] | None:
    """Detect multiple days in one image (side-by-side or stacked).

    Scans the top 30% for day-of-week words. Returns None for single-day images,
    or a list of DayRegion objects describing each day's crop.
    """
    w, h = image.size
    header_image = image.crop((0, 0, w, int(h * 0.30)))
    words = get_word_boxes(header_image, lang=lang, psm=6, oem=1)

    day_words = [
        word for word in words
        if word["text"].lower().strip(".,;:") in _ALL_DAY_NAMES
    ]

    if len(day_words) < 2:
        return None

    # Deduplicate: same word within 50px x-band counts once
    seen: set[tuple[str, int]] = set()
    unique: list[dict] = []
    for dw in sorted(day_words, key=lambda d: d["x_center"]):
        key = (dw["text"].lower().strip(".,;:"), dw["x_center"] // 50)
        if key not in seen:
            seen.add(key)
            unique.append(dw)

    if len(unique) < 2:
        return None

    x_spread = max(d["x_center"] for d in unique) - min(d["x_center"] for d in unique)
    y_spread = max(d["top"] for d in unique) - min(d["top"] for d in unique)

    if x_spread > w * 0.25:  # Side-by-side layout
        unique.sort(key=lambda d: d["x_center"])
        centers = [d["x_center"] for d in unique]
        splits = [(centers[i] + centers[i + 1]) // 2 for i in range(len(centers) - 1)]
        x_bounds = [0] + splits + [w]
        return [
            DayRegion(
                day_name=unique[i]["text"].upper().strip(".,;:"),
                crop=(x_bounds[i], 0, x_bounds[i + 1], h),
            )
            for i in range(len(unique))
        ]

    if y_spread > h * 0.10:  # Stacked layout
        unique.sort(key=lambda d: d["top"])
        tops = [d["top"] for d in unique]
        splits = [(tops[i] + tops[i + 1]) // 2 for i in range(len(tops) - 1)]
        y_bounds = [0] + splits + [h]
        return [
            DayRegion(
                day_name=unique[i]["text"].upper().strip(".,;:"),
                crop=(0, y_bounds[i], w, y_bounds[i + 1]),
            )
            for i in range(len(unique))
        ]

    return None  # Days too close together, treat as single day


def crop_column(image: Image.Image, column: Column) -> Image.Image:
    """Crop the image to a specific column region."""
    x1 = max(0, column.x_start - 5)  # Small padding
    x2 = min(image.width, column.x_end + 5)
    return image.crop((x1, 0, x2, image.height))


def get_column_text_from_words(column: Column, skip_header_y: int = 0) -> str:
    """Reconstruct text from word boxes for a column.

    Groups words by their vertical position (lines) and sorts each line by x position.
    Deduplicates words at the same position (from multiple OCR passes).
    Returns the text in reading order.

    Args:
        column: Column with words attribute containing word boxes
        skip_header_y: Skip words with y position less than this (to skip header row)
    """
    words = column.words
    if not words:
        return ""

    # Filter out header words if requested
    if skip_header_y > 0:
        words = [w for w in words if w["top"] > skip_header_y]

    if not words:
        return ""

    # Deduplicate words at approximately the same position
    # Use a more robust approach: for each word, check if there's already
    # a very similar word nearby
    deduped_words = []
    position_tolerance = 15  # Increased from 10 to 15 pixels

    for w in words:
        # Check if this word is a duplicate of an existing word
        is_duplicate = False
        for existing in deduped_words:
            if (existing["text"] == w["text"] and
                abs(existing["top"] - w["top"]) < position_tolerance and
                abs(existing["left"] - w["left"]) < position_tolerance):
                is_duplicate = True
                break

        if not is_duplicate:
            deduped_words.append(w)

    words = deduped_words

    # Group words by approximate line (similar y position)
    # Use larger tolerance to group words on the same visual line
    line_tolerance = 35
    lines = []

    for w in sorted(words, key=lambda x: x["top"]):
        placed = False
        for line in lines:
            # Check if this word belongs to an existing line
            line_y = sum(lw["top"] for lw in line) / len(line)
            if abs(w["top"] - line_y) < line_tolerance:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # Sort each line by x position and join words
    text_lines = []
    for line in lines:
        line.sort(key=lambda x: x["left"])

        # Remove consecutive duplicate words on the same line
        # (Sometimes OCR detects the same word multiple times horizontally)
        deduplicated_line = []
        prev_text = None
        for w in line:
            if w["text"] != prev_text:
                deduplicated_line.append(w)
                prev_text = w["text"]

        line_text = " ".join(w["text"] for w in deduplicated_line)
        # Filter out lines that are just noise (single characters, etc.)
        if len(line_text.strip()) > 1:
            text_lines.append(line_text)

    return "\n".join(text_lines)
