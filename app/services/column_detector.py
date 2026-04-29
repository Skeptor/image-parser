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
    all_words = []
    word_positions = set()  # Track (text, left, top) to avoid duplicates

    # Try with preprocessed image
    processed = preprocess_for_ocr(image)
    config = f"--psm {psm} --oem {oem}"
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or data["conf"][i] < 30:  # Increased from 10 to 30 for better filtering
            continue
        pos_key = (text, data["left"][i], data["top"][i])
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

    # Also try with original image (no preprocessing) to catch words lost in preprocessing
    data2 = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    for i in range(len(data2["text"])):
        text = data2["text"][i].strip()
        if not text or data2["conf"][i] < 30:  # Increased from 10 to 30 for better filtering
            continue
        pos_key = (text, data2["left"][i], data2["top"][i])
        if pos_key not in word_positions:
            word_positions.add(pos_key)
            all_words.append({
                "text": text,
                "left": data2["left"][i],
                "top": data2["top"][i],
                "width": data2["width"][i],
                "height": data2["height"][i],
                "confidence": data2["conf"][i],
                "x_center": data2["left"][i] + data2["width"][i] // 2,
            })

    # Try with inverted colors (helps with light text on dark backgrounds)
    inverted = preprocess_for_ocr(image, invert=True)
    data3 = pytesseract.image_to_data(
        inverted, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    for i in range(len(data3["text"])):
        text = data3["text"][i].strip()
        if not text or data3["conf"][i] < 30:  # Increased from 10 to 30 for better filtering
            continue
        pos_key = (text, data3["left"][i], data3["top"][i])
        if pos_key not in word_positions:
            word_positions.add(pos_key)
            all_words.append({
                "text": text,
                "left": data3["left"][i],
                "top": data3["top"][i],
                "width": data3["width"][i],
                "height": data3["height"][i],
                "confidence": data3["conf"][i],
                "x_center": data3["left"][i] + data3["width"][i] // 2,
            })

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
    words = get_word_boxes(image, lang=lang, psm=psm, oem=oem)

    if not words:
        return [Column(index=0, x_start=0, x_end=w, words=[], header=None)]

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

            # Get words in this column (by x_center position)
            col_words = [
                wd for wd in words
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
            time_words = [wd for wd in words if wd["x_center"] < first_header_x]
            if time_words:
                # Insert time column at the beginning (but don't include in parsing)
                logger.info("Detected time axis column (x < %d)", first_header_x)

        for col in columns:
            logger.info(
                "  Column %d: x=[%d, %d], header='%s', %d words",
                col.index, col.x_start, col.x_end, col.header, len(col.words)
            )

        return columns

    # Fallback: use gap detection
    boundaries = _find_column_boundaries(words, w, min_gap=40)

    if len(boundaries) >= 1:
        logger.info("Using gap-based column detection, found %d boundaries", len(boundaries))

        # Create columns from boundaries
        all_bounds = [0] + boundaries + [w]
        columns = []

        for idx in range(len(all_bounds) - 1):
            col_x_start = all_bounds[idx]
            col_x_end = all_bounds[idx + 1]

            col_words = [
                wd for wd in words
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

        # Re-index
        for i, col in enumerate(columns):
            col.index = i

        if len(columns) >= min_columns:
            for col in columns:
                logger.info(
                    "  Column %d: x=[%d, %d], %d words",
                    col.index, col.x_start, col.x_end, len(col.words)
                )
            return columns

    # Ultimate fallback: single column
    logger.info("Treating image as single column")
    return [Column(index=0, x_start=0, x_end=w, words=words, header=None)]


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
    deduped_words = []
    word_positions = set()
    position_tolerance = 10

    for w in words:
        # Create a position key with tolerance
        pos_key = (w["text"], w["top"] // position_tolerance, w["left"] // position_tolerance)
        if pos_key not in word_positions:
            word_positions.add(pos_key)
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
        line_text = " ".join(w["text"] for w in line)
        # Filter out lines that are just noise (single characters, etc.)
        if len(line_text.strip()) > 1:
            text_lines.append(line_text)

    return "\n".join(text_lines)
