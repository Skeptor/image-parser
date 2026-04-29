import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from app.models import ConcertSlot, StageSchedule

logger = logging.getLogger(__name__)

TIME_PATTERN = re.compile(
    r"(\d{1,2}:\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.|A\.M\.|P\.M\.|H)?",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b(\d{1,2})[/\-\.](\d{1,2})(?:[/\-\.](\d{2,4}))?\b"
)

STAGE_KEYWORDS = {
    "stage",
    "escenario",
    "main",
    "arena",
    "tent",
    "lawn",
    "garden",
    "pavilion",
    "hall",
    "theater",
    "theatre",
    "club",
    "room",
    "zone",
    "area",
    "platform",
    "podium",
}


KNOWN_STAGES = {
    "negrita": "Escenario Negrita",
    "cutty sark": "Escenario Cutty Sark",
    "cuttysark": "Escenario Cutty Sark",
    "cutty": "Escenario Cutty Sark",
    "villarrobledo": "Escenario Villarrobledo Argimiro Martinez",
    "argimiro": "Escenario Villarrobledo Argimiro Martinez",
    "castilla": "Escenario Castilla-La Mancha",
    "mancha": "Escenario Castilla-La Mancha",
    "babilonia": "Escenario Babilonia",
    "babel": "Escenario Babilonia",
    "vina dub": "Viña Dub",
    "dub": "Viña Dub",
}


@dataclass
class _ParsedLine:
    text: str
    is_stage_header: bool = False
    is_time: bool = False
    times: list[str] = field(default_factory=list)
    group_name: str | None = None


def normalize_time(time_str: str) -> str:
    """Convert any time format to 24-hour HH:MM. Handles '16:00H', '6:00 PM', etc."""
    match = TIME_PATTERN.search(time_str)
    if not match:
        return time_str

    time_part = match.group(1)
    suffix = (match.group(2) or "").strip().lower().replace(".", "")

    hours, minutes = time_part.split(":")
    hours = int(hours)

    if suffix in ("am", "a"):
        hours = hours % 12
    elif suffix in ("pm", "p"):
        hours = hours % 12 + 12

    return f"{hours:02d}:{minutes}"


def _looks_like_stage_header(line: str) -> bool:
    """Heuristic: a stage header is short, capitalized, and may contain stage keywords."""
    stripped = line.strip()
    if not stripped:
        return False
    words = stripped.split()

    if len(words) > 6:
        return False

    has_keyword = any(kw in stripped.lower() for kw in STAGE_KEYWORDS)

    all_caps = stripped.isupper() and len(stripped) > 2

    return has_keyword or all_caps


def _resolve_stage_name(raw_name: str) -> str:
    """Map a garbled OCR stage name to a known stage using fuzzy matching."""
    normalized = raw_name.lower().strip()

    for key, canonical in KNOWN_STAGES.items():
        if key in normalized or normalized in key:
            return canonical

    upper_words = sum(1 for w in raw_name.split() if w[0].isupper() if w)
    if upper_words >= 1 and len(raw_name) < 30:
        return raw_name.strip()

    return raw_name


def _extract_times(line: str) -> list[str]:
    """Extract all time values from a line."""
    matches = TIME_PATTERN.finditer(line)
    return [normalize_time(m.group(0)) for m in matches]


def _extract_date(text: str) -> str | None:
    """
    Extract date from OCR text and normalize to DD/MM/YYYY format.

    Looks for patterns like:
    - DD/MM/YYYY or DD-MM-YYYY
    - DD/MM or DD-MM (assumes current year)

    Args:
        text: Raw OCR text

    Returns:
        Date string in DD/MM/YYYY format, or None if no valid date found
    """
    # Look for dates in the text
    matches = DATE_PATTERN.findall(text)

    for match in matches:
        day_str, month_str, year_str = match[0], match[1], match[2] if len(match) > 2 else ""

        try:
            day = int(day_str)
            month = int(month_str)

            # Validate day and month ranges
            if not (1 <= day <= 31 and 1 <= month <= 12):
                continue

            # Determine year
            if year_str:
                year = int(year_str)
                # Handle 2-digit years
                if year < 100:
                    year = 2000 + year if year < 50 else 1900 + year
            else:
                # No year provided, use current year
                year = datetime.now().year

            # Validate the date
            try:
                datetime(year, month, day)
                return f"{day:02d}/{month:02d}/{year}"
            except ValueError:
                # Invalid date (e.g., 31/02)
                continue

        except ValueError:
            continue

    return None


def _is_valid_time(time_str: str) -> bool:
    """Check if time string is valid (0-23 hours, 0-59 minutes)."""
    try:
        h, m = time_str.split(":")
        return 0 <= int(h) <= 23 and 0 <= int(m) <= 59
    except:
        return False


def _infer_end_time(start_time: str) -> str:
    """Infer end time as start hour + 1 when end time is invalid."""
    try:
        h, m = start_time.split(":")
        next_hour = (int(h) + 1) % 24
        return f"{next_hour:02d}:{m}"
    except:
        return None


def _split_by_time_pairs(times: list[str]) -> list[tuple[str, str | None]]:
    """
    Split a list of times into (start, end) pairs for multi-slot cells.

    When a cell contains 3+ times, this splits them into consecutive pairs.
    Example: ["20:15", "22:15", "23:30"] → [("20:15", "22:15"), ("23:30", None)]

    Args:
        times: List of valid time strings

    Returns:
        List of (start_time, end_time) tuples
    """
    if not times:
        return []

    if len(times) == 1:
        return [(times[0], None)]

    # Pair consecutive times: 0-1, 2-3, 4-5, etc.
    pairs = []
    for i in range(0, len(times), 2):
        start = times[i]
        end = times[i + 1] if i + 1 < len(times) else None
        pairs.append((start, end))

    return pairs


def _detect_cells_in_column(words: list[dict], min_cell_gap: int = 50) -> list[list[dict]]:
    """Detect cells (artist slots) by clustering words with vertical gaps.

    Args:
        words: List of word dicts with 'top', 'height', 'text' keys
        min_cell_gap: Minimum Y-gap (pixels) to indicate new cell

    Returns:
        List of cells, where each cell is a list of word dicts
    """
    if not words:
        return []

    # Sort by Y position
    sorted_words = sorted(words, key=lambda w: w.get("top", 0))

    cells = []
    current_cell = [sorted_words[0]]

    for word in sorted_words[1:]:
        prev_word = current_cell[-1]
        # Calculate gap between bottom of previous word and top of current
        gap = word.get("top", 0) - (prev_word.get("top", 0) + prev_word.get("height", 0))

        if gap > min_cell_gap:
            # Large gap = new cell
            cells.append(current_cell)
            current_cell = [word]
        else:
            # Same cell
            current_cell.append(word)

    if current_cell:
        cells.append(current_cell)

    # Enhancement: Merge cells if continuation detected
    # DISABLED FOR NOW - was too aggressive
    merged_cells = cells
    # i = 0
    # while i < len(cells):
    #     current = cells[i]

    #     # Check if next cell is a continuation
    #     if i + 1 < len(cells):
    #         current_text = " ".join(w.get("text", "") for w in current)
    #         next_text = " ".join(w.get("text", "") for w in cells[i + 1])

    #         # Continuation markers: starts with lowercase, "and", "of", "&"
    #         if next_text and (
    #             (next_text[0].islower() if next_text else False) or
    #             next_text.lower().startswith(("and ", "of ", "& ", "y "))
    #         ):
    #             # Merge with next cell
    #             merged_cells.append(current + cells[i + 1])
    #             i += 2
    #             continue

    #     merged_cells.append(current)
    #     i += 1

    return merged_cells


def parse_from_table_rows(rows: list[list[str]]) -> StageSchedule:
    """Parse a list of table rows (from camelot) into a StageSchedule."""
    schedule = StageSchedule()
    current_stage = "Main Stage"

    for row in rows:
        row_text = " ".join(row)

        if _looks_like_stage_header(row_text):
            current_stage = _resolve_stage_name(row_text.strip())
            continue

        times = []
        for cell in row:
            cell_times = _extract_times(cell)
            times.extend(cell_times)

        non_time_cells = [
            c.strip() for c in row
            if not TIME_PATTERN.search(c) and c.strip()
        ]

        if times and non_time_cells:
            group = " / ".join(non_time_cells)
            start = times[0]
            end = times[1] if len(times) > 1 else None

            if current_stage not in schedule.stages:
                schedule.stages[current_stage] = []

            schedule.stages[current_stage].append(
                ConcertSlot(group=group, start_time=start, end_time=end)
            )

    return schedule


def _clean_group(text: str) -> str:
    """Extract the actual group name from a line, removing time ranges, separators, and noise."""
    # Remove times
    cleaned = re.sub(TIME_PATTERN, " ", text).strip()

    # Remove common separators (but keep periods for abbreviations)
    cleaned = re.sub(r"[\|·•\>\→\[\]\(\)]+", " ", cleaned).strip()

    # Remove standalone dashes (but keep them in compound words)
    # Match: space + dash + (space or end of string) OR (start or space) + dash + space
    cleaned = re.sub(r"\s+[-\–\—]+(?:\s+|$)", " ", cleaned).strip()
    cleaned = re.sub(r"^[-\–\—]+\s+", "", cleaned).strip()

    # Remove specific noise patterns (==, etc)
    cleaned = re.sub(r"==+", " ", cleaned).strip()

    # Remove duplicate consecutive words (OCR creates these)
    words = cleaned.split()
    deduped = []
    prev_word = None
    for word in words:
        word_clean = word.strip(".,")
        if word_clean.lower() != (prev_word or "").lower() and word_clean:
            deduped.append(word)
            prev_word = word_clean
    cleaned = " ".join(deduped)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Filter out specific noise words (but not valid single letters like "T.")
    noise_words = {"ji", "qq", "pf", "ey", "ha", "pl", "tira", "ul"}
    words = cleaned.split()
    filtered = []
    for word in words:
        word_lower = word.lower().strip(".,")
        if word_lower not in noise_words and len(word_lower) > 0:
            filtered.append(word)
    cleaned = " ".join(filtered)

    # Filter out lines that are just noise
    if not cleaned or len(cleaned) < 2:
        return ""

    # Strip trailing separators (dashes, underscores, etc.)
    cleaned = cleaned.rstrip('_-=•·|>→ ')

    return cleaned.strip()


def parse_column_text(text: str, stage_hint: str | None = None, column_words: list[dict] | None = None) -> StageSchedule:
    """Parse the OCR text from a single column (one stage).

    Strategy:
    - If column_words available: Use spatial cell detection (primary approach)
    - Otherwise: Fall back to line-based text parsing
    - Each cell represents one artist slot
    - Handle multi-line artist names within cells
    - Filter out noise and fragments

    Args:
        text: Raw OCR text (for fallback)
        stage_hint: Stage name from header detection
        column_words: List of word dicts with position info (top, height, left, text)
    """
    schedule = StageSchedule()
    stage_name = stage_hint or "Unknown Stage"

    # NEW: Spatial parsing path using word position data
    if column_words:
        logger.info(f"Column {stage_hint}: {len(column_words)} words")
        cells = _detect_cells_in_column(column_words, min_cell_gap=35)  # 35px gap works for most festival schedules
        logger.info(f"Column {stage_hint}: detected {len(cells)} cells")

        for i, cell in enumerate(cells):
            # Sort words left-to-right, top-to-bottom within cell
            cell_sorted = sorted(cell, key=lambda w: (w.get("top", 0), w.get("left", 0)))
            cell_text = " ".join(w.get("text", "") for w in cell_sorted)

            logger.info(f"  Cell {i}: {len(cell)} words, text='{cell_text[:60]}'")

            # Skip stage header if found
            if stage_name == "Unknown Stage" and _looks_like_stage_header(cell_text):
                stage_name = _resolve_stage_name(cell_text)
                logger.info(f"    -> Detected stage header: {stage_name}")
                continue

            # Extract times from cell
            times = _extract_times(cell_text)
            if times:
                logger.info(f"    -> Times: {times}, creating slot")

            if times:
                # Clean artist name (remove times and noise)
                artist_name = _clean_group(cell_text)

                if artist_name and len(artist_name) > 1:
                    # Deduplicate times
                    unique_times = []
                    seen_times = set()
                    for t in times:
                        if t not in seen_times:
                            unique_times.append(t)
                            seen_times.add(t)

                    # Validate times
                    valid_times = [t for t in unique_times if _is_valid_time(t)]

                    if valid_times:
                        # Split by time pairs if multiple times detected
                        time_pairs = _split_by_time_pairs(valid_times)

                        if len(time_pairs) > 1:
                            logger.info(f"Split cell into {len(time_pairs)} slots from {len(valid_times)} times")

                        for start_time, end_time in time_pairs:
                            if not end_time:
                                end_time = _infer_end_time(start_time)

                            slot = ConcertSlot(
                                group=artist_name,
                                start_time=start_time,
                                end_time=end_time
                            )
                            schedule.stages.setdefault(stage_name, []).append(slot)

        # Ensure stage exists even if empty
        if stage_name and stage_name not in schedule.stages:
            schedule.stages[stage_name] = []

        return schedule

    # FALLBACK: Line-based text parsing (original logic)
    lines = text.splitlines()
    accumulated_name = []
    last_slot_end = None  # Track last slot's end time to use as next slot's start if missing

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            continue

        times = _extract_times(stripped)

        # Skip stage header if found
        if stage_name == "Unknown Stage" and _looks_like_stage_header(stripped):
            stage_name = _resolve_stage_name(stripped)
            continue

        if times:
            # Found a time line - create a slot
            # Combine accumulated name with any text on this line (minus times)
            name_on_this_line = _clean_group(stripped)
            if name_on_this_line:
                accumulated_name.append(name_on_this_line)

            if accumulated_name:
                group_name = " ".join(accumulated_name).strip()

                # Deduplicate times (OCR often creates duplicates like "20:30 20:30")
                unique_times = []
                seen_times = set()
                for t in times:
                    if t not in seen_times:
                        unique_times.append(t)
                        seen_times.add(t)

                # Sometimes times are in wrong order or have weird values (like 90:53)
                # Filter out invalid times
                valid_times = [t for t in unique_times if _is_valid_time(t)]

                if group_name and len(valid_times) > 0:
                    start_time = valid_times[0]
                    # Use second valid time if available, otherwise infer from start time
                    end_time = valid_times[1] if len(valid_times) > 1 else _infer_end_time(start_time)

                    slot = ConcertSlot(
                        group=group_name,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    schedule.stages.setdefault(stage_name, []).append(slot)
                    last_slot_end = end_time

                accumulated_name = []
        else:
            # No times found - this might be part of an artist name
            cleaned = _clean_group(stripped)
            # Only check for stage header if we don't have a stage name yet
            is_header = (stage_name == "Unknown Stage" and _looks_like_stage_header(stripped))

            if cleaned and not is_header:
                # Skip very short fragments unless they look like valid artist name parts
                if len(cleaned) > 3 or (len(cleaned) >= 2 and any(c.isalpha() for c in cleaned)):
                    accumulated_name.append(cleaned)

    if stage_name and stage_name not in schedule.stages:
        schedule.stages[stage_name] = []

    return schedule


def parse_from_text(text: str) -> StageSchedule:
    """Parse raw OCR text using line-based heuristics (fallback for single-column)."""
    schedule = StageSchedule()
    lines = text.splitlines()

    current_stage = None
    last_time_start = None
    pending_group = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        times_in_line = _extract_times(stripped)

        if _looks_like_stage_header(stripped):
            if pending_group and last_time_start and current_stage:
                schedule.stages.setdefault(current_stage, []).append(
                    ConcertSlot(group=pending_group, start_time=last_time_start, end_time=None)
                )
            current_stage = _resolve_stage_name(stripped)
            pending_group = None
            last_time_start = None
            continue

        if times_in_line:
            if pending_group and last_time_start and current_stage:
                schedule.stages.setdefault(current_stage, []).append(
                    ConcertSlot(
                        group=pending_group,
                        start_time=last_time_start,
                        end_time=times_in_line[0],
                    )
                )
                pending_group = None

            last_time_start = times_in_line[0]
            remaining = _clean_group(stripped)
            if remaining:
                pending_group = remaining

        elif current_stage and not _looks_like_stage_header(stripped):
            cleaned = _clean_group(stripped)
            if cleaned:
                if not pending_group:
                    pending_group = cleaned
                else:
                    pending_group += f" {cleaned}"

    if pending_group and last_time_start and current_stage:
        schedule.stages.setdefault(current_stage, []).append(
            ConcertSlot(group=pending_group, start_time=last_time_start, end_time=None)
        )

    return schedule


def merge_schedules(schedules: list[StageSchedule]) -> StageSchedule:
    """Merge multiple StageSchedule objects into one."""
    merged = StageSchedule()
    for s in schedules:
        if s.date and not merged.date:
            merged.date = s.date
        for stage_name, slots in s.stages.items():
            merged.stages.setdefault(stage_name, []).extend(slots)
    return merged


def extract_schedule(
    ocr_text: str,
    table_data: list[dict] | None = None,
    column_data: list[dict] | None = None,
) -> StageSchedule:
    """Main entry point.

    Priority:
    1. Column-based data with headers (multi-column images)
    2. Table-based data (camelot)
    3. Raw text heuristic fallback

    Args:
        ocr_text: Raw OCR text from the full image
        table_data: Table data from camelot (rows + accuracy)
        column_data: List of dicts with 'text', 'header', and optionally 'words' keys from column detection
    """
    # Extract date from OCR text (check first 500 chars where headers usually are)
    date = _extract_date(ocr_text[:500])

    if column_data:
        logger.info("Using column-based parsing (%d columns)", len(column_data))
        partial = []
        for col in column_data:
            text = col.get("text", "")
            header = col.get("header")
            words = col.get("words", [])  # NEW: Get word position data
            parsed = parse_column_text(text, stage_hint=header, column_words=words)  # NEW: Pass words
            partial.append(parsed)
        schedule = merge_schedules(partial)
        schedule.date = date
        return schedule

    if table_data and any(t.get("rows") for t in table_data):
        best_table = max(table_data, key=lambda t: t.get("accuracy", 0))
        logger.info("Using table-based parsing, accuracy=%.1f", best_table.get("accuracy", 0))
        schedule = parse_from_table_rows(best_table["rows"])
        schedule.date = date
        return schedule

    logger.info("Using text-based heuristic parsing")
    schedule = parse_from_text(ocr_text)
    schedule.date = date
    return schedule
