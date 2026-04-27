import logging
import re
from dataclasses import dataclass, field

from app.models import ConcertSlot, StageSchedule

logger = logging.getLogger(__name__)

TIME_PATTERN = re.compile(
    r"(\d{1,2}:\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.|A\.M\.|P\.M\.)?",
    re.IGNORECASE,
)

TIME_24_PATTERN = re.compile(r"(\d{1,2}:\d{2})")

STAGE_KEYWORDS = {
    "stage",
    "stage",
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


@dataclass
class _ParsedLine:
    text: str
    is_stage_header: bool = False
    is_time: bool = False
    times: list[str] = field(default_factory=list)
    group_name: str | None = None


def normalize_time(time_str: str) -> str:
    """Convert any time format to 24-hour HH:MM."""
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
    words = stripped.split()

    if len(words) > 5:
        return False

    title_case = all(
        w[0].isupper() if w and w[0].isalpha() else True for w in words
    )

    has_keyword = any(
        kw in stripped.lower() for kw in STAGE_KEYWORDS
    )

    all_caps = stripped.isupper() and len(stripped) > 2

    return (title_case and has_keyword) or all_caps


def _extract_times(line: str) -> list[str]:
    """Extract all time values from a line."""
    matches = TIME_PATTERN.findall(line)
    return [normalize_time(f"{t} {s}") for t, s in matches if t]


def parse_from_table_rows(rows: list[list[str]]) -> StageSchedule:
    """Parse a list of table rows (from camelot) into a StageSchedule.

    Expected table shape:
      - Rows under each stage header contain: [group, start_time, end_time, ...]
      - Or: [time, group] per row
    """
    schedule = StageSchedule()
    current_stage = "Main Stage"

    for row in rows:
        row_text = " ".join(row)

        if _looks_like_stage_header(row_text):
            current_stage = row_text.strip()
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


def parse_from_text(text: str) -> StageSchedule:
    """Parse raw OCR text using line-based heuristics."""
    schedule = StageSchedule()
    lines = text.splitlines()

    current_stage = None
    festival_name = None
    festival_date = None
    last_time_start = None
    pending_group = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        times_in_line = _extract_times(stripped)

        if _looks_like_stage_header(stripped):
            current_stage = stripped
            if pending_group and last_time_start and current_stage:
                slot = ConcertSlot(
                    group=pending_group,
                    start_time=last_time_start,
                    end_time=None,
                )
                schedule.stages.setdefault(current_stage, []).append(slot)
                pending_group = None
                last_time_start = None
            continue

        if times_in_line:
            if pending_group and last_time_start and current_stage:
                slot = ConcertSlot(
                    group=pending_group,
                    start_time=last_time_start,
                    end_time=times_in_line[0],
                )
                schedule.stages.setdefault(current_stage, []).append(slot)
                pending_group = None

            last_time_start = times_in_line[0]

            remaining = re.sub(TIME_PATTERN, "", stripped).strip()
            if remaining:
                cleaned = re.sub(r"[\|·•\-–—>→]", "", remaining).strip()
                if cleaned and not _looks_like_stage_header(cleaned):
                    pending_group = cleaned

        elif current_stage and not _looks_like_stage_header(stripped):
            if not pending_group:
                pending_group = stripped
            else:
                pending_group += f" {stripped}"

        if not festival_name and len(stripped.split()) <= 6:
            if stripped.isupper() or (stripped[0].isupper() if stripped else False):
                upper_words = sum(1 for w in stripped.split() if w[0].isupper())
                if upper_words >= 3 and len(stripped.split()) >= 3:
                    festival_name = stripped

    if pending_group and last_time_start and current_stage:
        slot = ConcertSlot(
            group=pending_group,
            start_time=last_time_start,
            end_time=None,
        )
        schedule.stages.setdefault(current_stage, []).append(slot)

    if festival_name:
        schedule.festival = festival_name
    if festival_date:
        schedule.date = festival_date

    return schedule


def extract_schedule(
    ocr_text: str,
    table_data: list[dict] | None = None,
) -> StageSchedule:
    """Main entry point: try table-based parsing first, fall back to text.

    Args:
        ocr_text: Raw OCR text from the image
        table_data: Optional table data from camelot (list of {rows, accuracy})
    """
    if table_data and any(t.get("rows") for t in table_data):
        best_table = max(table_data, key=lambda t: t.get("accuracy", 0))
        logger.info(
            "Using table-based parsing, accuracy=%.1f",
            best_table.get("accuracy", 0),
        )
        return parse_from_table_rows(best_table["rows"])

    logger.info("Using text-based heuristic parsing")
    return parse_from_text(ocr_text)
