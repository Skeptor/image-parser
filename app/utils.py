import logging
import re

logger = logging.getLogger(__name__)


def clean_group_name(name: str) -> str:
    """Clean up artist/group names."""
    name = name.strip()
    name = re.sub(r"[\|·•\-–—>→]", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def clean_stage_name(name: str) -> str:
    """Normalize stage names. Preserve Spanish 'Escenario' prefix."""
    name = name.strip()
    if name.lower().startswith("escenario"):
        return name
    suffixes = ["stage", "stg", "area", "zone", "platform", "podium"]
    lower = name.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    if not name:
        name = "Main Stage"
    return name


def compute_confidence(schedule) -> float:
    """Compute a confidence score based on parsed data quality.

    Factors:
    - Number of stages found
    - Number of slots per stage
    - Whether end_time was found
    - Whether date was detected
    """
    if not schedule or not schedule.stages:
        return 0.0

    total_slots = sum(len(slots) for slots in schedule.stages.values())
    if total_slots == 0:
        return 0.0

    scores: list[float] = []

    num_stages = len(schedule.stages)
    scores.append(min(num_stages / 3, 1.0) * 0.2)

    slots_with_end = sum(
        1 for slots in schedule.stages.values()
        for s in slots if s.end_time
    )
    end_ratio = slots_with_end / total_slots if total_slots > 0 else 0
    scores.append(end_ratio * 0.3)

    slots_with_start = sum(
        1 for slots in schedule.stages.values()
        for s in slots if s.start_time
    )
    start_ratio = slots_with_start / total_slots if total_slots > 0 else 0
    scores.append(start_ratio * 0.3)

    if schedule.date:
        scores.append(0.2)
    else:
        scores.append(0.05)

    confidence = sum(scores)
    return round(min(confidence, 1.0), 2)
