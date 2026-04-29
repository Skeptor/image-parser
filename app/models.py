from pydantic import BaseModel, Field


class ConcertSlot(BaseModel):
    group: str
    start_time: str
    end_time: str | None = None


class StageSchedule(BaseModel):
    stages: dict[str, list[ConcertSlot]] = Field(default_factory=dict)
    date: str | None = None  # Format: DD/MM/YYYY


class ParseResponse(BaseModel):
    success: bool
    data: StageSchedule | None = None
    confidence: float = 0.0
    raw_text: str = ""
    error: str | None = None
