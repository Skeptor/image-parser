from pydantic import BaseModel, Field, model_validator


class ConcertSlot(BaseModel):
    group: str
    start_time: str
    end_time: str | None = None

    @model_validator(mode="after")
    def fix_swapped_times(self) -> "ConcertSlot":
        if self.end_time is None:
            return self
        start = self.start_time
        end = self.end_time
        # Midnight-crossing: start hour begins with "2" and end with "0" — keep as-is
        if start[:1] == "2" and end[:1] == "0":
            return self
        if end < start:
            self.start_time, self.end_time = end, start
        return self


class StageSchedule(BaseModel):
    stages: dict[str, list[ConcertSlot]] = Field(default_factory=dict)
    date: str | None = None  # Format: DD/MM/YYYY


class ParseResponse(BaseModel):
    success: bool
    data: StageSchedule | None = None
    confidence: float = 0.0
    raw_text: str = ""
    error: str | None = None
