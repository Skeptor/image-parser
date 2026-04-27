from app.utils import clean_group_name, clean_stage_name, compute_confidence
from app.models import ConcertSlot, StageSchedule


class TestCleanGroupName:
    def test_removes_separators(self):
        assert clean_group_name("Band A | Band B") == "Band A  Band B"

    def test_strips_whitespace(self):
        assert clean_group_name("  The Band  ") == "The Band"

    def test_collapses_spaces(self):
        assert clean_group_name("Band   Name") == "Band Name"


class TestCleanStageName:
    def test_removes_stage_suffix(self):
        assert clean_stage_name("Main Stage") == "Main"

    def test_removes_stg_suffix(self):
        assert clean_stage_name("B Stage") == "B"

    def test_fallback(self):
        assert clean_stage_name("Stage") == "Main Stage"

    def test_no_change(self):
        assert clean_stage_name("Red Zone") == "Red"


class TestComputeConfidence:
    def test_empty_schedule(self):
        s = StageSchedule()
        assert compute_confidence(s) == 0.0

    def test_no_stages(self):
        assert compute_confidence(None) == 0.0

    def test_populated_schedule(self):
        s = StageSchedule(
            festival="Test Fest",
            stages={
                "Main": [
                    ConcertSlot(group="Band A", start_time="18:00", end_time="18:45"),
                    ConcertSlot(group="Band B", start_time="19:00", end_time="19:50"),
                ],
                "Side": [
                    ConcertSlot(group="Band C", start_time="17:00", end_time="17:45"),
                ],
            },
        )
        conf = compute_confidence(s)
        assert 0 < conf <= 1.0

    def test_missing_end_times(self):
        s = StageSchedule(
            festival="Test",
            stages={
                "Main": [
                    ConcertSlot(group="Band A", start_time="18:00", end_time=None),
                ],
            },
        )
        conf = compute_confidence(s)
        assert 0 < conf < 1.0
