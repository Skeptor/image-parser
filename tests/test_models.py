from app.models import ParseResponse, ConcertSlot, StageSchedule


class TestParseResponse:
    def test_success_response(self):
        resp = ParseResponse(
            success=True,
            data=StageSchedule(
                stages={
                    "Main": [
                        ConcertSlot(group="Band A", start_time="18:00", end_time="18:45")
                    ]
                },
            ),
            confidence=0.9,
            raw_text="sample",
        )
        assert resp.success is True
        assert resp.data.stages["Main"][0].group == "Band A"
        assert resp.confidence == 0.9

    def test_failure_response(self):
        resp = ParseResponse(success=False, error="Something went wrong")
        assert resp.success is False
        assert resp.data is None
        assert resp.error == "Something went wrong"
