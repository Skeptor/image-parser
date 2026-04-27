from app.services.schedule_extractor import normalize_time


class TestNormalizeTime:
    def test_24h_format_passthrough(self):
        assert normalize_time("18:00") == "18:00"

    def test_12h_pm(self):
        assert normalize_time("6:00 PM") in ("18:00", "06:00")

    def test_12h_am(self):
        assert normalize_time("9:00 AM") == "09:00"

    def test_noon(self):
        assert normalize_time("12:00 PM") == "12:00"

    def test_midnight(self):
        assert normalize_time("12:00 AM") == "00:00"

    def test_no_match(self):
        assert normalize_time("hello") == "hello"
