"""Tests for fixture images to ensure parsing stability across changes."""
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from app.app import app


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_sansan_festival_parsing(client):
    """Test San San Festival schedule parsing.

    This image has colored backgrounds which make OCR challenging.
    The test validates that:
    - Parsing succeeds
    - 4 columns/stages are detected
    - Slots are extracted (with some OCR errors expected)
    - Times are in valid format
    """
    fixture_path = FIXTURES_DIR / "sansan_festival.png"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = await client.post(
            "/parse",
            params={"lang": "spa"},
            files={"image": ("sansan_festival.png", f, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()

    # Basic success assertions
    assert data["success"] is True
    assert data["confidence"] > 0.6, f"Low confidence: {data['confidence']}"
    assert data["data"] is not None

    # Stage detection assertions
    stages = data["data"]["stages"]
    assert len(stages) == 4, f"Expected 4 stages, got {len(stages)}: {list(stages.keys())}"

    # Check that known stages are detected (or fallback names are used)
    stage_names = list(stages.keys())
    assert any("SanSan" in name or "Escenario 1" in name for name in stage_names), \
        f"SanSan stage not found in {stage_names}"
    assert any("Johnnie" in name or "Walker" in name for name in stage_names), \
        f"Johnnie Walker stage not found in {stage_names}"
    assert any("Santuario" in name or "Adamo" in name for name in stage_names), \
        f"Santuario stage not found in {stage_names}"

    # Validate slot extraction - expect at least 10 total slots across all stages
    total_slots = sum(len(slots) for slots in stages.values())
    assert total_slots >= 10, f"Expected at least 10 total slots, got {total_slots}"

    # Validate slot structure and data quality
    for stage_name, slots in stages.items():
        for slot in slots:
            # Check required fields exist
            assert "group" in slot, f"Slot missing 'group' in {stage_name}"
            assert "start_time" in slot, f"Slot missing 'start_time' in {stage_name}"

            # Validate artist name is not empty
            assert slot["group"], f"Empty group name in {stage_name}"

            # Validate time format (HH:MM)
            assert ":" in slot["start_time"], f"Invalid start_time format in {stage_name}: {slot['start_time']}"
            h, m = slot["start_time"].split(":")
            assert h.isdigit() and m.isdigit(), f"Invalid time digits in {stage_name}: {slot['start_time']}"
            assert 0 <= int(h) <= 30, f"Invalid hour in {stage_name}: {h}"
            assert 0 <= int(m) <= 59, f"Invalid minute in {stage_name}: {m}"


@pytest.mark.anyio
async def test_vina_rock_parsing(client):
    """Test Viña Rock schedule parsing.

    This image has better contrast than San San but still has OCR challenges.
    The test validates that:
    - Parsing succeeds
    - Multiple stages are detected (at least 5)
    - Stage names match known venues
    - A reasonable number of slots are extracted
    - Times are in valid format
    """
    fixture_path = FIXTURES_DIR / "vina_rock_sabado.png"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = await client.post(
            "/parse",
            params={"lang": "spa"},
            files={"image": ("vina_rock_sabado.png", f, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()

    # Basic success assertions
    assert data["success"] is True
    assert data["confidence"] > 0.7, f"Low confidence: {data['confidence']}"
    assert data["data"] is not None

    # Stage detection assertions
    stages = data["data"]["stages"]
    assert len(stages) >= 5, f"Expected at least 5 stages, got {len(stages)}: {list(stages.keys())}"

    # Check for known Viña Rock stages
    stage_names = list(stages.keys())
    assert any("Negrita" in name for name in stage_names), \
        f"Negrita stage not found in {stage_names}"
    assert any("Cutty" in name or "Sark" in name for name in stage_names), \
        f"Cutty Sark stage not found in {stage_names}"

    # Validate slot extraction - expect at least 20 total slots
    total_slots = sum(len(slots) for slots in stages.values())
    assert total_slots >= 20, f"Expected at least 20 total slots, got {total_slots}"

    # Validate that most stages have slots
    stages_with_slots = [name for name, slots in stages.items() if len(slots) > 0]
    assert len(stages_with_slots) >= 5, \
        f"Expected at least 5 stages with slots, got {len(stages_with_slots)}"

    # Validate slot structure and data quality
    for stage_name, slots in stages.items():
        for slot in slots:
            assert "group" in slot, f"Slot missing 'group' field in {stage_name}"
            assert "start_time" in slot, f"Slot missing 'start_time' field in {stage_name}"
            assert slot["group"], f"Empty group name in {stage_name}"
            assert slot["start_time"], f"Empty start_time in {stage_name}"

            # Validate time format
            assert ":" in slot["start_time"], f"Invalid start_time format: {slot['start_time']}"
            h, m = slot["start_time"].split(":")
            assert h.isdigit() and m.isdigit(), f"Invalid time: {slot['start_time']}"
            assert 0 <= int(h) <= 30, f"Invalid hour: {h}"
            assert 0 <= int(m) <= 59, f"Invalid minute: {m}"


@pytest.mark.anyio
async def test_fixtures_stability(client):
    """Smoke test to ensure both fixtures parse without errors.

    This test ensures that code changes don't break parsing entirely.
    More detailed assertions are in the individual fixture tests.
    """
    fixtures = ["sansan_festival.png", "vina_rock_sabado.png"]

    for fixture_name in fixtures:
        fixture_path = FIXTURES_DIR / fixture_name
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        with open(fixture_path, "rb") as f:
            resp = await client.post(
                "/parse",
                params={"lang": "spa"},
                files={"image": (fixture_name, f, "image/png")},
            )

        assert resp.status_code == 200, f"Failed to parse {fixture_name}"
        data = resp.json()
        assert data["success"] is True, f"Parsing failed for {fixture_name}: {data.get('error')}"
        assert len(data["data"]["stages"]) > 0, f"No stages detected in {fixture_name}"
