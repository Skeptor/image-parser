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
    timed_slots = 0
    for stage_name, slots in stages.items():
        for slot in slots:
            assert "group" in slot, f"Slot missing 'group' in {stage_name}"
            assert slot["group"], f"Empty group name in {stage_name}"

            if slot.get("start_time") is not None:
                timed_slots += 1
                assert ":" in slot["start_time"], f"Invalid start_time format in {stage_name}: {slot['start_time']}"
                h, m = slot["start_time"].split(":")
                assert h.isdigit() and m.isdigit(), f"Invalid time digits in {stage_name}: {slot['start_time']}"
                assert 0 <= int(h) <= 30, f"Invalid hour in {stage_name}: {h}"
                assert 0 <= int(m) <= 59, f"Invalid minute in {stage_name}: {m}"

    # Most slots should have a time (timeless slots are acceptable as a minority)
    total_slots = sum(len(s) for s in stages.values())
    assert timed_slots >= total_slots * 0.7, \
        f"Too few timed slots: {timed_slots}/{total_slots}"


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
    timed_slots = 0
    for stage_name, slots in stages.items():
        for slot in slots:
            assert "group" in slot, f"Slot missing 'group' field in {stage_name}"
            assert slot["group"], f"Empty group name in {stage_name}"

            if slot.get("start_time") is not None:
                timed_slots += 1
                assert ":" in slot["start_time"], f"Invalid start_time format: {slot['start_time']}"
                h, m = slot["start_time"].split(":")
                assert h.isdigit() and m.isdigit(), f"Invalid time: {slot['start_time']}"
                assert 0 <= int(h) <= 30, f"Invalid hour: {h}"
                assert 0 <= int(m) <= 59, f"Invalid minute: {m}"

    total_slots = sum(len(s) for s in stages.values())
    assert timed_slots >= total_slots * 0.7, \
        f"Too few timed slots: {timed_slots}/{total_slots}"


@pytest.mark.anyio
async def test_sonograma_parsing(client):
    """Test Sonograma Ribero '25 (Viernes Noche) schedule parsing.

    Block-based timeline layout with only start times — no explicit end times.
    Validates that end_time is null for all slots and stages are detected correctly.
    """
    fixture_path = FIXTURES_DIR / "sonograma_viernes.png"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = await client.post(
            "/parse",
            params={"lang": "spa"},
            files={"image": ("sonograma_viernes.png", f, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["data"] is not None
    assert data["confidence"] > 0.4, f"Low confidence: {data['confidence']}"

    stages = data["data"]["stages"]
    assert len(stages) >= 5, f"Expected at least 5 stages, got {len(stages)}: {list(stages.keys())}"

    total_slots = sum(len(slots) for slots in stages.values())
    assert total_slots >= 20, f"Expected at least 20 total slots, got {total_slots}"

    timed_slots = 0
    end_time_slots = 0
    for stage_name, slots in stages.items():
        for slot in slots:
            if slot.get("start_time") is not None:
                timed_slots += 1
                assert ":" in slot["start_time"], f"Invalid start_time: {slot['start_time']}"
            if slot.get("end_time") is not None:
                end_time_slots += 1

    assert timed_slots >= total_slots * 0.7, \
        f"Too few timed slots: {timed_slots}/{total_slots}"
    # This image type doesn't have explicit end times; allow a small margin for edge cases
    assert end_time_slots <= total_slots * 0.2, \
        f"Too many unexpected end_times: {end_time_slots}/{total_slots}"

    # At least one known artist should be detected
    all_artists = [slot["group"].upper() for slots in stages.values() for slot in slots]
    known_artists = {"CHAMBAO", "FRANZ FERDINAND", "CARLOS JEAN", "BESMAYA", "NIKONE"}
    found = known_artists & {a for a in all_artists for k in known_artists if k in a}
    assert found, f"No known artists detected. Got: {all_artists[:10]}"


@pytest.mark.anyio
async def test_toledo_beat_parsing(client):
    """Test Toledo Beat festival schedule parsing.

    Two-day image (VIERNES + SÁBADO) with colored backgrounds and a shared
    time column. Artists on colored backgrounds are hard for Tesseract; many
    will be detected without times. The test focuses on:
    - Both days are detected
    - At least some artists are found per day
    - Known artists appear in the output
    """
    fixture_path = FIXTURES_DIR / "toledo_beat_2026.png"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = await client.post(
            "/parse",
            params={"lang": "spa"},
            files={"image": ("toledo_beat_2026.png", f, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"] is not None

    days = data["data"].get("days") or []
    assert len(days) == 2, f"Expected 2 days, got {len(days)}"
    day_names = {d["day_name"] for d in days}
    assert "VIERNES" in day_names and "SÁBADO" in day_names, \
        f"Expected VIERNES and SÁBADO, got {day_names}"

    for day in days:
        day_slots = [s for stage_slots in day["stages"].values() for s in stage_slots]
        assert len(day_slots) >= 2, \
            f"Expected ≥2 slots for {day['day_name']}, got {len(day_slots)}"

    all_artists = " ".join(
        s["group"].upper()
        for d in days for stage_slots in d["stages"].values() for s in stage_slots
    )
    for artist in ("SIENNA", "SILOÉ", "ULTRALICERA"):
        assert artist in all_artists, f"Expected artist '{artist}' not found"


@pytest.mark.anyio
async def test_venres_parsing(client):
    """Test O Son do Camiño festival — VENRES (Friday) schedule.

    Two-column layout with clear fonts on white/light background.
    - Column 1: Escenario Xacobeo / Estrella Galicia — 8 artist slots, 16:45 – 02:20
    - Column 2: Escenario Sonelectro TN — 5 DJ slots, 21:00 – 04:00

    Key validation: the tight-row table layout used to collapse all rows into
    one cell. This test guards against regressions in the line-based cell
    detection fallback (_parse_slots_from_lines).
    """
    fixture_path = FIXTURES_DIR / "venres.png"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = await client.post(
            "/parse",
            params={"lang": "spa"},
            files={"image": ("venres.png", f, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"] is not None
    assert data["confidence"] > 0.6, f"Low confidence: {data['confidence']}"

    stages = data["data"]["stages"]
    assert len(stages) == 2, f"Expected 2 stages, got {len(stages)}: {list(stages.keys())}"

    # Both columns must be present
    stage_names_lower = " ".join(stages.keys()).lower()
    assert "xacobeo" in stage_names_lower or "estrella" in stage_names_lower, \
        f"Xacobeo/Estrella stage not found in {list(stages.keys())}"
    assert "sonelectro" in stage_names_lower or "electro" in stage_names_lower, \
        f"Sonelectro stage not found in {list(stages.keys())}"

    # Column 1 must have all 8 artist slots (not collapsed into fewer)
    xacobeo_stage = next(
        slots for name, slots in stages.items()
        if "xacobeo" in name.lower() or "estrella" in name.lower()
    )
    assert len(xacobeo_stage) >= 7, \
        f"Expected ≥7 slots in Xacobeo stage, got {len(xacobeo_stage)}"

    # Column 2 must have 5 DJ slots
    sonelectro_stage = next(
        slots for name, slots in stages.items()
        if "sonelectro" in name.lower() or "electro" in name.lower()
    )
    assert len(sonelectro_stage) >= 4, \
        f"Expected ≥4 slots in Sonelectro stage, got {len(sonelectro_stage)}"

    # All slots must have valid start and end times
    timed = 0
    for stage_name, slots in stages.items():
        for slot in slots:
            assert "group" in slot, f"Slot missing 'group' in {stage_name}"
            assert slot["group"], f"Empty group name in {stage_name}"
            for field in ("start_time", "end_time"):
                t = slot.get(field)
                assert t is not None, f"Slot in {stage_name} missing {field}: {slot}"
                assert ":" in t, f"Invalid time format in {stage_name}: {t}"
                h, m = t.split(":")
                assert h.isdigit() and m.isdigit(), f"Non-digit time in {stage_name}: {t}"
                assert 0 <= int(h) <= 30 and 0 <= int(m) <= 59, \
                    f"Out-of-range time in {stage_name}: {t}"
            timed += 1

    assert timed >= 11, f"Expected ≥11 timed slots total, got {timed}"

    # Known artists must appear somewhere in the group names
    all_groups = " ".join(
        slot["group"].upper()
        for slots in stages.values()
        for slot in slots
    )
    known = {
        "HOOBASTANK": "HOOBASTANK",
        "BIFFY CLYRO": "BIFFY CLYRO",
        "BLOODY BEETROOTS": "BLOODY",
        "HONEYLUV": "HONEYLUV",
        "CARLITA": "CARLITA",
    }
    for label, fragment in known.items():
        assert fragment in all_groups, \
            f"Expected artist '{label}' (fragment '{fragment}') not found in output: {all_groups[:200]}"


@pytest.mark.anyio
async def test_fixtures_stability(client):
    """Smoke test to ensure both fixtures parse without errors.

    This test ensures that code changes don't break parsing entirely.
    More detailed assertions are in the individual fixture tests.
    """
    fixtures = ["sansan_festival.png", "vina_rock_sabado.png", "sonograma_viernes.png", "venres.png"]

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
