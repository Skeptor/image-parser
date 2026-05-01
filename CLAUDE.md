# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI service that extracts concert schedules from images using OCR (Tesseract) and returns structured JSON. Accepts image uploads or URLs, performs OCR, detects table structures, and parses schedule data using heuristics.

## Commands

### Run with Docker (recommended)
```bash
docker compose up --build
```

### Run locally (requires Tesseract installed)
```bash
# Install Tesseract first: sudo apt install tesseract-ocr
pip install -r requirements.txt
uvicorn app.app:app --reload
```

### Run tests
```bash
pytest
pytest tests/test_main.py -v          # specific test file
pytest -k "test_health" -v            # specific test by name
```

## Architecture

**Request flow:** `POST /parse` → ImageLoader → OCR → (Column Detection | Table Parser) → Schedule Extractor → JSON Response

**Key design decision:** The parser uses a three-tier strategy based on image layout:
1. **Multi-column images** — `column_detector.py` splits the image, OCR runs per column, results merged
2. **Table-based layouts** — `table_parser.py` uses camelot-py to detect grid structures
3. **Fallback** — `schedule_extractor.py` uses line-based heuristics on raw OCR text

**Entry points:**
- `app/app.py` — FastAPI app instance, CORS, lifespan
- `app/main.py` — Route handlers (`/parse`, `/health`)

**Parsing pipeline in `schedule_extractor.py`:**
- `extract_schedule()` — Main entry point, chooses parsing strategy
- `parse_column_text()` — Parses single column OCR text
- `parse_from_table_rows()` — Parses camelot table data
- `parse_from_text()` — Heuristic fallback for unstructured text
- `KNOWN_STAGES` dict maps garbled OCR stage names to canonical names (Spanish festival focused)

## Testing

Tests use `pytest-asyncio` with `httpx.AsyncClient` for async endpoint testing. The `anyio_backend` fixture in `conftest.py` configures asyncio.

### Test Fixtures

- `tests/fixtures/vina_rock_sabado.png` — Viña Rock festival schedule (good parsing accuracy)
- `tests/fixtures/sansan_festival.png` — San San Festival schedule (colored backgrounds, more challenging)
- `tests/test_fixtures.py` — Regression tests ensuring these images continue to parse correctly

Run fixture tests: `docker-compose exec -T api pytest tests/test_fixtures.py -v`

## OCR Improvements Made

Recent improvements to handle complex multi-column schedules:

1. **Smart column detection** — `column_detector.py` finds column boundaries using word position clustering instead of fixed widths. Runs OCR with multiple preprocessing passes (grayscale + contrast enhancement, original, and inverted) to maximize word detection on colored backgrounds.

2. **Header-based column splitting** — Detects stage names in the header region and uses them to define column boundaries. Falls back to gap-based detection if headers aren't found.

3. **Spanish language support** — Added `tesseract-ocr-spa` to Docker image for better accuracy on Spanish festival schedules.

4. **Fallback stage naming** — When OCR can't read a stage name (e.g., light text on dark background), assigns positional names like "Escenario 2" to ensure all columns are parsed.

## Known Limitations

- **Colored backgrounds** — Light text on dark/colored backgrounds (especially blues/purples) can be difficult for Tesseract to read. Some stage names may be incomplete or require fallback naming.
- **Text fragmentation** — OCR may split artist names into separate words or create duplicates. The parser attempts to deduplicate but results may vary.
- **Confidence scores** — Images with colored backgrounds typically score lower (0.5-0.8) vs. high-contrast images (0.7-0.9+).
- **Best results** — Printed schedules with high contrast, minimal colors, and clear table structures work best.

## API

- `POST /parse` — Main endpoint. Accepts `image` (file) or `url` (form field). Optional query params: `lang` (Tesseract language, default "eng"), `psm` (page segmentation mode, default 6)
- `GET /health` — Liveness probe
