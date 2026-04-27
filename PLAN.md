# Concert Schedule Image Parser — Plan

## Overview

A FastAPI service that accepts an image (upload or URL), extracts concert schedule
data via OCR, parses the tabular layout, and returns structured JSON:

```json
{
  "festival": "Woodstock 2025",
  "stages": {
    "Main Stage": [
      { "group": "The Strokes", "start_time": "18:00", "end_time": "18:45" },
      { "group": "Arctic Monkeys", "start_time": "19:00", "end_time": "19:50" }
    ],
    "Acoustic Stage": [
      { "group": "Bon Iver", "start_time": "17:30", "end_time": "18:15" }
    ]
  }
}
```

## Architecture

```
client
  │  POST /parse  { image: file | url: string }
  ▼
FastAPI (uvicorn)
  │
  ├─ ImageLoader        → download from URL or read uploaded file
  ├─ OCR Engine         → Tesseract / Google Cloud Vision / AWS Textract
  ├─ Table Parser       → detect grid structure, extract cells
  ├─ Schedule Extractor → heuristics / LLM to map cells → stages + slots
  └─ JSON Serializer    → validate and return canonical schema
```

## Tech Stack

| Layer            | Choice                          | Rationale                                          |
|------------------|---------------------------------|----------------------------------------------------|
| Framework        | FastAPI + uvicorn               | Async, auto OpenAPI docs, easy to dockerize        |
| OCR              | Tesseract (via pytesseract)     | Offline, no API key, good enough for printed text  |
| Table detection  | camelot-py / tabula-py          | Extract tabular regions from images/PDFs           |
| Fallback parsing | regex + heuristic rules         | Handle cases where table detection fails           |
| Validation       | Pydantic                        | Request/response models, schema enforcement         |
| Containerization | Docker + Docker Compose          | Reproducible environment with Tesseract system deps|
| Testing          | pytest + httpx                  | Unit tests + integration tests against live server  |

## Project Structure

```
image-parser/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, routes, lifecycle
│   ├── models.py           # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_loader.py   # fetch image from URL or upload
│   │   ├── ocr.py            # Tesseract wrapper
│   │   ├── table_parser.py   # detect tables, extract cells
│   │   └── schedule_extractor.py  # map cells → stages + slots
│   └── utils.py             # time normalization, string cleaning
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py         # integration tests
│   ├── test_ocr.py
│   ├── test_table_parser.py
│   ├── test_schedule_extractor.py
│   └── fixtures/            # sample concert images for testing
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## API Specification

### Endpoint: `POST /parse`

**Request** (multipart form):

| Field   | Type   | Required | Description                        |
|---------|--------|----------|------------------------------------|
| image   | file   | no*      | Image file (PNG, JPG, WEBP, PDF)   |
| url     | string | no*      | Public URL to an image             |

\* Exactly one of `image` or `url` must be provided.

**Query params** (optional):

| Param        | Type    | Default      | Description                        |
|--------------|---------|--------------|------------------------------------|
| lang         | string  | `eng`        | Tesseract language code            |
| psm          | int     | `6`          | Tesseract page segmentation mode   |
| use_llm      | boolean | `false`      | Enable LLM-based parsing fallback  |
| llm_api_key  | string  | env var      | API key for LLM fallback           |

**Response** (200 OK):

```json
{
  "success": true,
  "data": {
    "festival": "Summer Vibes Fest 2025",
    "date": "2025-07-19",
    "stages": {
      "Main Stage": [
        { "group": "The Strokes", "start_time": "18:00", "end_time": "18:45" },
        { "group": "Arctic Monkeys", "start_time": "19:00", "end_time": "19:50" }
      ],
      "Acoustic Stage": [
        { "group": "Bon Iver", "start_time": "17:30", "end_time": "18:15" }
      ]
    }
  },
  "confidence": 0.87,
  "raw_text": "..."
}
```

**Error responses**:

| Status | Reason                              |
|--------|-------------------------------------|
| 400    | Neither `image` nor `url` provided  |
| 400    | Unsupported file type               |
| 422    | Validation error (Pydantic)         |
| 500    | OCR failed / parsing failed         |

### Endpoint: `GET /health`

Returns `{"status": "ok"}` for liveness probes.

## Parsing Strategy

Concert schedules have a predictable structure. The parser works in phases:

### Phase 1 — OCR
- Run Tesseract on the full image with PSM 6 (uniform block of text)
- Extract raw text with word-level bounding boxes (`--psm 6`, `get_words=True`)
- Save intermediate text for debugging

### Phase 2 — Table Detection
- Use **camelot-py** (OpenCV backend) to detect table grids in the image
- Extract cells with their (x, y) coordinates
- If camelot finds no tables, fall back to line-based parsing

### Phase 3 — Heuristic Parsing (line-based fallback)
```
1. Split OCR text into lines
2. Detect header row: look for stage names (capitalized words, often centered)
3. Detect time columns: match patterns like HH:MM, H:MM AM/PM
4. Detect group names: remaining text between times and stage headers
5. Group rows under their closest stage header above them
```

### Phase 4 — Post-processing
- Normalize times to 24h format (`6:00 PM` → `18:00`)
- Strip stage suffixes (`Stage`, `Stg`, `→`)
- Deduplicate entries
- Compute confidence score based on how many fields were matched

### Phase 5 — LLM Fallback (optional)
If heuristic parsing yields low confidence (< 0.5), send the raw OCR text
to an LLM with a structured prompt to extract the schedule. This is opt-in
via `use_llm=true`.

## Docker Setup

### Dockerfile

```dockerfile
# Stage 1: system deps (Tesseract, libopencv)
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Stage 2: runtime
FROM base AS runtime
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLm_API_KEY=${LLM_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      start_period: 10s
```

## Implementation Steps

### Step 1 — Scaffolding (Day 1)
- [ ] Initialize project structure
- [ ] Set up FastAPI app with `/parse` and `/health` routes
- [ ] Define Pydantic models for request/response
- [ ] Write Dockerfile + docker-compose.yml
- [ ] Verify container builds and serves `/health`

### Step 2 — Image Loading (Day 1)
- [ ] Implement `ImageLoader` — accept file upload or fetch from URL
- [ ] Validate image format (PNG, JPG, WEBP)
- [ ] Write tests with sample images

### Step 3 — OCR Integration (Day 2)
- [ ] Wrap pytesseract with configurable PSM and language
- [ ] Return raw text + word bounding boxes
- [ ] Write tests against fixture images

### Step 4 — Table Detection (Day 2-3)
- [ ] Integrate camelot-py for table grid detection
- [ ] Extract cells with coordinates
- [ ] Write tests with known table images

### Step 5 — Heuristic Parser (Day 3-4)
- [ ] Implement header detection (stage names)
- [ ] Implement time slot extraction with regex
- [ ] Implement group name extraction
- [ ] Build stage → slots mapping
- [ ] Write tests with sample concert schedules

### Step 6 — Post-processing & Confidence (Day 4)
- [ ] Normalize time formats
- [ ] Clean artist/stage names
- [ ] Calculate confidence score
- [ ] End-to-end tests with real concert images

### Step 7 — LLM Fallback (Day 5, optional)
- [ ] Implement opt-in LLM parsing path
- [ ] Write structured prompt for schedule extraction
- [ ] Parse LLM response into canonical schema

### Step 8 — Polish (Day 5)
- [ ] Add request validation and error handling
- [ ] Add logging and metrics
- [ ] Write README with usage examples
- [ ] Final integration tests

## Risks & Mitigations

| Risk                                  | Mitigation                                              |
|---------------------------------------|---------------------------------------------------------|
| Tesseract fails on handwritten text   | Accept only printed schedules; document limitation      |
| Complex multi-column layouts          | camelot-py + LLM fallback for hard cases                |
| Non-English festivals                 | Support multi-language Tesseract packs                  |
| Large image files                     | Enforce size limit (e.g., 10 MB), resize before OCR     |
| LLM cost if fallback is used often    | Make LLM opt-in, cache results, set timeout             |

## Future Enhancements

- Support PDF input (many schedules are distributed as PDFs)
- Batch endpoint: parse multiple images at once
- Web UI for upload and preview
- Configurable output formats (ICS calendar, CSV)
- Fine-tuned model for schedule-specific OCR
