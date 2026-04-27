# Concert Schedule Image Parser

FastAPI service that extracts concert schedules from images and returns structured JSON.

## Quick Start

```bash
docker compose up --build
```

The API is available at `http://localhost:8000` with auto-generated docs at `http://localhost:8000/docs`.

## Usage

### Upload an image file

```bash
curl -X POST http://localhost:8000/parse \
  -F "image=@schedule.png"
```

### Provide an image URL

```bash
curl -X POST http://localhost:8000/parse \
  -F "url=https://example.com/schedule.png"
```

### Optional query parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `lang` | string | `eng` | Tesseract language code |
| `psm` | int | `6` | Page segmentation mode |

```bash
curl -X POST "http://localhost:8000/parse?lang=spa" \
  -F "image=@schedule.png"
```

## Response

```json
{
  "success": true,
  "data": {
    "festival": "Summer Music Fest 2025",
    "date": "2025-07-19",
    "stages": {
      "Main": [
        { "group": "The Strokes", "start_time": "18:00", "end_time": "18:45" },
        { "group": "Arctic Monkeys", "start_time": "19:00", "end_time": "19:50" }
      ],
      "Acoustic": [
        { "group": "Bon Iver", "start_time": "17:30", "end_time": "18:15" }
      ]
    }
  },
  "confidence": 0.87,
  "raw_text": "..."
}
```

### Error response

```json
{
  "success": false,
  "data": null,
  "confidence": 0.0,
  "error": "No text detected in the image",
  "raw_text": ""
}
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/parse` | Parse concert schedule from image |

## How It Works

1. **Image loading** — accepts file upload or downloads from URL (with browser-like headers to bypass bot protection)
2. **OCR** — Tesseract extracts text and word bounding boxes
3. **Table detection** — camelot-py detects grid structures (image converted to PDF for processing)
4. **Heuristic parsing** — identifies stage headers, time slots (12h/24h), and group names
5. **Post-processing** — normalizes times to 24h format, cleans names, computes confidence score

## Architecture

```
client
  │  POST /parse  { image: file | url: string }
  ▼
FastAPI (uvicorn)
  │
  ├─ ImageLoader        → download from URL or read uploaded file
  ├─ OCR Engine         → Tesseract
  ├─ Table Parser       → camelot-py grid detection
  ├─ Schedule Extractor → heuristics (headers, times, groups)
  └─ JSON Serializer    → Pydantic validation
```

## Tech Stack

- **FastAPI** + uvicorn — async web framework
- **Tesseract** (pytesseract) — offline OCR
- **camelot-py** — table grid detection
- **reportlab** / **img2pdf** — image-to-PDF conversion
- **Pydantic** — request/response validation
- **Pillow** — image processing

## Project Structure

```
app/
├── app.py                    # FastAPI app, routes, CORS
├── models.py                 # Pydantic schemas
├── services/
│   ├── image_loader.py       # file upload / URL fetch
│   ├── ocr.py                # Tesseract wrapper
│   ├── table_parser.py       # camelot-py table detection
│   └── schedule_extractor.py # heuristic parser
├── utils.py                  # time normalization, confidence scoring
tests/
├── test_main.py              # integration tests
├── test_models.py
├── test_schedule_extractor.py
├── test_utils.py
```

## Limitations

- Works best with **printed, high-contrast** schedules (PDF exports, screenshots)
- Complex multi-column layouts may produce noisy output
- Handwritten text is not supported
- Confidence score indicates parsing quality; below 0.5 suggests manual review

## Running Locally (Without Docker)

```bash
# Install Tesseract system package
sudo apt install tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt

# Run
uvicorn app.app:app --reload
```
