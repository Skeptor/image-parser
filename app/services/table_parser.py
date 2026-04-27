import logging
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_pdf(image: Image.Image, pdf_path: Path) -> None:
    """Convert a PIL Image to a single-page PDF for camelot processing."""
    tmp_img = pdf_path.with_suffix(".png")
    rgb = image.convert("RGB")
    rgb.save(str(tmp_img))

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        w, h = image.size
        page_w, page_h = A4

        ratio = min(page_w / w, page_h / h)
        new_w = w * ratio
        new_h = h * ratio
        offset_x = (page_w - new_w) / 2
        offset_y = (page_h - new_h) / 2

        c = canvas.Canvas(str(pdf_path))
        c.drawImage(str(tmp_img), offset_x, offset_y + new_h, width=new_w, height=-new_h)
        c.save()
    except ImportError:
        import img2pdf
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(str(tmp_img)))
    finally:
        tmp_img.unlink(missing_ok=True)


def detect_tables(image: Image.Image, output_path: Path) -> list[dict[str, Any]]:
    """Detect tables in an image by converting to PDF and using camelot-py.

    Returns list of tables, each as {rows: list[list[str]], accuracy: float}.
    """
    try:
        import camelot
    except ImportError:
        logger.warning("camelot-py not available, table detection disabled")
        return []

    pdf_path = output_path / "tmp.pdf"
    _image_to_pdf(image, pdf_path)

    try:
        tables = camelot.read_pdf(str(pdf_path), flavor="grid", pages="1")

        if not tables or len(tables) == 0:
            tables = camelot.read_pdf(str(pdf_path), flavor="stream", pages="1")

        result = []
        for table in tables:
            rows = []
            df = table.parsed_dataframe
            for r in range(df.shape[0]):
                row = []
                for c in range(df.shape[1]):
                    cell = str(df.iat[r, c]).strip()
                    if cell:
                        row.append(cell)
                if row:
                    rows.append(row)
            result.append({"rows": rows, "accuracy": table.accuracy})
            logger.info("Detected table: %d rows, accuracy=%.1f", len(rows), table.accuracy)

        return result

    except Exception as e:
        logger.warning("Table detection with camelot failed: %s", e)
        return []
    finally:
        pdf_path.unlink(missing_ok=True)
