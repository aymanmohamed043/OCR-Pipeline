from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2 as cv
import numpy as np

try:
    from src.tesseract_pipeline import run_tesseract_ocr_pipeline
except ImportError:
    from tesseract_pipeline import run_tesseract_ocr_pipeline

try:
    from src.paddleocr_pipeline import run_paddleocr_ocr_pipeline
except ImportError:
    from paddleocr_pipeline import run_paddleocr_ocr_pipeline

app = FastAPI(title="OCR API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


async def _decode_image_upload(file: UploadFile) -> tuple[np.ndarray, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv.imdecode(np_buffer, cv.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    return image, file.filename or "upload"


def _text_stats(text: str) -> dict:
    return {
        "characters": len(text.strip()),
        "lines": len([line for line in text.splitlines() if line.strip()]),
    }


async def _run_ocr(
    image: np.ndarray,
    filename: str,
    engine: str,
    config: str,
    paddle_lang: str = "en",
) -> dict:
    engine = engine.lower().strip()
    if engine not in ("tesseract", "paddle"):
        raise HTTPException(
            status_code=400,
            detail='engine must be "tesseract" or "paddle".',
        )

    paddle_lang = (paddle_lang or "en").strip().lower() or "en"

    if engine == "tesseract":
        text = run_tesseract_ocr_pipeline(image, config=config)
    else:
        try:
            text = run_paddleocr_ocr_pipeline(
                image,
                config=config,
                lang=paddle_lang,
            )
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"PaddleOCR is not available: {exc}",
            ) from exc

    stats = _text_stats(text)
    out = {
        "filename": filename,
        "text": text,
        "engine": engine,
        **stats,
    }
    if engine == "paddle":
        out["paddle_lang"] = paddle_lang
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    engine: str = Query(
        "tesseract",
        description='OCR backend: "tesseract" or "paddle".',
    ),
    config: str = Query("", description="Tesseract config (e.g. --psm 6). Ignored by Paddle for now."),
    lang: str = Query(
        "en",
        description='PaddleOCR language code when engine=paddle (e.g. en, ar, ch). Ignored for Tesseract.',
    ),
):
    image, filename = await _decode_image_upload(file)
    try:
        return await _run_ocr(image, filename, engine, config, paddle_lang=lang)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@app.post("/ocr/paddle")
async def ocr_paddle(
    file: UploadFile = File(...),
    config: str = Query("", description="Reserved for future Paddle options."),
    lang: str = Query(
        "en",
        description="PaddleOCR language code (e.g. en, ar, ch, korean, japan).",
    ),
):
    image, filename = await _decode_image_upload(file)
    try:
        return await _run_ocr(image, filename, "paddle", config, paddle_lang=lang)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
