from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn
import cv2 as cv
import numpy as np


from tesseract_pipeline import run_tesseract_ocr_pipeline

app = FastAPI(title="OCR API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ocr")   
async def ocr_image(file: UploadFile = File(...), config: str = "") -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv.imdecode(np_buffer, cv.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        text = run_tesseract_ocr_pipeline(image, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    return {
        "filename": file.filename,
        "text": text,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)