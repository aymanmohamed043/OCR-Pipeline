
```
# OCR Pipeline

Optical Character Recognition system supporting Tesseract and PaddleOCR with a web interface.

## Overview

An OCR pipeline that extracts text from images with support for multiple OCR engines and languages.

**Features:**
- Dual OCR engines: Tesseract and PaddleOCR
- Multi-language support (12+ languages)
- Web interface for image upload and OCR options
- REST API endpoints
- Docker containerization
- Text statistics (character count, line detection)

## Project Structure

```text
ocr_pipeline/
├── src/
│   ├── api.py                    # FastAPI application & REST endpoints
│   ├── tesseract_pipeline.py     # Tesseract OCR implementation
│   ├── paddleocr_pipeline.py     # PaddleOCR implementation with multi-lang support
│   ├── preprocessing.py          # Image preprocessing (grayscale conversion, etc.)
│   ├── ocr_engines.py            # OCR engine utilities
│   ├── debug_utilities.py        # Debug & testing helpers
│   └── evaluation.py             # Evaluation metrics (expandable)
├── templates/
│   └── index.html                # Web UI (OCR Studio)
├── static/
│   └── style.css                 # Styling for the web interface
├── data/                         # Test images and sample data
├── ocr_pipeline.ipynb            # Jupyter notebook for experimentation
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
└── README.md                     # This file
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda
- Docker (optional)

### Local Setup

1. Create a virtual environment
   ```bash
   python -m venv ocr-pipeline-venv
   ocr-pipeline-venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API server
   ```bash
   python -m src.api
   ```
   
Access the web interface at `http://localhost:8000`

## API Endpoints
### Health Check
```
GET /health
```
Returns API status.

### OCR Endpoint
```
POST /ocr
```
Performs OCR on uploaded image.

Parameters:
- `file`: Image file (required)
- `engine`: "tesseract" or "paddle" (default: tesseract)
- `config`: Tesseract options (optional, e.g. --psm 6)
- `lang`: Language code for PaddleOCR (optional, default: en)

Response:
```json
{
  "filename": "document.jpg",
  "text": "Extracted text...",
  "engine": "tesseract",
  "characters": 150,
  "lines": 5
}
```

### PaddleOCR Endpoint
```
POST /ocr/paddle
```
PaddleOCR with multi-language support.

Languages: en, ar, ch, chinese_cht, japan, korean, fr, de, es, ru, th, hi

### `src/api.py`
Main FastAPI application serving the web interface and REST API.

**Key Functions:**
- `health()`: Health check endpoint
- `index()`: Serves the OCR Studio web interface
- `ocr_image()`: Generic OCR endpoint
- `ocr_paddle()`: PaddleOCR-specific endpoint
- `_decode_image_upload()`: Validates and decodes image uploads
- `_run_ocr()`: Orchestrates OCR processing with selected engine
- `_text_stats()`: Calculates text statistics (characters, lines)

---

### `src/tesseract_pipeline.py`
Tesseract OCR pipeline wrapper.

**Key Functions:**
- `run_tesseract_ocr(image, config)`: Performs OCR with optional Tesseract configuration
- `run_tesseract_ocr_pipeline(image, config)`: Full pipeline with preprocessing

**Popular Tesseract Config Options:**
- `--psm 6`: Assume a single uniform block of text
- `--psm 3`: Fully automatic page segmentation
- `--oem 3`: Use both legacy and neural network OCR modes

---

### `src/paddleocr_pipeline.py`
PaddleOCR multi-language pipeline.

**Key Functions:**
- `_get_paddle_ocr(lang)`: Lazy-loads PaddleOCR instance by language (cached for efficiency)
- `_normalize_lang(lang)`: Normalizes language codes
- `_text_from_paddle_result(result)`: Extracts text from PaddleOCR result format
- `run_paddleocr_ocr_pipeline(image, config, lang)`: Full pipeline with language support

**Supported Languages:**
- English, Arabic, Chinese (Simplified & Traditional)
- Japanese, Korean, French, German, Spanish, Russian, Thai, Hindi

---

### `src/preprocessing.py`
Image preprocessing utilities.

**Key Functions:**
- `to_grayscale(image)`: Converts BGR/RGB images to grayscale for better OCR accuracy

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for REST API |
| `uvicorn` | ASGI server |
| `opencv-python` | Image processing |
| `pytesseract` | Tesseract OCR wrapper |
| `paddleocr` | PaddleOCR engine |
| `paddlepaddle` | PaddleOCR backend |
| `numpy` | Numerical computing |
| `Pillow` | Image handling |
| `jinja2` | HTML templating |
| `python-multipart` | File upload handling |

---

## 🐳 Docker Deployment

Build and run the application in a Docker container:

```bash
# Build the image
docker build -t ocr-pipeline .

# Run the container
docker run -p 8000:8000 ocr-pipeline

# Access the web interface
# http://localhost:8000
```

The Dockerfile includes:
- Python 3.10 slim base image
- OpenCV and system dependencies (libglib2.0-0, libsm6, etc.)
- **Tesseract binary** with English language data
- OpenMP support for Paddle
- All Python dependencies

---

## 💡 Usage Examples

### Via Web UI
1. Navigate to `http://localhost:8000`
2. Upload an invoice or document image
3. Select engine (Tesseract or PaddleOCR)
4. Click "Run OCR" and copy the extracted text

### Via cURL (Tesseract)
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@document.jpg" \
  -F "engine=tesseract" \
  -F "config=--psm 6"
```

### Via cURL (PaddleOCR - Arabic)
```bash
curl -X POST "http://localhost:8000/ocr/paddle" \
  -F "file=@arabic_text.jpg" \
  -F "lang=ar"
```

### Via Python
```python
import requests

with open("image.jpg", "rb") as f:
    files = {"file": f}
    params = {
        "engine": "paddle",
        "lang": "ch"  # Chinese
    }
    response = requests.post(
        "http://localhost:8000/ocr",
        files=files,
        params=params
    )
    print(response.json())
```

---

## 🎓 Jupyter Notebook

For experimentation and testing, use the provided `ocr_pipeline.ipynb`:

```bash
jupyter notebook ocr_pipeline.ipynb
```

The notebook includes:
- Image loading and preprocessing examples
- Grayscale conversion demonstration
- Test cases with sample images from `data/`

---

## 📊 Performance Notes

- **Tesseract**: Fast, lightweight, best for typed/printed text
- **PaddleOCR**: Slower but more accurate, especially for:
  - Handwritten text
  - Complex layouts
  - Non-Latin scripts (Chinese, Arabic, etc.)
  - Rotated or skewed text

---

## 🤝 Contributing

To extend the pipeline:

1. **Add new preprocessing methods** in `src/preprocessing.py`
2. **Integrate additional OCR engines** following the pattern in `tesseract_pipeline.py`
3. **Add metrics** to `src/evaluation.py` for accuracy assessment
4. **Extend language support** by updating PaddleOCR configuration

---

## 📝 License

This project is open-source. Feel free to use and modify as needed.

---

## 🆘 Troubleshooting

### Tesseract not found
Ensure the Tesseract binary is installed:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### PaddleOCR download errors
PaddleOCR downloads language models on first use. Ensure you have internet access and sufficient disk space (~100 MB per language).

### Image upload size limits
FastAPI has default upload limits. To increase:
```python
# In api.py
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
```

---

## 📧 Support

For issues or questions, refer to:
- [Tesseract Documentation](https://tesseract-ocr.github.io/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
```