
```
ocr_pipeline/
├── src/
│   ├── preprocessing.py  # OpenCV-based preprocessing
│   ├── ocr_engines.py    # Tesseract + PaddleOCR wrappers
│   ├── evaluation.py     # CER/WER calculation
│   └── main.py           # Pipeline orchestration
├── data/
│   ├── test_images/      # Clean, noisy, rotated samples
│   └── ground_truth/     # Manual annotations for evaluation
├── requirements.txt
└── README.md             # Test instructions + results
```