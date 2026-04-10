from __future__ import annotations

from typing import Any

import numpy as np

_paddle_ocr_by_lang: dict[str, Any] = {}


def _normalize_lang(lang: str) -> str:
    s = (lang or "en").strip().lower()
    return s if s else "en"


def _get_paddle_ocr(lang: str = "en"):
    lang = _normalize_lang(lang)
    if lang not in _paddle_ocr_by_lang:
        from paddleocr import PaddleOCR

        _paddle_ocr_by_lang[lang] = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
        )
    return _paddle_ocr_by_lang[lang]


def _text_from_paddle_result(result: Any) -> str:
    if result is None:
        return ""
    lines: list[str] = []

    def add_from_mapping(data: dict) -> bool:
        for key in ("rec_texts", "texts", "text", "res_text"):
            val = data.get(key)
            if isinstance(val, list):
                lines.extend(str(t) for t in val if str(t).strip())
                return True
            if isinstance(val, str) and val.strip():
                lines.append(val)
                return True
        inner = data.get("res")
        if isinstance(inner, dict):
            return add_from_mapping(inner)
        return False

    for item in result:
        if item is None:
            continue

        data = None
        if isinstance(item, dict):
            data = item
        elif hasattr(item, "json"):
            j = getattr(item, "json")
            if callable(j):
                try:
                    data = j()
                except TypeError:
                    data = j
            else:
                data = j

        if isinstance(data, dict) and add_from_mapping(data):
            continue

        if not isinstance(item, (str, bytes)) and hasattr(item, "__getitem__"):
            try:
                rt = item["rec_texts"]
                if isinstance(rt, (list, tuple)):
                    lines.extend(str(t) for t in rt if str(t).strip())
                    continue
            except (KeyError, TypeError):
                pass

        if isinstance(item, (list, tuple)):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    seg = line[1]
                    if isinstance(seg, (list, tuple)) and seg:
                        lines.append(str(seg[0]))
                    elif isinstance(seg, str) and seg.strip():
                        lines.append(seg)

    return "\n".join(lines)


def run_paddleocr_ocr_pipeline(
    image: np.ndarray,
    config: str = "",
    lang: str = "en",
) -> str:
    """
    Run PaddleOCR on a BGR image (OpenCV decode).
    `lang` is passed to PaddleOCR (e.g. en, ar, ch). `config` is reserved.
    """
    if image is None:
        raise ValueError("Input image is None.")
    _ = config
    ocr = _get_paddle_ocr(lang)
    result = ocr.predict(image)
    return _text_from_paddle_result(result)
