"""
experiments/textflux/extract_metadata.py

Implements `extract_metadata(front_page_img)` to pull:
  - issue_date in dd/mm/yyyy
  - issue_number (e.g., "N. 196")

Strategy:
  - Crop the top header band.
  - Detect text lines (simple morphology) and run OCR over the header only.
  - Parse with robust regexes (Italian months + numeric formats).
  - Return best-guess values + diagnostics.

Dependencies (optional but recommended):
  - opencv-python
  - numpy
  - pytesseract (or any OCR you prefer; the code is structured to swap engines)

This module is importable from the project root as:
    from experiments.textflux.extract_metadata import extract_metadata
"""

from __future__ import annotations
import re
import cv2
import numpy as np
from typing import Dict, Any, Optional


MONTHS_IT = {
    "gennaio": "01", "febbraio": "02", "marzo": "03", "aprile": "04",
    "maggio": "05", "giugno": "06", "luglio": "07", "agosto": "08",
    "settembre": "09", "ottobre": "10", "novembre": "11", "dicembre": "12",
    "gen": "01", "feb": "02", "mar": "03", "apr": "04", "mag": "05", "giu": "06",
    "lug": "07", "ago": "08", "set": "09", "ott": "10", "nov": "11", "dic": "12",
}

DATE_PATTERNS = [
    re.compile(r'(?P<d>\b\d{1,2})[\/\-.](?P<m>\d{1,2})[\/\-.](?P<y>\d{2,4})'),
    re.compile(r'(?P<d>\b\d{1,2})\s+(?P<mese>gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\s+(?P<y>\d{4})', re.IGNORECASE),
]

SERIES_PATTERNS = [
    re.compile(r'\bN[º°.]?\s*(?:di\s*serie\s*)?(?P<num>\d{1,4})\b', re.IGNORECASE),
    re.compile(r'\bNumero\s+(?P<num>\d{1,4})\b', re.IGNORECASE),
    re.compile(r'\bN\.\s*(?P<num>\d{1,4})\b', re.IGNORECASE),
]


def _crop_header(img: np.ndarray, ratio: float = 0.35) -> np.ndarray:
    h = img.shape[0]
    cut = max(30, int(h * ratio))
    return img[:cut, :]


def _read_text_lines(image: np.ndarray) -> str:
    """OCR the header strip. Returns a single concatenated string."""
    try:
        import pytesseract
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)
        # Optionally attempt a light adaptive threshold if low contrast
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
        text = pytesseract.image_to_string(thr, lang="ita+eng")
        if not text.strip():
            text = pytesseract.image_to_string(gray, lang="ita+eng")
        return text
    except Exception:
        # Fallback: return empty; caller will handle with defaults
        return ""


def _normalize_date(match: re.Match) -> Optional[str]:
    gd = match.groupdict()
    if "mese" in gd and gd["mese"]:
        d = int(gd["d"])
        m = MONTHS_IT.get(gd["mese"].lower(), None)
        y = int(gd["y"])
        if not m:
            return None
    else:
        d = int(gd["d"])
        m = int(gd["m"])
        y = int(gd["y"])
        if y < 100:
            y += 1900 if y >= 50 else 2000
        m = f"{m:02d}"
    d = f"{d:02d}"
    y = f"{y:04d}"
    return f"{d}/{m}/{y}"


def _parse_date(blob: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = pat.search(blob)
        if m:
            norm = _normalize_date(m)
            if norm:
                return norm
    return None


def _parse_series(blob: str) -> Optional[str]:
    candidates = []
    for pat in SERIES_PATTERNS:
        for m in pat.finditer(blob):
            num = m.group("num")
            try:
                val = int(num)
                if 1 <= val <= 9999:
                    candidates.append(val)
            except Exception:
                continue
    if not candidates:
        return None
    # Heuristic: pick the largest (often the issue number is the larger visible integer)
    best = max(candidates)
    return f"N. {best}"


def extract_metadata(front_page_img: np.ndarray) -> Dict[str, Any]:
    """
    Extract issue_date (dd/mm/yyyy) and issue_number (e.g., "N. 196") from a front page image.

    Args:
        front_page_img: np.ndarray (BGR)

    Returns:
        dict with keys:
            - issue_date: str or ""
            - issue_number: str or ""
            - ocr_header_text: raw OCR for debugging
    """
    header = _crop_header(front_page_img, ratio=0.35)
    blob = _read_text_lines(header)

    issue_date = _parse_date(blob) or ""
    issue_number = _parse_series(blob) or ""

    return {
        "issue_date": issue_date,
        "issue_number": issue_number,
        "ocr_header_text": blob
    }
