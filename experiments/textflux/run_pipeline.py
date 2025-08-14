"""
experiments/textflux/run_pipeline.py

Batch driver that:
  - walks the ./imgs directory (from project root)
  - classifies each image with classify_front_page(image)
  - if front page, extracts metadata with extract_metadata(front_page_img)
  - writes a JSON list to ./output/results.json

Run:
    python run_pipeline.py
"""

from __future__ import annotations
import os, json, re
from typing import List, Dict, Any
import cv2

from classify_front_page import classify_front_page, ClassifyConfig
from extract_metadata import extract_metadata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMGS_DIR     = os.path.join(PROJECT_ROOT, "imgs")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, "textflux.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _infer_page_number(filename: str) -> int:
    """
    Best-effort page number inference from filenames like ..._0002.jpg or ...-p2.png.
    Returns -1 if not found.
    """
    base = os.path.basename(filename)
    # common patterns: _0002, -0002, p2, page2
    m = re.search(r'(?:_|-)(\d{4})\b', base)
    if m:
        try:
            return int(m.group(1).lstrip("0") or "0")
        except Exception:
            pass
    m = re.search(r'\bp(?:age)?\s?(\d{1,3})\b', base, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return -1


def _iter_images(img_dir: str):
    for root, _, files in os.walk(img_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                yield os.path.join(root, f)


def main():
    results: List[Dict[str, Any]] = []
    cfg = ClassifyConfig()

    for path in _iter_images(IMGS_DIR):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        is_front, details = classify_front_page(img, cfg=cfg)

        classification = "front_page" if is_front else "not_front_page"
        meta = {"issue_date": "", "issue_number": ""}
        if is_front:
            m = extract_metadata(img)
            meta["issue_date"] = m.get("issue_date", "")
            meta["issue_number"] = m.get("issue_number", "")

        rec = {
            "filename": os.path.basename(path),
            "page_number": _infer_page_number(path),
            "classification": classification,
            "issue_date": meta["issue_date"],
            "issue_number": meta["issue_number"],
            # Diagnostics (optional; remove if you want a minimal schema)
            "_diagnostics": {
                "p_textflux": details.get("p_textflux"),
                "masthead_rect": details.get("masthead_rect"),
                "price_found": details.get("price_found"),
                "votes": details.get("votes")
            }
        }
        results.append(rec)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(results)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
