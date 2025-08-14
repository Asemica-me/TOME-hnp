"""
experiments/textflux/classify_front_page.py

Implements `classify_front_page(image)` using a TextFlux-inspired glyph-concatenation
verifier PLUS lightweight visual priors (masthead rectangle, "L." price token).

The function returns:
    - is_front (bool): final classification
    - details (dict): diagnostic scores to help you tune thresholds downstream

Dependencies (optional but recommended):
    - torch, torchvision (for TextFlux-style DiT/LoRA head)  [optional]
    - opencv-python
    - numpy
    - pytesseract (only for small top-strip OCR)             [optional]

This module is importable from the project root as:
    from experiments.textflux.classify_front_page import classify_front_page
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json

# Optional imports guarded at runtime
try:
    import pytesseract  # Only for small regex probes in the header strip
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass
class ClassifyConfig:
    # Geometry
    header_ratio: float = 0.32      # top portion considered "header"
    min_banner_aspect: float = 6.0  # wide rectangle aspect (w/h)
    min_banner_rel_width: float = 0.6  # relative to page width
    # Thresholds
    p_textflux_thresh: float = 0.60
    votes_required: int = 2
    # OCR regex for the price token "L. <num>"
    price_regex: str = r"(?i)\bL\.\s?\d+[.,]?\d*\b"
    # Glyph template for concatenation (we render text, but here we only simulate pipeline)
    glyph_text: str = "GIORNALE DELL'EMILIA"


def _crop_header(img: np.ndarray, ratio: float) -> np.ndarray:
    h = img.shape[0]
    cut = max(20, int(h * ratio))
    return img[:cut, :]


def _detect_banner_rectangle(img: np.ndarray, cfg: ClassifyConfig) -> bool:
    """Detect a wide, thin rectangle near the top (masthead banner)."""
    header = _crop_header(img, cfg.header_ratio)
    gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = header.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0: 
            continue
        aspect = w / max(1, h)
        rel_w = w / W
        # Near the very top, reasonably thin & wide, spanning good width
        if y < H * 0.35 and aspect >= cfg.min_banner_aspect and rel_w >= cfg.min_banner_rel_width:
            return True
    return False


def _ocr_price_token(img: np.ndarray, cfg: ClassifyConfig) -> bool:
    """Look for 'L. <number>' token in the header strip using light OCR if available."""
    if not _HAS_TESS:
        return False
    header = _crop_header(img, cfg.header_ratio)
    gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
    # Light contrast boost helps a lot on newsprint
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=10)
    try:
        text = pytesseract.image_to_string(gray, lang="ita+eng")
    except Exception:
        return False
    import re
    return re.search(cfg.price_regex, text) is not None


# ---- TextFlux-style classifier stub ----
class _TextFluxMastheadClassifier:
    """
    A *minimal* placeholder for a TextFlux-style concatenation verifier.
    In a real implementation, you would:
      - build Iglyph (white-on-black) by rendering cfg.glyph_text with PIL
      - horizontally concat with the page image (same height)
      - pass to a small DiT+LoRA head to obtain p(front_page)
    Here we produce a heuristic probability based on normalized template matching
    in the header to keep this script runnable without heavy deps.
    """
    def __init__(self, cfg: ClassifyConfig):
        self.cfg = cfg

    def __call__(self, img: np.ndarray) -> float:
        header = _crop_header(img, self.cfg.header_ratio)
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        # High-pass-ish emphasize text
        hp = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        score = float(np.clip(hp.mean() / 10.0, 0, 1))
        # Weak prior: taller pages with strong edges up top → slightly higher prob
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.mean() / 255.0
        p = np.clip(0.5*score + 0.5*edge_density, 0, 1)
        self.last_scores = {
            "comp_texture": float(score),
            "comp_edge_density": float(edge_density)
        }
        return float(p)

def _save_debug(debug_dir: str, image: np.ndarray, header: np.ndarray, edges: np.ndarray, cands, details: Dict[str, Any], ocr_header_text: str = ""):
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "page.png"), image)
    if header is not None:
        cv2.imwrite(os.path.join(debug_dir, "header.png"), header)
    if edges is not None:
        cv2.imwrite(os.path.join(debug_dir, "header_edges.png"), edges)
    if cands:
        overlay = header.copy()
        for (x, y, w, h) in cands:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "banner_overlay.png"), overlay)
    if ocr_header_text:
        with open(os.path.join(debug_dir, "header_ocr.txt"), "w", encoding="utf-8") as f:
            f.write(ocr_header_text)
    with open(os.path.join(debug_dir, "details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

def classify_front_page(image: np.ndarray, cfg: ClassifyConfig | None = None, debug_dir: str | None = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Classify a page image as (front page vs not).

    Args:
        image: np.ndarray (BGR, as loaded by cv2.imread)
        cfg: ClassifyConfig (optional)

    Returns:
        is_front: bool
        details: dict with keys:
            - p_textflux: float
            - masthead_rect: bool
            - price_found: bool
            - votes: int
    """
    if cfg is None:
        cfg = ClassifyConfig(
            header_ratio=0.3,          # Adjust header crop height
            p_textflux_thresh=0.65,    # Increase confidence requirement
            votes_required=2           # Require 2/3 indicators
        )

    # 1) TextFlux-style verifier (placeholder impl; swap with your DiT/LoRA scorer)
    tf_model = _TextFluxMastheadClassifier(cfg)
    components = getattr(tf_model, "last_scores", {})
    p_textflux = tf_model(image)

    # 2) Visual priors
    masthead_rect = _detect_banner_rectangle(image, cfg)
    price_found = _ocr_price_token(image, cfg)

    # 3) Voting rule
    votes = int(p_textflux >= cfg.p_textflux_thresh) + int(masthead_rect) + int(price_found)
    is_front = votes >= cfg.votes_required

    details = {
    "p_textflux": p_textflux,
    "masthead_rect": bool(masthead_rect),
    "price_found": bool(price_found),
    "votes": votes,
    "components": {
        "comp_texture": round(p_textflux, 3),  # Just a placeholder if not split
        "comp_edge_density": round(p_textflux, 3)  # Same here; split if needed
    },
    "banner_candidates": cands if debug_dir is not None else []
}
    if debug_dir is not None:
        header = _crop_header(image, cfg.header_ratio)
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        W = header.shape[1]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 0:
                aspect = w / h
                if y < header.shape[0] * 0.35 and aspect >= cfg.min_banner_aspect and (w / W) >= cfg.min_banner_rel_width:
                    cands.append((x, y, w, h))
        ocr_header_text = ""
        if _HAS_TESS:
            try:
                ocr_header_text = pytesseract.image_to_string(gray, lang="ita+eng")
            except Exception:
                pass
        _save_debug(debug_dir, image, header, edges, cands, details, ocr_header_text)
    return is_front, details

def debug_batch_classification(imgs_dir: str, output_dir: str):
    """
    For every image in imgs_dir, runs classify_front_page with debugging enabled.
    Saves visual + textual artifacts into output_dir/debug/<filename>/ and a CSV summary.

    Example:
        debug_batch_classification('./imgs', './output')
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    debug_root = os.path.join(output_dir, "debug")
    os.makedirs(debug_root, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    rows = [["filename", "p_textflux", "comp_texture", "comp_edge_density",
             "masthead_rect", "price_found", "votes", "banner_candidates_count"]]

    for fname in sorted(os.listdir(imgs_dir)):
        if not os.path.splitext(fname)[1].lower() in exts:
            continue
        fpath = os.path.join(imgs_dir, fname)
        image = cv2.imread(fpath)
        if image is None:
            print(f"[WARN] Could not read {fpath}")
            continue

        out_dir = os.path.join(debug_root, os.path.splitext(fname)[0])
        is_front, details = classify_front_page(image, debug_dir=out_dir)

        p = details.get("p_textflux", 0.0)
        comps = details.get("components", {})
        rows.append([
            fname,
            f"{p:.3f}",
            f"{comps.get('comp_texture', 0.0):.3f}",
            f"{comps.get('comp_edge_density', 0.0):.3f}",
            int(details.get("masthead_rect", False)),
            int(details.get("price_found", False)),
            int(details.get("votes", 0)),
            len(details.get("banner_candidates", []))
        ])

    csv_path = os.path.join(output_dir, "debug_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[OK] Debug artifacts → {debug_root}")
    print(f"[OK] Summary CSV → {csv_path}")

# For single image classification
def classify_single_image(image_path: str, debug_dir: str = None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    is_front, details = classify_front_page(image, debug_dir=debug_dir)
    print(f"Classification Result: {'FRONT PAGE' if is_front else 'NOT FRONT PAGE'}")
    print("Details:", json.dumps(details, indent=2))

# For batch processing
def classify_batch(input_dir: str, output_dir: str):
    debug_batch_classification(input_dir, output_dir)
    print(f"Batch results saved to: {output_dir}")

# CONFIGURATION - SET YOUR PATHS HERE
IMAGE_PATH = "path/to/your/image.jpg"
DEBUG_DIR = "path/to/debug/output"  # Set to None to disable debugging

# Load image and classify
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

is_front, details = classify_front_page(image, debug_dir=DEBUG_DIR)

# Print results
print("Classification Result:", "FRONT PAGE" if is_front else "NOT FRONT PAGE")
print("Details:")
for k, v in details.items():
    if k == "components":
        print("  Components:")
        for comp, score in v.items():
            print(f"    {comp}: {score:.3f}")
    else:
        print(f"  {k}: {v}")

# CONFIGURATION - SET YOUR PATHS HERE
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMGS_DIR     = os.path.join(PROJECT_ROOT, "imgs")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process all images in directory
debug_batch_classification(IMGS_DIR, OUTPUT_DIR)
print(f"Batch processing complete! Results saved to {OUTPUT_DIR}")