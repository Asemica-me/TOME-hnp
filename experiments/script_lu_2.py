# -*- coding: utf-8 -*-
"""
Front-page classification + metadata extraction for historical newspapers
Proof of concept focused on: GIORNALE DELL'EMILIA
- Strict masthead detection (contours + size gating)
- OCR tuned for masthead (uppercase, single-line)
- Date + issue extraction from top-right strip with multi-pass OCR
"""

import os
import glob
import logging
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image
import pytesseract
import unicodedata
import cv2
import numpy as np
from rapidfuzz import fuzz
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(LOGS_DIR, "script_lu_2.log"), encoding='utf-8')]
)

IMAGE_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'imgs'))
CATALOG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'catalog', 'catalog_cleaned.xml'))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "script_lu_2.json")

START_DATE = os.getenv('START_DATE', '1946-07-21')   # yyyy-mm-dd
END_DATE   = os.getenv('END_DATE',   '1947-05-15')
NEWSPAPER_TITLE = os.getenv('NEWSPAPER_TITLE', "Giornale dell'Emilia")

# Tesseract configuration (env override -> common defaults)
TESS_CMD = os.getenv('TESSERACT_CMD',
                     r'C:\Program Files\Tesseract-OCR\tesseract.exe' if os.name == 'nt' else '/usr/bin/tesseract')
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD
    logging.info(f"Tesseract: {TESS_CMD}")
else:
    logging.warning(f"Tesseract not found at {TESS_CMD}. If installed elsewhere, set TESSERACT_CMD env variable.")

# -----------------------------------------------------------------------------
# Helpers: normalization & OCR
# -----------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_title(s: str) -> str:
    s = strip_accents(s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def configure_ocr():
    # nothing special here; placeholder if you later add user-words or configs
    pass

def ocr_lines_with_boxes(pil_img, psm=6):
    """Line-level OCR with coordinates (Tesseract)."""
    cfg = f'--psm {psm} --oem 3 -l ita'
    d = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
    lines = {}
    for i, txt in enumerate(d['text']):
        if not txt or not txt.strip():
            continue
        key = (d['page_num'][i], d['block_num'][i], d['par_num'][i], d['line_num'][i])
        lines.setdefault(key, []).append({
            'text': txt, 'conf': float(d['conf'][i]) if d['conf'][i] != '-1' else 0.0,
            'x': d['left'][i], 'y': d['top'][i], 'w': d['width'][i], 'h': d['height'][i],
        })
    merged = []
    for toks in lines.values():
        x = min(t['x'] for t in toks); y = min(t['y'] for t in toks)
        w = max(t['x'] + t['w'] for t in toks) - x
        h = max(t['y'] + t['h'] for t in toks) - y
        merged.append({'text': " ".join(t['text'] for t in toks), 'x': x, 'y': y, 'w': w, 'h': h,
                       'conf': float(np.mean([t['conf'] for t in toks]))})
    return merged

# -----------------------------------------------------------------------------
# UNIMARC loader
# -----------------------------------------------------------------------------
def load_headtitles(catalog_path):
    """Load title headings from UNIMARC 200$a."""
    headtitles = set()
    try:
        tree = ET.parse(catalog_path)
        root = tree.getroot()
        ns = {'marc': 'http://www.loc.gov/MARC21/slim'}
        for record in root.findall('marc:record', ns):
            for datafield in record.findall("marc:datafield[@tag='200']", ns):
                a = datafield.find("marc:subfield[@code='a']", ns)
                if a is not None and a.text:
                    headtitles.add(normalize_title(a.text))
        logging.info(f"Loaded {len(headtitles)} titles from catalog")
    except Exception as e:
        logging.error(f"Error reading catalog: {e}")
    return list(headtitles)

# -----------------------------------------------------------------------------
# Geometry cues
# -----------------------------------------------------------------------------
def _has_long_top_lines(cv_band):
    g = cv2.cvtColor(cv_band, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=int(cv_band.shape[1]*0.6), maxLineGap=15)
    return lines is not None

def _has_large_text(lines, band_h):
    if not lines:
        return False
    h95 = np.percentile([ln['h'] for ln in lines], 95) if len(lines) >= 5 else max(ln['h'] for ln in lines)
    return (h95 / max(1, band_h)) >= 0.06

# (optional) small deskew for band — helps on slight rotations
def deskew_band(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None:
        return gray
    angles = []
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        ang = (theta - np.pi/2) * 180/np.pi
        if -5 <= ang <= 5:
            angles.append(ang)
    if not angles:
        return gray
    angle = float(np.median(angles))
    H, W = gray.shape[:2]
    M = cv2.getRotationMatrix2D((W//2, H//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# -----------------------------------------------------------------------------
# Masthead ROI detection
# -----------------------------------------------------------------------------
def masthead_roi_from_band(band_gray_np: np.ndarray):
    """
    Find wide, tall masthead in the top band. Return (x1,y1,x2,y2) or None.
    Tuned for 'GIORNALE dell'EMILIA' style; excludes tiny 'Cronaca di Bologna' cases.
    """
    H, W = band_gray_np.shape[:2]
    blur = cv2.GaussianBlur(band_gray_np, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)), 1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1.0, h)
        if (w >= 0.55 * W) and (h >= 0.28 * H) and (y < 0.55 * H) and (aspect >= 3.0):
            cand.append((x, y, w, h))

    if not cand:
        return None

    x1 = min(x for x, y, w, h in cand)
    y1 = min(y for x, y, w, h in cand)
    x2 = max(x + w for x, y, w, h in cand)
    y2 = max(y + h for x, y, w, h in cand)

    pad_x = int(0.02 * W); pad_y = int(0.03 * H)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)
    return (x1, y1, x2, y2)

# -----------------------------------------------------------------------------
# Front page classifier (strict)
# -----------------------------------------------------------------------------
def is_front_page(image_path, headtitles):
    """
    Accept only if:
      - masthead ROI is LARGE (size gating) AND
      - OCR of ROI fuzzy-matches target title; or
      - fallback finds a match on a LARGE line (small text suppressed)
    """
    target = headtitles[0]  # "giornale dell emilia"

    img = Image.open(image_path)
    W, H = img.size
    band_h = int(H * 0.30)
    band = img.crop((0, 0, W, band_h)).convert('L')

    # optional deskew
    band_np = np.array(band)
    band_np = deskew_band(band_np)
    band = Image.fromarray(band_np)

    title_score = 0
    price_cue = False

    # --- A) Masthead-targeted OCR on ROI ---
    roi_box = masthead_roi_from_band(band_np)
    size_ok = False
    if roi_box:
        x1, y1, x2, y2 = roi_box
        roi = band.crop((x1, y1, x2, y2))
        roi = roi.point(lambda x: 0 if x < 150 else 255)
        roi = roi.resize((roi.width * 3, roi.height * 3), Image.LANCZOS)

        mast_txt = pytesseract.image_to_string(
            roi,
            config="--psm 7 --oem 1 -l ita -c preserve_interword_spaces=1 "
                   "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
        )
        mast_norm = normalize_title(mast_txt)
        title_score = max(title_score, fuzz.token_set_ratio(target, mast_norm))

        roi_w, roi_h = (x2 - x1), (y2 - y1)
        size_ok = (roi_w >= 0.60 * W) and (roi_h >= 0.32 * band_h)

    # --- B) Fallback OCR on full band (with small-text suppression) ---
    if title_score < 75 or not size_ok:
        band2 = band.point(lambda x: 0 if x < 160 else 255)
        band2 = band2.resize((W, band_h * 2), Image.LANCZOS)
        lines = ocr_lines_with_boxes(band2, psm=6)
        large_line_min = max(18, int(0.20 * band_h))
        for l in lines:
            if l['h'] >= large_line_min:
                norm = normalize_title(l['text'])
                title_score = max(title_score, fuzz.token_set_ratio(target, norm))
            if re.search(r"\bL\.\s*\d+\b", l['text']):
                price_cue = True

    # --- Geometry cue ---
    cv_band = cv2.cvtColor(band_np, cv2.COLOR_GRAY2BGR)
    geom_cue = _has_long_top_lines(cv_band) and _has_large_text(
        ocr_lines_with_boxes(band, psm=6), band_h
    )

    # --- Decision ---
    if roi_box:
        decision = (title_score >= 75 and size_ok) or (title_score >= 85)
    else:
        decision = (title_score >= 80 and (geom_cue or price_cue))

    score = 0.65 * (title_score / 100.0) + 0.20 * (1 if geom_cue else 0) + 0.15 * (1 if price_cue else 0)
    logging.info(
        f"[{os.path.basename(image_path)}] title={title_score} size_ok={size_ok} "
        f"geom={geom_cue} price={price_cue} -> front_score={score:.2f}"
    )
    return bool(decision)

def classify_page(image_path, headtitles):
    try:
        if is_front_page(image_path, headtitles):
            return "Front Page", 0.95
        return "Internal Page", 0.60
    except Exception as e:
        logging.error(f"Classification failed: {e}")
        return "Classification Error", 0.0

# -----------------------------------------------------------------------------
# Date & Issue extraction (top-right priority + multi-pass OCR)
# -----------------------------------------------------------------------------
MONTHS = {
    'gennaio':'01','febbraio':'02','marzo':'03','aprile':'04','maggio':'05','giugno':'06',
    'luglio':'07','agosto':'08','settembre':'09','ottobre':'10','novembre':'11','dicembre':'12'
}
WEEKDAYS = r'(luned[iì]|marted[iì]|mercoled[iì]|gioved[iì]|venerd[iì]|sabato|domenica)'
DATE_RX = re.compile(rf'(?:{WEEKDAYS}\s*[-–—]?\s*)?(\d{{1,2}})\s+(' + '|'.join(MONTHS.keys()) + r')\s+(\d{4})', re.IGNORECASE)
NUM_RX  = re.compile(r'\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b')
ISSUE_RX = re.compile(r'\b[Nn]\.?\s*(\d{1,4})\b')

def _ddmmyyyy(day, mon, year):
    mm = MONTHS.get(normalize_title(mon), f"{int(mon):02d}")
    return f"{int(day):02d}/{mm}/{year}"

def _ocr_roi_try(pil_img, psm, upscale, hard_thresh=False):
    """Preprocess + OCR a ROI and return text."""
    img = pil_img.convert('L')
    if hard_thresh:
        img = img.point(lambda x: 0 if x < 170 else 255)
    img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    cfg = f'--psm {psm} --oem 3 -l ita -c preserve_interword_spaces=1'
    return pytesseract.image_to_string(img, config=cfg)

def extract_metadata_front(image_path):
    img = Image.open(image_path)
    W, H = img.size
    rois = [
        (int(W*0.60), 0, W, int(H*0.20)),        # top-right (highest priority)
        (0, 0, W, int(H*0.18)),                  # full top strip
        (int(W*0.40), 0, int(W*0.75), int(H*0.16)),  # top-center-right
        (int(W*0.60), int(H*0.20), W, int(H*0.30))   # secondary band (20–30%)
    ]
    best = {'date': None, 'date_conf': 0.0, 'issue': None, 'issue_conf': 0.0}

    for idx, (x1, y1, x2, y2) in enumerate(rois):
        band = img.crop((x1, y1, x2, y2))
        # Two-pass OCR: soft (psm7), then hard threshold if needed; upscale ×3 for small print
        texts = [
            _ocr_roi_try(band, psm=7, upscale=3, hard_thresh=False),
            _ocr_roi_try(band, psm=7, upscale=3, hard_thresh=True),
            _ocr_roi_try(band, psm=6, upscale=3, hard_thresh=False),
        ]
        text = " ".join(t for t in texts if t)

        bonus = 0.2 if idx == 0 else 0.0
        # date (month words)
        m = DATE_RX.search(text)
        if m:
            d, mon, y = m.group(1), m.group(2), m.group(3)
            cand = _ddmmyyyy(d, mon, y)
            score = min(1.0, 0.7 + bonus)
            if score > best['date_conf']:
                best['date'], best['date_conf'] = cand, score
        else:
            n = NUM_RX.search(text)
            if n:
                d, mon, y = n.groups()
                cand = _ddmmyyyy(d, mon, y)
                score = min(1.0, 0.65 + bonus)
                if score > best['date_conf']:
                    best['date'], best['date_conf'] = cand, score

        k = ISSUE_RX.search(text)
        if k:
            issue = k.group(1)
            score = min(1.0, 0.7 + bonus)
            if score > best['issue_conf']:
                best['issue'], best['issue_conf'] = issue, score

        # short-circuit if both are strong
        if best['date_conf'] >= 0.9 and best['issue_conf'] >= 0.9:
            break

    return best

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def process_newspaper_images():
    """Main processing loop with fallback headtitles and PoC target filter."""
    # Load existing results (dedupe by filename)
    processed_data = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                for entry in json.load(f):
                    processed_data[entry["filename"]] = entry
            logging.info(f"Loaded {len(processed_data)} existing records")
        except Exception as e:
            logging.warning(f"Failed to load existing JSON: {e}")

    # Headtitles from catalog (keep fallback if empty)
    headtitles = load_headtitles(CATALOG_PATH) if os.path.exists(CATALOG_PATH) else []
    if not headtitles:
        headtitles = [
            normalize_title("Il Resto del Carlino"),
            normalize_title("Giornale dell'Emilia"),
            normalize_title("Giornale dell'Emilia Riunite"),
            normalize_title("Il Piccolo Faust"),
            normalize_title("Periodico fantastico artistico teatrale"),
        ]
        logging.info(f"Using fallback headtitles: {headtitles}")

    # PoC: only Giornale dell'Emilia
    TARGET_CANONICAL = normalize_title("Giornale dell'Emilia")  # -> "giornale dell emilia"
    headtitles = [t for t in headtitles if t == TARGET_CANONICAL] or [TARGET_CANONICAL]
    logging.info(f"Using PoC target(s): {headtitles}")

    configure_ocr()

    # Collect images
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    image_paths.sort()
    if not image_paths:
        logging.error(f"No images found in: {IMAGE_FOLDER}")
        return
    logging.info(f"Found {len(image_paths)} images to process")

    # Date range and chronology guard
    min_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    max_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    last_valid_date = None

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        page_num = idx + 1

        logging.info("\n" + "="*40)
        logging.info(f"Processing page {page_num}/{len(image_paths)}: {filename}")
        logging.info("="*40)

        result = {
            "filename": filename,
            "page_number": page_num,
            "classification": "Unknown",
            "classification_confidence": 0.0,
            "issue_date": "N/A",  # dd/mm/yyyy
            "issue_number": "N/A",
            "hasGraphics": False,
            "processing_errors": []
        }

        try:
            classification, confidence = classify_page(img_path, headtitles)
            result["classification"] = classification
            result["classification_confidence"] = float(confidence)

            if classification == "Front Page":
                meta = extract_metadata_front(img_path)

                # Date
                if meta.get('date'):
                    try:
                        d_obj = datetime.strptime(meta['date'], "%d/%m/%Y")
                        if min_date <= d_obj <= max_date:
                            if last_valid_date is None or d_obj >= last_valid_date:
                                result["issue_date"] = meta['date']
                                last_valid_date = d_obj
                            else:
                                result["processing_errors"].append(
                                    f"Date {meta['date']} earlier than previous front-page date"
                                )
                        else:
                            result["processing_errors"].append(
                                f"Date {meta['date']} outside range {min_date:%d/%m/%Y}-{max_date:%d/%m/%Y}"
                            )
                    except Exception as e:
                        result["processing_errors"].append(f"Failed to parse extracted date '{meta['date']}': {e}")
                else:
                    result["processing_errors"].append("No date extracted")

                # Issue number
                if meta.get('issue'):
                    result["issue_number"] = meta['issue']
                else:
                    result["processing_errors"].append("No issue number extracted")

        except Exception as e:
            logging.error(f"Processing error on {filename}: {e}", exc_info=True)
            result["processing_errors"].append(str(e))

        processed_data[filename] = result

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(list(processed_data.values()), f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to: {OUTPUT_JSON}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    process_newspaper_images()
