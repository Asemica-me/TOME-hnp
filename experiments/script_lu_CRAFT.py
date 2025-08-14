import os
import glob
import logging
from pathlib import Path
from typing import List, Tuple
import json
import re
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps
import unicodedata
import pytesseract
from sentence_transformers import SentenceTransformer, util
import torch
from hezar.models import Model
from PIL import ImageDraw
import difflib
from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
logs_folder = os.path.join(ROOT_DIR, 'logs')
os.makedirs(logs_folder, exist_ok=True)
log_file_path = os.path.join(logs_folder, "script_lu_CRAFT.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path)
    ]
)

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")      
DEBUG_DIR = os.path.join(ARTIFACTS_DIR, "debug")  
MASKED_DIR = os.path.join(ARTIFACTS_DIR, "masked")    # masked header crops
OVERLAY_DIR = os.path.join(ARTIFACTS_DIR, "overlays")  # drawn polygons/overlays
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
output_json = os.path.join(OUTPUT_DIR, "script_lu_CRAFT.json")

for d in [ARTIFACTS_DIR, DEBUG_DIR, MASKED_DIR, OVERLAY_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

IMAGE_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'imgs'))
CATALOG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'catalog', 'catalog_wrapped.xml'))
START_DATE = os.getenv('START_DATE', '1946-07-21')
END_DATE = os.getenv('END_DATE', '1947-05-15')
NEWSPAPER_TITLE = os.getenv('NEWSPAPER_TITLE', 'Giornale dell Emilia')

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_PATH = pytesseract.pytesseract.tesseract_cmd
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    logging.error(f"Tesseract not found at {TESSERACT_PATH}. Please install it or update the path.")
    exit(1)

CRAFT_MODEL = None

def load_craft_model():
    global CRAFT_MODEL
    if CRAFT_MODEL is None:
        try:
            CRAFT_MODEL = Model.load("hezarai/CRAFT")
            logging.info("CRAFT text detection model loaded")
        except Exception as e:
            logging.error(f"Failed to load CRAFT model: {str(e)}")
    return CRAFT_MODEL

# --- XML Catalog Processing ---
def load_headtitles(catalog_path):
    """
    Extract all historical headtitles from UNIMARC catalog.
    Supports both MARC21 namespaced and non-namespaced XML.
    Returns a list of normalized headtitles.
    """
    headtitles = set()
    try:
        tree = ET.parse(catalog_path)
        root = tree.getroot()
        
        # Detect if MARC21 namespace is used
        if root.tag.startswith("{http://www.loc.gov/MARC21/slim}"):
            ns = {'marc': 'http://www.loc.gov/MARC21/slim'}
            records = root.findall('marc:record', ns)
            tag_prefix = "marc"
        else:
            ns = {}  # No namespace
            records = root.findall('record')
            tag_prefix = ""

        for record in records:
            for tag in [182, 200]:
                tag_path = f"{tag_prefix}:datafield[@tag='{tag}']" if tag_prefix else f"datafield[@tag='{tag}']"
                for datafield in record.findall(tag_path, ns):
                    for subfield in datafield.findall(f"{tag_prefix}:subfield[@code='a' or @code='e']" if tag_prefix else "subfield[@code='a' or @code='e']", ns):
                        if subfield.text:
                            title = subfield.text.strip()
                            normalized = re.sub(r'[^\w\s]', '', title).lower()
                            headtitles.add(normalized)

        logging.info(f"Loaded {len(headtitles)} historical headtitles from catalog")
        return list(headtitles)

    except Exception as e:
        logging.error(f"Error processing catalog: {str(e)}")
        return []

# --- OCR Configuration ---
def configure_ocr():
    """Configure Tesseract parameters"""
    # Update this path to your Tesseract installation
    tesseract_path = TESSERACT_PATH
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logging.info("Tesseract configured successfully")
    else:
        logging.warning(f"Tesseract path not found: {tesseract_path}")

def _normalize_craft(results):
    """
    Normalize various CRAFT outputs into (polygons, scores).
    Supports Hezar's TextDetectionOutput, dicts, and lists.
    """
    import numpy as np
    import torch
    from collections.abc import Mapping

    def _to_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        return None

    def _box_to_poly(box):
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def _extract_from_item(item):
        """
        Accepts: Hezar TextDetectionOutput, dict-like, or any object that exposes
        boxes/polygons/scores as attributes or via __getitem__.
        Returns (polys, scores).
        """
        polys, scs = [], []

        # Try attribute access first
        boxes    = getattr(item, "boxes",    None)
        polygons = getattr(item, "polygons", None)
        scores   = getattr(item, "scores",   None)

        # Fallback to dict-style access
        if boxes is None:
            try: boxes = item["boxes"]
            except Exception: pass
        if polygons is None:
            try: polygons = item["polygons"]
            except Exception: pass
        if scores is None:
            try: scores = item["scores"]
            except Exception: pass

        # Some variants expose 'bboxes' / 'points'
        if boxes is None:
            boxes = getattr(item, "bboxes", None)
        if polygons is None:
            polygons = getattr(item, "points", None)

        # Normalize to lists
        boxes    = _to_list(boxes)    if boxes    is not None else None
        polygons = _to_list(polygons) if polygons is not None else None
        scores   = _to_list(scores)   if scores   is not None else None

        if boxes:
            for b in boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    polys.append(_box_to_poly(b))
        elif polygons:
            for poly in polygons:
                if isinstance(poly, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly):
                    polys.append([(float(p[0]), float(p[1])) for p in poly])

        if scores:
            scs = scores

        return polys, scs

    polygons, scores = [], []

    try:
        # Hezar returns a list; pick first element if so
        if isinstance(results, (list, tuple)) and results:
            first = results[0]
            # If it's a dict-like or TextDetectionOutput with __getitem__
            if isinstance(first, Mapping) or hasattr(first, "__getitem__"):
                polygons, scores = _extract_from_item(first)
            else:
                # list of boxes or list of dicts fallback
                first_list = _to_list(first)
                if first_list and isinstance(first_list[0], (list, tuple)) and len(first_list[0]) == 4:
                    for b in first_list:
                        polygons.append(_box_to_poly(b))
                elif first_list and isinstance(first_list[0], dict):
                    for det in first_list:
                        if "box" in det and len(det["box"]) == 4:
                            polygons.append(_box_to_poly(det["box"]))
                            scores.append(float(det.get("score", 1.0)))
                        elif "polygon" in det:
                            poly = det["polygon"]
                            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly):
                                polygons.append([(float(p[0]), float(p[1])) for p in poly])
                                scores.append(float(det.get("score", 1.0)))

        elif isinstance(results, dict):
            polygons, scores = _extract_from_item(results)

        if not scores or len(scores) != len(polygons):
            scores = [1.0] * len(polygons)

    except Exception as e:
        logging.warning(f"Failed to normalize CRAFT output: {e}")
        polygons, scores = [], []

    return polygons, scores

# --- Front Page Verification ---
def is_valid_front_page(image_path, headtitles, min_ratio=0.4):
    """ Headtitle verification using CRAFT (full-page + header fallback) and OCR with text-region masking. """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # ---- Tunables (envs with safe defaults)
        header_ratio = float(os.getenv('HEADER_RATIO'))            # % of page height
        min_conf = float(os.getenv('CRAFT_MIN_CONFIDENCE'))         # CRAFT score cutoff
        use_craft = os.getenv('USE_CRAFT', 'true').lower() == 'true'
        header_upscale = int(os.getenv('HEADER_UPSCALE', '3'))             # fallback header upscaling factor
        craft_max_side = int(os.getenv('CRAFT_MAX_SIDE', '1600'))          # full-page resize cap before CRAFT

        # ---- Header crop (we'll OCR this region; date is inside)
        header_h = int(height * header_ratio)
        region = (0, 0, width, header_h)
        cropped = img.crop(region)
        original_cropped = cropped.copy()  # fallback
        masked = original_cropped.copy()
        unmasked = original_cropped.copy()

        # ---- Helpers (local)
        def _resize_for_craft(im: Image.Image, max_side: int = 1600) -> Tuple[Image.Image, float]:
            w, h = im.size
            m = max(w, h)
            if m <= max_side:
                return im, 1.0
            scale = max_side / float(m)
            return im.resize((int(w * scale), int(h * scale)), Image.LANCZOS), scale

        def _poly_bbox(poly):
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            return min(xs), min(ys), max(xs), max(ys)

        def _iou_1d(a0, a1, b0, b1):
            inter = max(0, min(a1, b1) - max(a0, b0))
            denom = (a1 - a0) + (b1 - b0) - inter
            return inter / denom if denom > 0 else 0.0

        def _overlap_ratio_with_header(poly, H, header_ratio):
            _, y0, _, y1 = _poly_bbox(poly)
            header_y0, header_y1 = 0, H * header_ratio
            inter = max(0, min(y1, header_y1) - max(y0, header_y0))
            box_h = max(1.0, y1 - y0)
            return inter / box_h

        def _filter_boxes_to_header(polys, scs, img_size, header_ratio, min_conf, min_overlap=0.2):
            W, H = img_size
            keep_p, keep_s = [], []
            for poly, sc in zip(polys, scs):
                if sc < min_conf or not poly:
                    continue
                overlap = _overlap_ratio_with_header(poly, H, header_ratio)
                if overlap >= min_overlap:
                    keep_p.append(poly); keep_s.append(sc)
            return keep_p, keep_s

        def _to_header_coords(polys_page: List[List[Tuple[float, float]]], header_h: int):
            """Map page coords into header-crop coords (x same, y relative to top=0), then clip to header box."""
            mapped = []
            for poly in polys_page:
                mp = []
                for x, y in poly:
                    if y < 0 or y > header_h:
                        # still keep, but clamp so polygon draws safely within mask
                        y = max(0, min(header_h, y))
                    mp.append((x, y))
                mapped.append(mp)
            return mapped

        # ---- CRAFT detection (full page first, header fallback second)
        polygons_hdr = []
        scores_hdr = []

        craft_model = load_craft_model() if use_craft else None
        if craft_model:
            try:
                # 1) FULL PAGE : filter to header
                full_rgb = img.convert('RGB')
                full_resized, scale = _resize_for_craft(full_rgb, max_side=craft_max_side)
                results_full = craft_model.predict(full_resized)
                logging.info(f"CRAFT raw keys: {list(results_full.keys()) if isinstance(results_full, dict) else type(results_full)}")
                # If it's a list, peek at the first element for structure
                if isinstance(results_full, list) and results_full:
                    logging.info(f"CRAFT list[0] type/keys: {type(results_full[0])} / {list(results_full[0].keys()) if isinstance(results_full[0], dict) else 'n/a'}")
                polys_full, scores_full = _normalize_craft(results_full)
                logging.info(f"CRAFT(full) raw polygons: {len(polys_full)}")
                logging.info(f"CRAFT(normalized) polygons : {len(polys_full)}  - scores: {len(scores_full)}")

                # rescale back to original coords
                if scale != 1.0 and polys_full:
                    inv = 1.0 / scale
                    polys_full = [[(p[0]*inv, p[1]*inv) for p in poly] for poly in polys_full]

                # keep only header polys
                polys_head, scores_head = _filter_boxes_to_header(
                    polys_full, scores_full, img.size, header_ratio=header_ratio, min_conf=min_conf
                )
                logging.info(f"CRAFT(full) : {len(polys_head)} header polygons")

                # 2) FALLBACK: if none, run on upscaled header crop
                if not polys_head:
                    hdr_rgb = cropped.convert('RGB')
                    up_w, up_h = hdr_rgb.width * header_upscale, hdr_rgb.height * header_upscale
                    hdr_big = hdr_rgb.resize((up_w, up_h), Image.LANCZOS)
                    results_hdr = craft_model.predict(hdr_big)
                    polys_big, scores_big = _normalize_craft(results_hdr)
                    # map down to header-crop coords
                    polygons_hdr = [[(p[0]/header_upscale, p[1]/header_upscale) for p in poly] for poly in polys_big]
                    scores_hdr = scores_big
                    logging.info(f"CRAFT(header x {header_upscale}) : {len(polygons_hdr)} polygons (crop coords)")
                else:
                    polygons_hdr = _to_header_coords(polys_head, header_h)
                    scores_hdr = scores_head

                # Filter by confidence
                valid_polygons = []
                for i, polygon in enumerate(polygons_hdr):
                    sc = scores_hdr[i] if i < len(scores_hdr) else 1.0
                    if sc >= min_conf and polygon:
                        valid_polygons.append(polygon)

                if not valid_polygons:
                    logging.warning("No text regions met confidence threshold in header region")
                else:
                    # --- Build mask on the HEADER crop
                    mask = Image.new('L', (width, header_h), 0)  # same size as header crop
                    mask = mask.filter(ImageFilter.MaxFilter(9))
                    draw = ImageDraw.Draw(mask)
                    for polygon in valid_polygons:
                        try:
                            if isinstance(polygon, (list, tuple)) and all(isinstance(pt, (list, tuple, tuple)) and len(pt) == 2 for pt in polygon):
                                draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
                            else:
                                logging.warning(f"Invalid polygon format (skipped): {polygon}")
                        except Exception as e:
                            logging.warning(f"Failed to draw polygon: {e}")

                    # Apply mask
                    background = Image.new('RGB', (width, header_h), (255, 255, 255))
                    # keep the original header as "unmasked"
                    unmasked = original_cropped
                    # Keep the top band of the header even if CRAFT missed it (captures the masthead)
                    keep_top = float(os.getenv('HEADER_RATIO')) 
                    draw.rectangle([(0, 0), (width, int(header_h * keep_top))], fill=255)
                    masked = Image.composite(cropped.convert('RGB'), background, mask)
                    #cropped = text_focused # keep the masked header crop

                    # --- Visual debug 1: masked header crop
                    stem = Path(image_path).stem
                    masked_debug = os.path.join(MASKED_DIR, f"{stem}_masked.jpg")
                    try:
                        masked.save(masked_debug)
                        logging.info(f"Saved masked header debug image: {masked_debug}")
                    except Exception as e:
                        logging.warning(f"Failed to save masked header debug image: {e}")

                    # --- Visual debug 2: full-page overlay with header line + polygons
                    try:
                        dbg = img.copy().convert('RGB')
                        draw_full = ImageDraw.Draw(dbg)
                        # header boundary line
                        draw_full.line([(0, header_h), (width, header_h)], fill=(0, 255, 0), width=2)
                        # draw polygons in page coords (reconstruct page-space polys)
                        page_polys = polys_head if polys_head else [
                            # map crop polys to page coords (header is at y=0)
                            [(x, y) for (x, y) in poly] for poly in polygons_hdr
                        ]
                        for poly in page_polys:
                            if len(poly) >= 2:
                                draw_full.polygon(poly, outline=(255, 0, 0), width=2)
                        overlay_path = os.path.join(OVERLAY_DIR, f"{stem}_craft_header_boxes.jpg")
                        dbg.save(overlay_path)
                        logging.info(f"Saved CRAFT header overlay: {overlay_path}")
                    except Exception as e:
                        logging.warning(f"Failed saving CRAFT debug overlay: {e}")

            except Exception as e:
                logging.warning(f"CRAFT processing failed: {str(e)}", exc_info=True)
                cropped = original_cropped  # keep the original header crop for fallback to raw header

        # ---- OCR BOTH masked and unmasked headers; compare to env NEWSPAPER_TITLE
        def _norm(s: str) -> str:
            s = s.lower()
            # strip accents
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
            s = re.sub(r'\s+', ' ', s)
            s = re.sub(r'[^\w\s]', '', s)
            # common OCR confusions
            s = s.replace(' dellemilia', ' dell emilia') \
                 .replace('dellemilia', 'dell emilia') \
                 .replace('giornate', 'giornale') \
                 .replace('giornate dell emilia', 'giornale dell emilia')
            return s.strip()

        target_title = os.getenv('NEWSPAPER_TITLE', 'Giornale dell Emilia')
        target_norm = _norm(target_title)

        def _preprocess_variants(im: Image.Image):
            # Variant A: autocontrast only (keeps faint strokes)
            v1 = ImageOps.autocontrast(im.convert('L'))
            v1 = v1.resize((v1.width*2, v1.height*2), Image.LANCZOS)
            # Variant B: light binarization after autocontrast
            v2 = v1.point(lambda x: 0 if x < 150 else 255)
            return [v1, v2]

        def _ocr_best_ratio(im: Image.Image) -> Tuple[float, str]:
            cfgs = [
                "--psm 4 --oem 1 -c preserve_interword_spaces=1 -c tessedit_char_blacklist=|\\><[]{}`~_^",
                "--psm 6 --oem 1 -c preserve_interword_spaces=1 -c tessedit_char_blacklist=|\\><[]{}`~_^",
            ]
            best_r, best_txt = 0.0, ""
            for v in _preprocess_variants(im):
                for cfg in cfgs:
                    try:
                        raw = pytesseract.image_to_string(v, lang='ita', config=cfg)
                        clean = _norm(raw)
                        r = difflib.SequenceMatcher(None, target_norm, clean).ratio()
                        if r > best_r:
                            best_r, best_txt = r, clean
                    except Exception:
                        pass
            return best_r, best_txt

        ratio_masked, txt_masked = _ocr_best_ratio(masked)
        ratio_unmasked, txt_unmasked = _ocr_best_ratio(unmasked)

        logging.info(f"Headtitle fuzzy (masked vs '{target_norm}'): {ratio_masked:.2f}")
        logging.info(f"Headtitle fuzzy (unmasked vs '{target_norm}'): {ratio_unmasked:.2f}")

        if ratio_masked >= ratio_unmasked:
            best_ratio, best_source, best_text = ratio_masked, "masked", txt_masked
        else:
            best_ratio, best_source, best_text = ratio_unmasked, "unmasked", txt_unmasked

        logging.info(f"OCR headtitle BEST [{best_source}] (ratio {best_ratio:.2f}): '{best_text[:80]}'")

        if target_norm in best_text or best_ratio >= min_ratio:
            logging.info("Headtitle match accepted.")
            return True

        logging.warning(
            f"No headtitle match found against '{target_norm}'. Best ratio: {best_ratio:.2f}. "
            f"Sample: '{best_text[:60]}{'...' if len(best_text) > 60 else ''}'"
        )
        return False

    except Exception as e:
        logging.error(f"Headtitle verification failed: {str(e)}", exc_info=True)
        return False


# --- Date Extraction ---
def extract_date_with_ocr(image_path):
    """Date extraction with region handling"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Define priority regions for date extraction
        regions = [
            (0, 0, width, int(height * 0.15)),           # Top strip
            (int(width * 0.7), 0, width, int(height * 0.2)),  # Top-right corner
            (0, 0, int(width * 0.4), int(height * 0.1)),  # Top-left corner
            (int(width * 0.4), 0, int(width * 0.6), int(height * 0.1))  # Center top
        ]
        
        date_patterns = [
            r'\b\d{1,2}\s+[a-zA-Z]+\s+\d{4}\b',  # 12 Agosto 1946
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',         # 12/08/1946
            r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',       # 12.08.1946
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',         # 12-08-1946
            r'\b\d{4}-\d{1,2}-\d{1,2}\b'          # 1946-08-12
        ]
        
        for i, region in enumerate(regions):
            try:
                cropped = img.crop(region)
                # region preprocessing
                cropped = cropped.convert('L')  # Grayscale
                cropped = cropped.resize((cropped.width*2, cropped.height*2), Image.LANCZOS)  # Upscale
                cropped = cropped.point(lambda x: 0 if x < 180 else 255)  # Thresholding
                
                # Perform OCR
                text = pytesseract.image_to_string(
                    cropped, 
                    lang='ita',
                    config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
                )
                
                # Search for date patterns
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        date_str = match.group(0)
                        logging.info(f"Date found in region {i+1}: {date_str}")
                        return date_str
                        
            except Exception as e:
                logging.warning(f"OCR processing failed for region {i+1}: {str(e)}")
                continue
        
        # Final fallback: Full-page OCR
        try:
            full_text = pytesseract.image_to_string(
                img.convert('L'),
                lang='ita',
                config='--psm 3 --oem 3'
            )
            for pattern in date_patterns:
                match = re.search(pattern, full_text)
                if match:
                    date_str = match.group(0)
                    logging.info(f"Date found in full-page scan: {date_str}")
                    return date_str
        except Exception as e:
            logging.error(f"Full-page OCR failed: {str(e)}")
            
        return None
        
    except Exception as e:
        logging.error(f"Date extraction failed: {str(e)}")
        return None

# --- Multimodal Model Handling ---
MODEL_CACHE = None

def load_multimodal_model():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        try:
            MODEL_CACHE = SentenceTransformer('clip-ViT-B-32')
            logging.info("Multimodal model loaded successfully")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise
    return MODEL_CACHE

# --- Page Classification ---
def classify_page(image_path, headtitles):
    """
    Classify pages with headtitle verification for front pages
    Returns classification with confidence
    """
    try:
        # 1. Headtitle check (OCR)
        if is_valid_front_page(image_path, headtitles):
            return "Front Page", 1.0

        # 2. Model prediction
        model = load_multimodal_model()
        img = Image.open(image_path)
        img_emb = model.encode(img, convert_to_tensor=True)

        prompts = {
            "Front Page": [
                "Front page of an Italian daily newspaper, featuring a prominent masthead (e.g., GIORNALE dell'EMILIA), the publication date, issue date (e.g., 21 Luglio 1946) and number (e.g., N. 196) and main headlines"
            ],
            "Internal Page": [
                "Inside page of a periodical, featuring multiple articles, some of which may be continued from other pages, along with advertisements."
            ]
        }

        category_scores = {}
        for category, category_prompts in prompts.items():
            prompt_embs = model.encode(category_prompts, convert_to_tensor=True)
            similarities = util.cos_sim(img_emb, prompt_embs)
            category_scores[category] = torch.max(similarities).item()

        # Thresholds
        FRONT_THRESHOLD = 0.33
        INTERNAL_THRESHOLD = 0.30

        if category_scores["Front Page"] >= FRONT_THRESHOLD:
            return "Front Page", category_scores["Front Page"]
        elif category_scores["Internal Page"] >= INTERNAL_THRESHOLD:
            return "Internal Page", category_scores["Internal Page"]
        else:
            return "Uncertain", max(category_scores.values())

    except Exception as e:
        logging.error(f"Classification failed: {str(e)}")
        return "Classification Error", 0.0
    
# --- Main Processing Function ---
def process_newspaper_images():
    # Load headtitles from catalog
    catalog_path = CATALOG_PATH
    headtitles = []
    if os.path.exists(catalog_path):
        headtitles = load_headtitles(catalog_path)
    else:
        logging.warning(f"Catalog not found at {catalog_path}. Using fallback headtitles.")
        # Fallback to known titles
        headtitles = [
            "il resto del carlino",
            "giornale dell emilia",
            "giornale dell emilia riunite",
            "il piccolo faust",
            "periodico fantastico artistico teatrale"
        ]
    
    # Configure OCR
    configure_ocr()
    
    # Load existing results
    processed_data = {}
    
    # Find images
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    image_paths = [
    p for p in image_paths
    if "_masked." not in os.path.basename(p).lower()
]
    
    if not image_paths:
        logging.error(f"No images found in: {IMAGE_FOLDER}")
        return
        
    image_paths.sort()
    logging.info(f"Found {len(image_paths)} images to process")
    
    # Date validation range
    min_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    max_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    last_valid_date = None
    
    # Process images
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        page_num = idx + 1
        
        logging.info(f"\n{'='*40}")
        logging.info(f"Processing page {page_num}/{len(image_paths)}: {filename}")
        logging.info(f"{'='*40}")
        
        result = {
            "filename": filename,
            "page_number": page_num,
            "classification": "Unknown",
            "classification_confidence": 0.0,
            "issue_date": "N/A",
            "issue_number": "N/A",
            "hasGraphics": False,
            "processing_errors": []
        }
        
        try:
            # Classify page with headtitle verification
            classification, confidence = classify_page(img_path, headtitles)
            result["classification"] = classification
            result["classification_confidence"] = confidence
            
            # Date extraction for front pages
            if classification == "Front Page":
                date_str = extract_date_with_ocr(img_path)
                if date_str:
                    result["issue_date"] = date_str
                    
                    # Date validation
                    date_obj = parse_italian_date(date_str)
                    if date_obj:
                        # Range validation
                        if min_date <= date_obj <= max_date:
                            # Chronology validation
                            if last_valid_date is None or date_obj >= last_valid_date:
                                result["issue_date"] = date_obj.strftime('%Y-%m-%d')
                                last_valid_date = date_obj
                            else:
                                result["processing_errors"].append(
                                    f"Date {date_obj.strftime('%Y-%m-%d')} is earlier than last valid date"
                                )
                                result["issue_date"] = "N/A"
                        else:
                            result["processing_errors"].append(
                                f"Date {date_obj.strftime('%Y-%m-%d')} outside valid range"
                            )
                            result["issue_date"] = "N/A"
                    else:
                        result["processing_errors"].append(
                            f"Failed to parse date: {date_str}"
                        )
                        result["issue_date"] = "N/A"
                else:
                    result["processing_errors"].append("No date extracted from front page")
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            result["processing_errors"].append(error_msg)
        
        processed_data[page_num] = result

    # --- Write results ---
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(list(processed_data.values()), f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to: {output_json}")

# --- Italian Date Parser ---
def parse_italian_date(date_str):
    """Robust Italian date parser with multiple format support"""
    if not date_str or date_str.lower() in ["", "n/a", "null", "none"]:
        return None

    # Clean input string
    date_str = re.sub(r'[^\w\s/.-]', '', date_str.lower().strip())
    
    # Italian month mappings
    months = {
        'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4, 'maggio': 5, 'giugno': 6,
        'luglio': 7, 'agosto': 8, 'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12,
        'gen': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'mag': 5, 'giu': 6,
        'lug': 7, 'ago': 8, 'set': 9, 'ott': 10, 'nov': 11, 'dic': 12
    }
    
    # Try standard formats first
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    
    # Try formats with Italian month names
    for month_name, month_num in months.items():
        if month_name in date_str:
            # Day-Month-Year pattern
            pattern = rf'(\d{{1,2}})\s*{month_name}\s*(\d{{4}})'
            match = re.search(pattern, date_str)
            if match:
                try:
                    day = int(match.group(1))
                    year = int(match.group(2))
                    return datetime(year, month_num, day)
                except ValueError:
                    continue
                    
            # Year-Month-Day pattern
            pattern = rf'(\d{{4}})\s*{month_name}\s*(\d{{1,2}})'
            match = re.search(pattern, date_str)
            if match:
                try:
                    year = int(match.group(1))
                    day = int(match.group(2))
                    return datetime(year, month_num, day)
                except ValueError:
                    continue
    
    # Numeric formats with different separators
    for sep in [' ', '.', '-', '/']:
        # Day-Month-Year
        pattern = rf'(\d{{1,2}}){sep}(\d{{1,2}}){sep}(\d{{4}})'
        match = re.search(pattern, date_str)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                continue
        
        # Year-Month-Day
        pattern = rf'(\d{{4}}){sep}(\d{{1,2}}){sep}(\d{{1,2}})'
        match = re.search(pattern, date_str)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                continue
    
    logging.warning(f"Could not parse date string: '{date_str}'")
    return None

# --- Argument Parsing ---
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.image_folder = IMAGE_FOLDER
            self.start_date = START_DATE
            self.end_date = END_DATE
            self.title = NEWSPAPER_TITLE
    
    args = Args()
    process_newspaper_images()