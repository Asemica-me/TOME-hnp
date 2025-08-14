import os
import glob
import logging
import json
import re
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, util
import torch
import difflib
from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
logs_folder = os.path.join(root_dir, 'logs')
os.makedirs(logs_folder, exist_ok=True)

log_file_path = os.path.join(logs_folder, "script_lu.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path)
    ]
)

IMAGE_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'imgs'))
CATALOG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'catalog', 'catalog_cleaned.xml'))
START_DATE = os.getenv('START_DATE', '1946-07-21')
END_DATE = os.getenv('END_DATE', '1947-05-15')
NEWSPAPER_TITLE = os.getenv('NEWSPAPER_TITLE', 'Giornale dell Emilia')

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_PATH = pytesseract.pytesseract.tesseract_cmd
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    logging.error(f"Tesseract not found at {TESSERACT_PATH}. Please install it or update the path.")
    exit(1)


# --- XML Catalog Processing ---
def load_headtitles(catalog_path):
    """
    Extract all historical headtitles from UNIMARC catalog
    Returns a list of normalized headtitles
    """
    headtitles = set()
    try:
        tree = ET.parse(catalog_path)
        root = tree.getroot()
        
        # Namespace handling
        ns = {'marc': 'http://www.loc.gov/MARC21/slim'}
        
        for record in root.findall('marc:record', ns):
            # Extract both code="a" and code="e" from datafield tags 182 and 200
            for tag in [182, 200]:
                for datafield in record.findall(f"marc:datafield[@tag='{tag}']", ns):
                    for subfield in datafield.findall("marc:subfield[@code='a' or @code='e']", ns):
                        if subfield.text:
                            title = subfield.text.strip()
                            # Normalize: lowercase, remove non-alphanumeric
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

# --- Front Page Verification ---
def is_valid_front_page(image_path, headtitles, min_ratio=0.4):
    """
    Optimized headtitle verification for full-width top region
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # Define full-width top region (top 10% of page)
        region = (0, 0, width, int(height * 0.08))
        cropped = img.crop(region)

        # Optimized preprocessing for newspaper headlines
        def preprocess(image):
            # Convert to grayscale
            img_gray = image.convert('L')
            
            # Enhance contrast
            img_contrast = Image.eval(img_gray, lambda x: 0 if x < 100 else 255)
            
            # Upscale for better OCR
            img_large = img_contrast.resize(
                (img_contrast.width * 2, img_contrast.height * 2),
                Image.LANCZOS
            )
            return img_large

        processed = preprocess(cropped)
        
        # OCR with newspaper-optimized configuration
        text = pytesseract.image_to_string(
            processed,
            lang='ita',
            config=(
                '--psm 6 '        # Assume uniform block of text
                '--oem 3 '        # LSTM + Legacy engine
                '-c preserve_interword_spaces=1 '
                '-c tessedit_char_blacklist=|\\><[]{}`~_^'
            )
        ).lower()

        # Clean and normalize text
        clean_text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        clean_text = re.sub(r'[^\w\s]', '', clean_text).strip()
        logging.info(f"OCR headtitle text: '{clean_text}'")

        # Target headtitle variations
        target_titles = [
            "giornale dell emilia",
            "il giornale dell emilia",
            "giornale emilia"
        ]
        
        # Check for direct matches first
        for title in target_titles:
            if title in clean_text:
                logging.info(f"Exact headtitle match: '{title}'")
                return True
        
        # Fuzzy match as fallback
        for title in target_titles:
            ratio = difflib.SequenceMatcher(None, title, clean_text).ratio()
            if ratio >= min_ratio:
                logging.info(f"Headtitle fuzzy match: '{title}' (ratio: {ratio:.2f})")
                return True

        logging.warning(f"No headtitle match found. Best text: '{clean_text[:50]}...'")
        return False

    except Exception as e:
        logging.error(f"Headtitle verification failed: {str(e)}")
        return False

# --- Date Extraction Improvements ---
def extract_date_with_ocr(image_path):
    """Date extraction with improved region handling"""
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
        
        # Final fallback: Full-page OCR with optimized settings
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
    
    # Setup output
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, "script_lu.json")
    
    # Load existing results
    processed_data = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                for entry in json.load(f):
                    processed_data[entry["filename"]] = entry
            logging.info(f"Loaded {len(processed_data)} existing records")
        except Exception as e:
            logging.warning(f"Failed to load existing data: {str(e)}")
    
    # Find images
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    
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
        
        # --- Only keep the latest result for each filename ---
        processed_data[filename] = result

    # --- Write deduplicated results ---
    #output_json = os.path.join(output_folder, "extraction_output.json")
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