import os
import glob
import logging
import json
import re
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Union
from PIL import Image
import pytesseract
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer, util
import torch

# --- Configure Tesseract Path ---

# Windows example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux example: '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_PATH = pytesseract.pytesseract.tesseract_cmd
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    logging.warning(f"Tesseract path not found: {TESSERACT_PATH}. OCR will be disabled.")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("newspaper_processing.log")
    ]
)

# --- Date Parsing Helper ---
def parse_italian_date(date_str: str) -> Optional[datetime]:
    """Robustly parse Italian date strings with multiple format support"""
    if not date_str or date_str.lower() in ["", "n/a", "null", "none"]:
        return None

    italian_months = {
        'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4, 'maggio': 5, 'giugno': 6,
        'luglio': 7, 'agosto': 8, 'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'mag': 5, 'giu': 6,
        'lug': 7, 'ago': 8, 'set': 9, 'ott': 10, 'nov': 11, 'dic': 12
    }
    
    # Pre-clean the string
    date_str = re.sub(r'[^\w\s/.-]', '', date_str.lower().strip())
    
    # Try common formats first
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass

    # Try formats with Italian month names
    for month_name, month_num in italian_months.items():
        if month_name in date_str:
            # Pattern: day + month + year
            pattern = rf'(\d{{1,2}})\s*{month_name}\s*(\d{{4}})'
            match = re.search(pattern, date_str)
            if match:
                try:
                    day = int(match.group(1))
                    year = int(match.group(2))
                    return datetime(year, month_num, day)
                except ValueError:
                    continue
                    
            # Pattern: month + day + year
            pattern = rf'{month_name}\s*(\d{{1,2}}),\s*(\d{{4}})'
            match = re.search(pattern, date_str)
            if match:
                try:
                    day = int(match.group(1))
                    year = int(match.group(2))
                    return datetime(year, month_num, day)
                except ValueError:
                    continue

    # Try numeric formats with different separators
    for separator in [' ', '.', '-', '/']:
        pattern = rf'(\d{{1,2}}){separator}(\d{{1,2}}){separator}(\d{{4}})'
        match = re.search(pattern, date_str)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                continue

    logging.warning(f"Could not parse date string: '{date_str}'")
    return None

# --- OCR Date Extraction ---
def extract_date_with_ocr(image_path: str) -> Optional[str]:
    """Extract date from newspaper image using OCR with region focus"""
    if not os.path.exists(TESSERACT_PATH):
        logging.error("Tesseract not configured. Skipping OCR.")
        return None
        
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Common newspaper date locations
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
                # Preprocess image for better OCR
                cropped = cropped.convert('L')  # Grayscale
                cropped = cropped.point(lambda x: 0 if x < 150 else 255)  # Thresholding
                
                # Perform OCR with Italian language
                text = pytesseract.image_to_string(
                    cropped, 
                    lang='ita',
                    config='--psm 6 -c preserve_interword_spaces=1'
                )
                
                # Search for date patterns in OCR text
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        date_str = match.group(0)
                        logging.info(f"Found date in region {i+1}: {date_str}")
                        return date_str
                        
            except Exception as e:
                logging.warning(f"OCR failed for region {i+1}: {str(e)}")
                continue
        
        # Fallback: Full-page OCR
        try:
            full_text = pytesseract.image_to_string(
                img.convert('L'),
                lang='ita',
                config='--psm 3'
            )
            for pattern in date_patterns:
                match = re.search(pattern, full_text)
                if match:
                    date_str = match.group(0)
                    logging.info(f"Found date in full-page scan: {date_str}")
                    return date_str
        except Exception as e:
            logging.error(f"Full-page OCR failed: {str(e)}")
            
    except Exception as e:
        logging.error(f"OCR processing failed: {str(e)}")
    
    return None

# --- Multimodal Model Handling ---
MODEL_CACHE = None

def load_multimodal_model():
    """Load and cache the multimodal model"""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        try:
            MODEL_CACHE = SentenceTransformer('clip-ViT-B-32')
            logging.info("Multimodal model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    return MODEL_CACHE

# --- Page Classification ---
def classify_page(image_path: str) -> str:
    """Classify newspaper pages using multimodal comparison"""
    try:
        model = load_multimodal_model()
        img = Image.open(image_path)
        img_emb = model.encode(img, convert_to_tensor=True)
        
        # Define classification prompts
        prompts = {
            "Front Page": [
                "Front page of an Italian newspaper with masthead and date",
                "First page of a newspaper with headlines and publication date",
                "Cover page of a daily newspaper showing main stories",
                "Newspaper front cover featuring prominent headlines"
            ],
            "Internal Page": [
                "Inside page of a newspaper with multiple articles",
                "Newspaper page with continued articles and advertisements",
                "Regular content page in a periodical publication",
                "Inner page of a newspaper with text columns"
            ],
            "Back Page": [
                "Last page of a newspaper",
                "Back cover of a publication",
                "Final page of a periodical"
            ]
        }
        
        # Calculate similarity scores
        category_scores = {}
        for category, category_prompts in prompts.items():
            prompt_embs = model.encode(category_prompts, convert_to_tensor=True)
            similarities = util.cos_sim(img_emb, prompt_embs)
            category_scores[category] = torch.max(similarities).item()
        
        # Determine classification with confidence thresholds
        FRONT_THRESHOLD = 0.35
        INTERNAL_THRESHOLD = 0.3
        BACK_THRESHOLD = 0.25
        
        # Get best score and category
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # Apply thresholds
        if best_category == "Front Page" and best_score > FRONT_THRESHOLD:
            return "Front Page"
        elif best_category == "Internal Page" and best_score > INTERNAL_THRESHOLD:
            return "Internal Page"
        elif best_category == "Back Page" and best_score > BACK_THRESHOLD:
            return "Back Page"
        elif best_score > 0.2:  # General threshold
            return best_category
        else:
            return "Uncertain"
            
    except Exception as e:
        logging.error(f"Classification failed: {str(e)}")
        return "Classification Error"

# --- Classification Normalization ---
def normalize_classification(raw_classification: str) -> str:
    """Normalize classification strings to standard categories"""
    if not isinstance(raw_classification, str):
        return "Unknown"

    raw_classification = raw_classification.strip()
    raw_lower = raw_classification.lower()

    # Standard categories
    standard_categories = [
        "Front Page", "Internal Page", "Back Page", 
        "External Front Cover", "External Back Cover",
        "Internal Front Cover", "Internal Back Cover",
        "Full-Page Advertisement", "Editorial Page",
        "Sports Page", "Classifieds Page", "Blank Page", "Reference"
    ]

    # Direct matches
    for cat in standard_categories:
        if raw_lower == cat.lower():
            return cat

    # Common misspelling corrections
    corrections = {
        "fro cover": "Front Cover",
        "bac cover": "Back Cover",
        "iernal": "Internal",
        "external": "External",
        "advertisment": "Advertisement",
        "add": "Advertisement"
    }

    # Apply corrections
    corrected = raw_classification
    for error, fix in corrections.items():
        if error in raw_lower:
            corrected = corrected.replace(error, fix)
            logging.info(f"Corrected '{raw_classification}' to '{corrected}'")

    # Fuzzy matching fallback
    choices_lower = [c.lower() for c in standard_categories]
    best_match, score = process.extractOne(raw_lower, choices_lower)
    
    if score > 75:
        normalized = standard_categories[choices_lower.index(best_match)]
        if normalized != raw_classification:
            logging.info(f"Fuzzy matched: '{raw_classification}' -> '{normalized}' ({score}%)")
        return normalized
    
    return raw_classification  # Return original if no good match

# --- Main Processing Function ---
def process_newspaper_images():
    # Setup output paths
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, f"extraction_output.json")
    
    # Load existing results
    processed_data = []
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
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
    
    # Track date consistency
    last_valid_date = None
    date_range_valid = True
    min_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    max_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Process images
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        page_num = idx + 1
        
        # Skip already processed
        if any(entry.get("filename") == filename for entry in processed_data):
            logging.info(f"Skipping already processed: {filename}")
            continue
            
        logging.info(f"\n{'='*40}")
        logging.info(f"Processing page {page_num}/{len(image_paths)}: {filename}")
        logging.info(f"{'='*40}")
        
        result = {
            "filename": filename,
            "page_number": page_num,
            "classification": "Unknown",
            "issue_date": "N/A",
            "issue_number": "N/A",
            "hasGraphics": False,
            "processing_errors": []
        }
        
        try:
            # Step 1: Page classification
            classification = classify_page(img_path)
            normalized_class = normalize_classification(classification)
            result["classification"] = normalized_class
            logging.info(f"Classification: {normalized_class}")
            
            # Step 2: Date extraction for front pages
            if normalized_class in ["Front Page", "External Front Cover", "Internal Front Cover"]:
                date_str = extract_date_with_ocr(img_path)
                if date_str:
                    result["issue_date"] = date_str
                    logging.info(f"Extracted date: {date_str}")
                else:
                    logging.warning("No date found on front page")
            
            # Step 3: Date validation
            if result["issue_date"] != "N/A":
                date_obj = parse_italian_date(result["issue_date"])
                if date_obj:
                    # Validate date range
                    if not (min_date <= date_obj <= max_date):
                        result["processing_errors"].append(
                            f"Date {date_obj.strftime('%Y-%m-%d')} outside range "
                            f"{args.start_date} to {args.end_date}"
                        )
                        result["issue_date"] = "N/A"
                        date_range_valid = False
                    # Validate chronology
                    elif last_valid_date and date_obj < last_valid_date:
                        result["processing_errors"].append(
                            f"Date {date_obj.strftime('%Y-%m-%d')} is earlier than "
                            f"previous date {last_valid_date.strftime('%Y-%m-%d')}"
                        )
                        result["issue_date"] = "N/A"
                        date_range_valid = False
                    else:
                        # Valid date
                        result["issue_date"] = date_obj.strftime('%Y-%m-%d')
                        last_valid_date = date_obj
                        date_range_valid = True
                        logging.info(f"Validated date: {result['issue_date']}")
                else:
                    result["processing_errors"].append(
                        f"Failed to parse date: {result['issue_date']}"
                    )
                    result["issue_date"] = "N/A"
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            result["processing_errors"].append(error_msg)
        
        # Save results
        processed_data.append(result)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved results for {filename}")
    
    logging.info("\nProcessing complete!")
    logging.info(f"Results saved to: {output_json}")

# --- Argument Parsing ---
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Italian Newspaper Processor")
    # parser.add_argument("--image-folder", required=True, 
    #                     help="Folder containing newspaper images")
    # parser.add_argument("--start-date", default="1946-07-21",
    #                     help="Earliest valid date (YYYY-MM-DD)")
    # parser.add_argument("--end-date", default="1947-05-15",
    #                     help="Latest valid date (YYYY-MM-DD)")
    # parser.add_argument("--title", default="Resto del Carlino",
    #                     help="Newspaper title")
    
    class Args:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        def __init__(self):
            self.image_folder = os.path.join(self.script_dir, "imgs")
            self.start_date = "1946-07-21"
            self.end_date = "1947-05-15"
            self.title = "Resto del Carlino"
    
    args = Args()
    IMAGE_FOLDER = args.image_folder
    
    # Verify image folder
    if not os.path.exists(IMAGE_FOLDER):
        logging.error(f"Image folder not found: {IMAGE_FOLDER}")
        exit(1)
    
    process_newspaper_images()
    
    # Verify Tesseract
    if not os.path.exists(TESSERACT_PATH):
        logging.warning("Tesseract not found - OCR features disabled")
    
    process_newspaper_images()