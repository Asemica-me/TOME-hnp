import os
import glob
import logging
import json
import re
import base64
import requests
from PIL import Image
from fuzzywuzzy import process
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Union
from sentence_transformers import SentenceTransformer, util

# --- Date Conversion Helper ---
def parse_italian_date(date_str: str) -> Optional[datetime]:
    """
    Parses a date string with Italian month names into a datetime object.
    Handles formats like '6 Agosto 1946' or '21/07/1946'.
    """
    if not date_str or date_str == "N/A":
        return None

    italian_months = {
        'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4, 'maggio': 5, 'giugno': 6,
        'luglio': 7, 'agosto': 8, 'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12
    }
    
    date_str_lower = date_str.lower()
    
    # Try dd/mm/yyyy format first
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except ValueError:
        pass

    # Try format with month name
    for month_name, month_num in italian_months.items():
        if month_name in date_str_lower:
            # Replace month name with number and clean the string
            date_str_cleaned = re.sub(r'[^\d\s]', '', date_str_lower.replace(month_name, str(month_num)))
            parts = date_str_cleaned.split()
            if len(parts) >= 3:
                try:
                    # Attempt to build date from day, month, year
                    day = int(parts[0])
                    month = int(parts[1])
                    year = int(parts[2])
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    continue # Try next month if parsing fails
    
    logging.warning(f"Could not parse date string: '{date_str}'")
    return None

# --- Local Model Inference ---
def load_multimodal_model():
    """Load a multimodal model from Sentence Transformers"""
    return SentenceTransformer('clip-ViT-B-32')

def query_local_model(prompt: str, image_path: str) -> str:
    """
    Use a local multimodal model to answer questions about an image
    """
    try:
        # Load model (cached after first load)
        model = load_multimodal_model()
        
        # Encode text and image
        text_emb = model.encode([prompt], convert_to_tensor=True)
        img_emb = model.encode(Image.open(image_path), convert_to_tensor=True)
        
        # Calculate similarity
        cos_scores = util.cos_sim(text_emb, img_emb)
        score = cos_scores.item()
        
        # Simple threshold-based decision
        return "Front Page" if score > 0.3 else "Internal Page"
    
    except Exception as e:
        logging.error(f"Local model inference failed: {str(e)}")
        return "MODEL_ERROR"

# --- Classification Normalization Helper ---
def normalize_classification(raw_classification: str) -> str:
    """
    Normalizes raw classification strings to predefined categories,
    prioritizing specific known misspellings, then using fuzzy matching.
    """
    if not isinstance(raw_classification, str):
        return "Unknown"

    raw_classification_lower = raw_classification.lower().strip()

    # Predefined correct classifications for a newspaper
    correct_classifications = [
        "Front Page",
        "Internal Page",
        "Back Page",
        "Full-Page Advertisement",
        "Editorial Page",
        "Sports Page",
        "Classifieds Page",
        "Blank Page",
        "Reference"
    ]

    # Step 1: Specific corrections for common misspellings
    if raw_classification_lower == "external fro cover":
        logging.info(f"Specific correction: '{raw_classification}' to 'External Front Cover'.")
        return "External Front Cover"
    if raw_classification_lower == "iernal back cover" or raw_classification_lower == "interal back cover":
        logging.info(f"Specific correction: '{raw_classification}' to 'Internal Back Cover'.")
        return "Internal Back Cover"
    if raw_classification_lower == "iernal fro cover":
        logging.info(f"Specific correction: '{raw_classification}' to 'Internal Front Cover'.")
        return "Internal Front Cover"
    
    # Extend with other specific variants
    if "fro cover" in raw_classification_lower and "external" in raw_classification_lower:
        logging.info(f"Specific correction (substring): '{raw_classification}' to 'External Front Cover'.")
        return "External Front Cover"
    if "fro cover" in raw_classification_lower and "internal" in raw_classification_lower:
        logging.info(f"Specific correction (substring): '{raw_classification}' to 'Internal Front Cover'.")
        return "Internal Front Cover"
    if "bac cover" in raw_classification_lower and "internal" in raw_classification_lower:
        logging.info(f"Specific correction (substring): '{raw_classification}' to 'Internal Back Cover'.")
        return "Internal Back Cover"
    if "bac cover" in raw_classification_lower and "external" in raw_classification_lower:
        logging.info(f"Specific correction (substring): '{raw_classification}' to 'External Back Cover'.")
        return "External Back Cover"

    # Step 2: Case-insensitive direct match
    for correct_cat in correct_classifications:
        if raw_classification_lower == correct_cat.lower():
            logging.info(f"Direct match: '{raw_classification}' to '{correct_cat}'.")
            return correct_cat
            
    # Step 3: Fuzzy Matching as general fallback
    choices_lower = [c.lower() for c in correct_classifications]
    best_match_lower, score = process.extractOne(raw_classification_lower, choices_lower)

    SIMILARITY_THRESHOLD = 85

    if score >= SIMILARITY_THRESHOLD:
        normalized_category = correct_classifications[choices_lower.index(best_match_lower)]
        if raw_classification != normalized_category:
            logging.info(f"Fuzzy normalized: '{raw_classification}' (score: {score}) to '{normalized_category}'.")
        return normalized_category
    else:
        logging.warning(f"Classification '{raw_classification}' (score: {score}) did not meet similarity threshold. Returning as is.")
        return raw_classification

# --- Main Processing Function ---
def run_api_based_inference():
    # Define output folder and filename
    OUTPUT_FOLDER = "output"
    OUTPUT_JSON_FILENAME = "output_extraction.json"

    # Make sure the output folder exists; create if it doesn't
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Build the full path to the output JSON file
    output_json_path = os.path.join(OUTPUT_FOLDER, OUTPUT_JSON_FILENAME)

    final_structured_outputs = [] 
    last_valid_date = None

    logging.info("Script started.") 

    # Load existing data if JSON file exists
    if os.path.exists(output_json_path) and os.path.getsize(output_json_path) > 0:
        logging.info(f"Checking for existing output file: {output_json_path}") 
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                final_structured_outputs = json.load(f)
            logging.info(f"Loaded {len(final_structured_outputs)} existing records from {output_json_path}.")
        except json.JSONDecodeError as e:
            logging.warning(f"Existing {output_json_path} is corrupted or empty: {e}. Starting with an empty list.")
            final_structured_outputs = []
        except Exception as e:
            logging.warning(f"Error loading existing {output_json_path}: {e}. Starting with an empty list.")
            final_structured_outputs = []
    else:
        logging.info(f"No existing output file found at {output_json_path}. Starting fresh.") 

    logging.info(f"Scanning for images in: {IMAGE_FOLDER}")
    image_paths = []

    # Look for images in the resized subfolder
    images_subfolder = os.path.join(IMAGE_FOLDER)

    if not os.path.exists(images_subfolder):
        logging.warning(f"Images subfolder does not exist: {images_subfolder}. Exiting.")
        return
    
    logging.info(f"Looking for images in: {images_subfolder}")
    
    for ext in SUPPORTED_EXTENSIONS:
        current_glob_pattern = os.path.join(images_subfolder, f'*{ext}')
        image_paths.extend(glob.glob(current_glob_pattern))
    
    image_paths.sort()
    logging.info(f"Found {len(image_paths)} images after sorting.") 

    if not image_paths:
        logging.warning(f"No supported images found in: {IMAGE_FOLDER}. Exiting.")
        return

    logging.info(f"Processing {len(image_paths)} images page by page.")
    logging.debug(f"First image path to process: {image_paths[0] if image_paths else 'N/A'}") 

    for i, img_path in enumerate(image_paths):
        current_page_num = i + 1
        filename = os.path.basename(img_path)
        
        # Initialize result record for this page
        page_result_record = {
            "filename": filename,
            "llm_raw_response": None,
            "parsed_llm_output": None
        }

        logging.debug(f"Checking if {filename} (page {current_page_num}) has already been processed.") 
        if any(record.get('filename') == filename for record in final_structured_outputs):
            logging.info(f"Page {current_page_num} ({filename}) already processed. Skipping.")
            continue

        try:
            logging.info(f"--- Processing Page {current_page_num}/{len(image_paths)}: {filename} ---")

            # --- PROMPT 1: CLASSIFICATION ONLY ---
            classification_prompt = (
                f"Is this image the front page of an Italian newspaper? Look for masthead, date, and main headlines."
            )
            
            logging.info(f"Classifying page {current_page_num} with local model...")
            raw_classification = query_local_model(classification_prompt, img_path)
            
            classification = normalize_classification(raw_classification)
            logging.info(f"Page {current_page_num} classified as: {classification}")

            entry = {
                "filename": filename,
                "classification": classification,
                "issue_date": "N/A",
                "issue_number": "N/A",
                "hasGraphics": False,
                "page_number": "N/A"
            }

            # For front pages, attempt date extraction
            if classification == "Front Page":
                logging.info(f"Front Page detected. Attempting date extraction for page {current_page_num}.")
                # Simplified date extraction prompt
                date_prompt = "What is the publication date shown on this newspaper? Respond only with the date."
                date_response = query_local_model(date_prompt, img_path)
                
                # Simple pattern matching for dates
                date_pattern = r'\b\d{1,2}\s+[\w]+\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
                date_match = re.search(date_pattern, date_response)
                
                if date_match:
                    extracted_date = date_match.group(0)
                    entry["issue_date"] = extracted_date
                    logging.info(f"Extracted date for page {current_page_num}: {extracted_date}")
                else:
                    logging.warning(f"No date found for front page {current_page_num}")

            # --- DATE VALIDATION ---
            original_date_str = entry.get("issue_date", "N/A")
            extracted_date_obj = parse_italian_date(original_date_str)

            if extracted_date_obj:
                try:
                    min_date = datetime.strptime(args.start_date, "%Y-%m-%d")
                    max_date = datetime.strptime(args.end_date, "%Y-%m-%d")

                    # Global range check
                    if not (min_date <= extracted_date_obj <= max_date):
                        logging.warning(f"Page {current_page_num}: Date {original_date_str} -> {extracted_date_obj.strftime('%d/%m/%Y')} is outside the valid range. Resetting.")
                        entry["issue_date"] = "N/A"
                    # Chronological consistency check
                    elif last_valid_date and extracted_date_obj < last_valid_date:
                        logging.warning(f"Page {current_page_num}: Date {original_date_str} -> {extracted_date_obj.strftime('%d/%m/%Y')} is earlier than last valid date. Resetting.")
                        entry["issue_date"] = "N/A"
                    else:
                        # Valid date found - normalize and update state
                        entry["issue_date"] = extracted_date_obj.strftime('%d/%m/%Y')
                        last_valid_date = extracted_date_obj
                        logging.info(f"Page {current_page_num}: Valid date found and normalized: {entry['issue_date']}")
                except Exception as e:
                    logging.error(f"Error during date range validation: {e}")
                    entry["issue_date"] = "N/A"
            else:
                if original_date_str != "N/A":
                    logging.warning(f"Page {current_page_num}: Could not parse date '{original_date_str}'. Resetting to 'N/A'.")
                entry["issue_date"] = "N/A"
            
            # Assign final entry to page record
            page_result_record["parsed_llm_output"] = entry
            logging.info(f"Final data for page {current_page_num}: {entry}")

        except Exception as e:
            logging.error(f"Critical error during processing for page {current_page_num}: {e}", exc_info=True)
            page_result_record["parsed_llm_output"] = {
                "filename": filename,
                "classification": "Processing Critical Error",
                "hasGraphics": False,
                "details": str(e)
            }

        # Update and save JSON file after each page
        logging.debug(f"State before final append for {filename}: {page_result_record!r}")

        # Remove existing record for this page if present
        final_structured_outputs = [
            rec for rec in final_structured_outputs if rec.get('filename') != filename
        ]
        final_structured_outputs.append(page_result_record["parsed_llm_output"])
        
        logging.debug(f"Attempting to save partial output for page {current_page_num} to {output_json_path}.") 
        try:
            final_structured_outputs.sort(key=lambda x: x.get('filename', ''))
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_structured_outputs, f, indent=4, ensure_ascii=False)
            logging.info(f"Partial output for page {current_page_num} saved to: {output_json_path}")
        except Exception as save_e:
            logging.error(f"Error saving partial output after processing page {current_page_num} ({filename}): {save_e}")

    # Final summary
    logging.info("\n" + "="*80)
    logging.info("FINAL COLLATED RESULTS FROM ALL PAGES")
    logging.info("="*80)

    if final_structured_outputs:
        logging.info(f"Processing complete. All results saved to: {output_json_path}")
    else:
        logging.info("No outputs were generated from any page.")
    logging.info("="*80)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Script Parameters ---
parser = argparse.ArgumentParser(description="Process catalogue images with local multimodal model")
parser.add_argument(
    "--image-folder",
    type=str,
    default="imgs",
    help="Path to the folder containing images to process."
)
parser.add_argument(
    "--start-date",
    type=str,
    default="1946-07-21",
    help="The start of the valid date range in YYYY-MM-DD format."
)
parser.add_argument(
    "--end-date",
    type=str,
    default="1947-05-15",
    help="The end of the valid date range in YYYY-MM-DD format."
)
parser.add_argument(
    "--title",
    type=str,
    default="Resto del Carlino",
    help="Titolo della testata"
)
args = parser.parse_args()

IMAGE_FOLDER = args.image_folder
title = args.title

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
OUTPUT_JSON_FILENAME = os.path.basename(os.path.normpath(IMAGE_FOLDER)) + ".json"

if __name__ == "__main__":
    run_api_based_inference()