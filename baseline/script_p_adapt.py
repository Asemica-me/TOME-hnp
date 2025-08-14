import os
import glob
import logging
import json
import re
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
from fuzzywuzzy import process
import torch
import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
logs_folder = os.path.join(root_dir, 'logs')
os.makedirs(logs_folder, exist_ok=True)

log_file_path = os.path.join(logs_folder, "script_p_adapt.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path)
    ]
)

IMAGE_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'imgs'))

# Setup output folder path inside the script directory
output_folder = os.path.join(root_dir, "output")
os.makedirs(output_folder, exist_ok=True)
output_json = os.path.join(output_folder, "extraction_p.json")

# --- Date Conversion Helper ---
def parse_italian_date(date_str: str) -> datetime | None:
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
    
    # Prova il formato dd/mm/yyyy prima
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except ValueError:
        pass

    # Prova il formato con il nome del mese
    for month_name, month_num in italian_months.items():
        if month_name in date_str_lower:
            # Sostituisci il nome del mese con il numero e pulisci la stringa
            date_str_cleaned = re.sub(r'[^\d\s]', '', date_str_lower.replace(month_name, str(month_num)))
            parts = date_str_cleaned.split()
            if len(parts) >= 3:
                try:
                    # Tenta di costruire la data da giorno, mese, anno
                    day = int(parts[0])
                    month = int(parts[1])
                    year = int(parts[2])
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    continue # Prova il prossimo mese se il parsing fallisce
    
    logging.warning(f"Could not parse date string: '{date_str}'")
    return None

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Script Parameters ---
parser = argparse.ArgumentParser(description="Process auction catalogue images with LLM.")
parser.add_argument(
    "--image-folder",
    type=str,
    default="/home/vidiuser/Zeri/BO0624_83709",
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
title = args.title

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
OUTPUT_JSON_FILENAME = os.path.basename(os.path.normpath(IMAGE_FOLDER)) + ".json"

# --- Classification Normalization Helper ---
def normalize_classification(raw_classification: str) -> str:
    """
    Normalizes raw classification strings to predefined categories,
    prioritizing specific known misspellings, then using fuzzy matching.
    """
    if not isinstance(raw_classification, str):
        return "Unknown"

    raw_classification_lower = raw_classification.lower().strip()

    # Predefined correct classifications for a newspaper (changed to a list)
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

    # --- Passo 1: Correzioni specifiche per misspellings comuni e ricorrenti ---
    # Queste regole hanno la precedenza per catturare errori noti con certezza.
    if raw_classification_lower == "external fro cover":
        logging.info(f"Specific correction: '{raw_classification}' to 'External Front Cover'.")
        return "External Front Cover"
    if raw_classification_lower == "iernal back cover" or raw_classification_lower == "interal back cover":
        logging.info(f"Specific correction: '{raw_classification}' to 'Internal Back Cover'.")
        return "Internal Back Cover"
    if raw_classification_lower == "iernal fro cover": # Nuovo caso segnalato
        logging.info(f"Specific correction: '{raw_classification}' to 'Internal Front Cover'.")
        return "Internal Front Cover"
    
    # Estendi con altre varianti specifiche se emergono, usando "in" per catturare più variazioni
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


    # --- Passo 2: Corrispondenza diretta (case-insensitive) ---
    # Controlla se la stringa raw corrisponde esattamente (ignorando il caso) a una delle categorie corrette
    for correct_cat in correct_classifications:
        if raw_classification_lower == correct_cat.lower():
            logging.info(f"Direct match: '{raw_classification}' to '{correct_cat}'.")
            return correct_cat # Ritorna la versione con la capitalizzazione corretta
            
    # --- Passo 3: Fuzzy Matching come fallback più generale ---
    choices_lower = [c.lower() for c in correct_classifications]
    best_match_lower, score = process.extractOne(raw_classification_lower, choices_lower)

    SIMILARITY_THRESHOLD = 85 # Questa soglia potrebbe aver bisogno di essere calibrata in base ai risultati

    if score >= SIMILARITY_THRESHOLD:
        normalized_category = correct_classifications[choices_lower.index(best_match_lower)]
        if raw_classification != normalized_category: # Logga solo se c'è stata una correzione effettiva
            logging.info(f"Fuzzy normalized: '{raw_classification}' (score: {score}) to '{normalized_category}'.")
        return normalized_category
    else:
        logging.warning(f"Classification '{raw_classification}' (score: {score}) did not meet similarity threshold. Returning as is.")
        return raw_classification # Se non trova una corrispondenza sufficientemente buona, restituisce l'originale

# --- Pixtral Model Helper Functions ---
def load_pixtral_model():
    """Load Pixtral model and processor"""
    model_name = "mistral-community/pixtral-12b"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    return processor, model

def query_pixtral(prompt: str, pil_image: Image, processor, model):
    """Query Pixtral model with a prompt and image"""
    try:
        # Convert PIL image to base64 for the model
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{img_base64}"
        
        # Prepare messages in the required format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_url},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Generate response
        inputs = processor(
            messages, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        # Generate response
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=200,
            num_beams=3,  # Optimized for speed
            early_stopping=True
        )
        
        # Decode and clean output
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        return generated_text.strip()
    except Exception as e:
        logging.error(f"Pixtral query failed: {str(e)}")
        return "MODEL_ERROR"

# --- Main Processing Function ---
def run_pixtral_multimodal_inference():
    output_json_path = os.path.join(IMAGE_FOLDER, OUTPUT_JSON_FILENAME)
    final_structured_outputs = [] 
    last_valid_date = None
    processor, model = None, None

    logging.info("Script started.") 

    # Carica i dati esistenti se il file JSON esiste già
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

    try:
        # Load Pixtral model
        logging.info("Loading Pixtral model...")
        processor, model = load_pixtral_model()
        logging.info("Pixtral model loaded successfully.")

        logging.info(f"Scanning for images in: {IMAGE_FOLDER}")
        image_paths = []

        # Modifica il percorso per cercare le immagini nella sottocartella ridimensionati
        images_subfolder = os.path.join(IMAGE_FOLDER, "ridimensionati")

        if not os.path.exists(images_subfolder):
            logging.warning(f"Images subfolder does not exist: {images_subfolder}. Exiting.")
            return
        
        logging.info(f"Looking for images in: {images_subfolder}")
        
        logging.debug("Starting image file glob search...") 
        for ext in SUPPORTED_EXTENSIONS:
            current_glob_pattern = os.path.join(images_subfolder, f'*{ext}')
            logging.debug(f"Searching for pattern: {current_glob_pattern}") 
            image_paths.extend(glob.glob(current_glob_pattern))
        logging.debug(f"Finished image file glob search. Found {len(image_paths)} raw paths.") 

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
            
            # Inizializza il record dei risultati per questa pagina all'inizio del ciclo
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

                pil_image = None
                logging.debug(f"Attempting to open image: {img_path}") 
                try:
                    pil_image = Image.open(img_path).convert("RGB")
                    logging.debug(f"Successfully loaded image {filename} into PIL.")
                except Exception as e:
                    logging.error(f"Error loading image {filename}: {e}. Skipping this page.")
                    page_error_record = {
                        "filename": filename,
                        "classification": "Image Load Error",
                        "hasGraphics": False,
                        "details": f"Error loading image: {e}"
                    }
                    final_structured_outputs.append(page_error_record)
                    try:
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(final_structured_outputs, f, indent=4, ensure_ascii=False)
                    except Exception as save_e:
                        logging.error(f"Error saving partial output after image loading error for {filename}: {save_e}")
                    continue

                if pil_image is None:
                    logging.debug(f"PIL image object is None for {filename}. Skipping.") 
                    page_error_record = {
                        "filename": filename,
                        "classification": "Image Data Missing",
                        "hasGraphics": False,
                        "details": "PIL image object is None after loading attempt."
                    }
                    final_structured_outputs.append(page_error_record)
                    try:
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(final_structured_outputs, f, indent=4, ensure_ascii=False)
                    except Exception as save_e:
                        logging.error(f"Error saving partial output after PIL image is None for {filename}: {save_e}")
                    continue

                # --- PROMPT 1: CLASSIFICATION ONLY ---
                classification_prompt = (
                    f"The image is a single page of a newspaper titled \"{title}\". Your task is to classify the page type.\n"
                    f"Choose EXACTLY one of the following categories:\n"
                    f" - \"Front Page\": Must contain the main title (\"{title}\"), headlines, and usually a date.\n"
                    f" - \"Internal Page\": A standard page with articles.\n"
                    f" - \"Back Page\": The final page.\n"
                    f" - \"Full-Page Advertisement\"\n"
                    f" - \"Editorial Page\"\n"
                    f" - \"Sports Page\"\n"
                    f" - \"Classifieds Page\"\n"
                    f" - \"Blank Page\"\n"
                    f" - \"Reference\"\n\n"
                    f"Your response must be ONLY the chosen category string, with no other text or JSON formatting."
                )
                
                logging.info(f"Classifying page {current_page_num} with Pixtral...")
                raw_classification = query_pixtral(classification_prompt, pil_image, processor, model)
                
                classification = normalize_classification(raw_classification)
                logging.info(f"Page {current_page_num} classified as: {classification}")

                entry = {"filename": filename, "classification": classification, "issue_date": "N/A", "issue_number": "N/A", "hasGraphics": False, "page_number": "N/A"}

                # --- PROMPT 2: DATA EXTRACTION (ONLY FOR FRONT PAGES) ---
                if classification == "Front Page":
                    logging.info(f"Front Page detected. Proceeding to data extraction for page {current_page_num}.")
                    extraction_prompt = (
                        f"The image is the front page of a newspaper.\n"
                        f"Extract the following information from the header (testata):\n"
                        f"1. `issue_date`: Extract the date string EXACTLY as you see it (e.g., '6 Agosto 1946'). Do NOT reformat it.\n"
                        f"2. `issue_number`: Extract the issue number as seen (e.g., '312').\n\n"
                        f"If a value is not found, use \"N/A\".\n"
                        f"Respond with ONLY a JSON object, no other text.\n"
                        f"Example: {{\"issue_date\": \"6 Agosto 1946\", \"issue_number\": \"312\"}}"
                    )
                    
                    llm_raw_response = query_pixtral(extraction_prompt, pil_image, processor, model)
                    page_result_record["llm_raw_response"] = llm_raw_response

                    try:
                        # Estrai il JSON dalla risposta
                        json_match = re.search(r'\{[\s\S]*\}', llm_raw_response)
                        if json_match:
                            extracted_data = json.loads(json_match.group(0))
                            entry.update(extracted_data)
                            logging.info(f"Successfully extracted data for page {current_page_num}: {extracted_data}")
                        else:
                            logging.warning(f"No JSON object found in extraction response for page {current_page_num}.")
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to decode JSON from extraction response for page {current_page_num}: {e}")

                # --- VALIDAZIONE A VALLE (ORA APPLICATA AI DATI ESTRATTI) ---
                original_date_str = entry.get("issue_date", "N/A")
                extracted_date_obj = parse_italian_date(original_date_str)

                if extracted_date_obj:
                    try:
                        min_date = datetime.strptime(args.start_date, "%Y-%m-%d")
                        max_date = datetime.strptime(args.end_date, "%Y-%m-%d")

                        # Controlla l'intervallo globale
                        if not (min_date <= extracted_date_obj <= max_date):
                            logging.warning(f"Page {current_page_num}: Date {original_date_str} -> {extracted_date_obj.strftime('%d/%m/%Y')} is outside the valid range. Resetting.")
                            entry["issue_date"] = "N/A"
                        # Controlla la coerenza cronologica
                        elif last_valid_date and extracted_date_obj < last_valid_date:
                            logging.warning(f"Page {current_page_num}: Date {original_date_str} -> {extracted_date_obj.strftime('%d/%m/%Y')} is earlier than last valid date. Resetting.")
                            entry["issue_date"] = "N/A"
                        else:
                            # Se la data è valida, la normalizziamo e aggiorniamo lo stato
                            entry["issue_date"] = extracted_date_obj.strftime('%d/%m/%Y')
                            last_valid_date = extracted_date_obj
                            logging.info(f"Page {current_page_num}: Valid date found and normalized: {entry['issue_date']}")
                    except Exception as e:
                        logging.error(f"Error during date range validation: {e}")
                        entry["issue_date"] = "N/A"
                else:
                    # Se il parsing fallisce o la data è "N/A", ci assicuriamo che sia "N/A"
                    if original_date_str != "N/A":
                         logging.warning(f"Page {current_page_num}: Could not parse date '{original_date_str}'. Resetting to 'N/A'.")
                    entry["issue_date"] = "N/A"
                
                # Assegna l'entry finale al record della pagina
                page_result_record["parsed_llm_output"] = entry
                logging.info(f"Final data for page {current_page_num}: {entry}")

            except Exception as e: # Catches errors during processing
                logging.error(f"Critical error during processing for page {current_page_num}: {e}", exc_info=True)
                page_result_record["parsed_llm_output"] = {
                    "filename": filename,
                    "classification": "Processing Critical Error",
                    "hasGraphics": False,
                    "details": str(e)
                }

            # --- Aggiorna e salva il file JSON ad ogni iterazione ---
            logging.debug(f"State before final append for {filename}: {page_result_record!r}")

            # Rimuovi il record esistente per questa pagina, se presente, per aggiornarlo
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

        # Fine del ciclo for
        
        # Stampa dei risultati finali
        logging.info("\n" + "="*80)
        logging.info("FINAL COLLATED RESULTS FROM ALL PAGES")
        logging.info("="*80)

        if final_structured_outputs:
            logging.info(f"Processing complete. All results saved to: {output_json_path}")
        else:
            logging.info("No outputs were generated from any page.")
        logging.info("="*80)

    finally:
        # Clean up resources
        logging.info("Starting final cleanup process.") 
        if processor is not None:
            del processor
        if model is not None:
            del model
        import gc
        gc.collect()
        logging.info("Model resources released.") 


if __name__ == "__main__":
    run_pixtral_multimodal_inference()