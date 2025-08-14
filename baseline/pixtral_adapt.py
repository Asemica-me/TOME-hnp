import os
from dotenv import load_dotenv
import glob
import logging
import json
import re
from PIL import Image
import argparse
from datetime import datetime
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

load_dotenv()

# ---------- Logging ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
logs_folder = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(logs_folder, exist_ok=True)
log_file_path = os.path.join(logs_folder, "pixtral.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)]
)

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Identify main pages of Giornale dell'Emilia newspaper.")
parser.add_argument("--image-folder", type=str, default=os.path.join(PROJECT_ROOT, "imgs"),
                    help="Folder containing input images (default: ./img)")
parser.add_argument("--output-folder", type=str, default=os.path.join(PROJECT_ROOT, "output"),
                    help="Folder for JSON output (default: ./output)")
parser.add_argument("--start-date", type=str, default="1946-07-21", help="Start of valid date range YYYY-MM-DD")
parser.add_argument("--end-date", type=str, default="1947-05-15", help="End of valid date range YYYY-MM-DD")
parser.add_argument("--title", type=str, default="Giornale dell'Emilia", help="Titolo della testata")
parser.add_argument("--max-new-tokens", type=int, default=10, help="Max tokens for classification response")
parser.add_argument("--max-image-size", type=int, default=768, help="Max image side length (default: 768)")
parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], default="none",
                    help="Quantization mode for GPU memory savings")
parser.add_argument("--skip-processed", action="store_true",
                    help="Skip images already present in output JSON")
args = parser.parse_args()

# ---------- Env / Model setup ----------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HUGGINGFACE token not found. Add HF_TOKEN to your .env")

PIXTRAL_MODEL_ID = os.getenv("PIXTRAL_MODEL_ID", "mistral-community/pixtral-12b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32

# Optimize CPU threads for non-GPU environments
if DEVICE == "cpu":
    try:
        torch.set_num_threads(min(4, os.cpu_count()))
    except Exception:
        pass

logging.info(f"Loading model: {PIXTRAL_MODEL_ID} on {DEVICE} with dtype={DTYPE}")

# ===== QUANTIZATION SUPPORT =====
if args.quantization != "none" and DEVICE == "cuda":
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=(args.quantization == "4bit"),
        load_in_8bit=(args.quantization == "8bit"),
        bnb_4bit_compute_dtype=DTYPE,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        PIXTRAL_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        token=HF_TOKEN
    )
    processor = AutoProcessor.from_pretrained(PIXTRAL_MODEL_ID, token=HF_TOKEN)
else:
    processor = AutoProcessor.from_pretrained(PIXTRAL_MODEL_ID, token=HF_TOKEN)
    model = LlavaForConditionalGeneration.from_pretrained(
        PIXTRAL_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        token=HF_TOKEN
    ).to(DEVICE)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "right"

model.eval()

# ---------- Paths ----------
IMAGE_FOLDER = args.image_folder
OUTPUT_FOLDER = args.output_folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
OUTPUT_JSON_FILENAME = "main_pages.json"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_JSON_FILENAME)

# ---------- Helpers ----------
def parse_italian_date(date_str: str) -> datetime | None:
    """Parse Italian date strings into datetime objects"""
    if not date_str or date_str == "N/A":
        return None
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except ValueError:
        pass
    
    italian_months = {
        'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4, 'maggio': 5, 'giugno': 6,
        'luglio': 7, 'agosto': 8, 'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12
    }
    
    for month_name, month_num in italian_months.items():
        if month_name in date_str.lower():
            parts = re.findall(r'\d+', date_str)
            if len(parts) >= 3:
                try:
                    day = int(parts[0]); month = int(parts[1]); year = int(parts[2])
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    continue
    return None

# ---------- Optimized Vision Call ----------
def call_vision_model(prompt: str, pil_image: Image.Image, max_new_tokens: int = 256) -> str:
    """Properly formatted Pixtral call with chat template"""
    # Downscale for faster processing
    w, h = pil_image.size
    scale = min(args.max_image_size / max(w, h), 1.0)
    if scale < 1.0:
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Get the model's expected image token
    image_tok = getattr(processor, "image_token", "<image>")
    
    # Build messages with the image token
    messages = [
        {
            "role": "user",
            "content": f"{image_tok}\n{prompt}"
        }
    ]
    
    # Apply chat template to format messages correctly
    template_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Ensure image token is present
    if image_tok not in template_text:
        template_text = f"{image_tok}\n" + template_text
    
    inputs = processor(
        text=template_text,
        images=[pil_image],
        return_tensors="pt",
        padding=True
    ).to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            temperature=0.01,
        )

    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

def is_main_page(pil_image: Image.Image) -> bool:
    """Determine if image is a main page by checking for the cubital title"""
    # Focused prompt to identify main pages
    prompt = (
        "Is this the main page of 'Giornale dell'Emilia' newspaper? "
        "Main pages MUST show the large cubital title 'GIORNALE DELL'EMILIA' in the upper section. "
        "Answer ONLY with 'yes' or 'no'."
    )
    
    response = call_vision_model(prompt, pil_image, max_new_tokens=args.max_new_tokens).lower()
    return "yes" in response

# ---------- Main Processing ----------
def process_images():
    # Load existing results to skip processed files
    processed_files = {}
    if args.skip_processed and os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                processed_files = {item['filename']: item for item in json.load(f)}
            logging.info(f"Loaded {len(processed_files)} existing records")
        except Exception as e:
            logging.warning(f"Could not load existing output: {e}")

    # Scan images
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, f"*{ext}")))
    image_paths.sort()
    
    results = []
    main_page_count = 0
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Skip already processed files
        if filename in processed_files:
            results.append(processed_files[filename])
            if processed_files[filename]["is_main_page"]:
                main_page_count += 1
            continue
        
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            results.append({
                "filename": filename,
                "is_main_page": False,
                "error": f"Image load error: {str(e)}"
            })
            continue
        
        try:
            # Step 1: Check if it's a main page
            is_main = is_main_page(pil_image)
            
            record = {
                "filename": filename,
                "is_main_page": is_main,
                "issue_date": "N/A",
                "issue_number": "N/A"
            }
            
            # Step 2: Extract metadata ONLY for main pages
            if is_main:
                main_page_count += 1
                logging.info(f"Main page detected: {filename}")
                
                # Extract date and issue number
                extraction_prompt = (
                    "Extract publication date and issue number from this newspaper's main page. "
                    "Date format: DD/MM/YYYY. "
                    "Respond ONLY with JSON: {\"issue_date\": \"...\", \"issue_number\": \"...\"}"
                )
                
                template_text = f"<image>\n{extraction_prompt}"
                inputs = processor(
                    text=template_text,
                    images=[pil_image],
                    return_tensors="pt",
                    padding=True
                ).to(DEVICE, dtype=DTYPE)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=96,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                
                # Extract JSON from response
                try:
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        extracted = json.loads(json_match.group(0))
                        record.update(extracted)
                except (json.JSONDecodeError, re.error) as e:
                    logging.warning(f"JSON extraction failed for {filename}: {e}")
            
            results.append(record)
            
        except Exception as e:
            logging.error(f"Processing error for {filename}: {e}", exc_info=True)
            results.append({
                "filename": filename,
                "is_main_page": False,
                "error": str(e)
            })
        
        # Save incremental results
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Final report
    logging.info(f"Processing complete. Found {main_page_count} main pages.")
    logging.info(f"Results saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    process_images()