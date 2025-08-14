import os
import cv2
import pytesseract
import logging
from datetime import datetime
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
IMG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'imgs'))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
DEBUG_DIR = os.path.join(ROOT_DIR, 'debug')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Set up logging
log_filename = datetime.now().strftime(f"{LOG_DIR}/masthead_classification.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Target text (case-insensitive)
TARGET_TEXT = "giornale dell'emilia"
TEXT_HEIGHT_RATIO = 1/12
TOP_CROP_RATIO = 0.15
RECTANGLE_AR_MIN = 3.0
RECTANGLE_AR_MAX = 8.0
DEBUG = True  # Set to False to disable visual debugging

def save_debug_image(image, filename, prefix, debug_info=None):
    """Save annotated debug image to debug directory"""
    if not DEBUG:
        return
    
    debug_img = image.copy()
    img_height = debug_img.shape[0]
    
    # Draw top crop region
    top_crop_y = int(img_height * TOP_CROP_RATIO)
    cv2.rectangle(debug_img, (0, 0), (debug_img.shape[1], top_crop_y), (255, 0, 0), 2)
    cv2.putText(debug_img, f"Top {int(TOP_CROP_RATIO*100)}%", 
                (10, top_crop_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if debug_info:
        # Draw OCR results
        if 'ocr_data' in debug_info:
            for i in range(len(debug_info['ocr_data']['text'])):
                if debug_info['ocr_data']['text'][i].strip():
                    x = debug_info['ocr_data']['left'][i]
                    y = debug_info['ocr_data']['top'][i]
                    w = debug_info['ocr_data']['width'][i]
                    h = debug_info['ocr_data']['height'][i]
                    
                    # Draw word bounding box
                    color = (0, 255, 0) if TARGET_TEXT in debug_info['ocr_data']['text'][i].lower() else (0, 0, 255)
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(debug_img, debug_info['ocr_data']['text'][i], 
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw detected contours
        if 'contours' in debug_info:
            for cnt in debug_info['contours']:
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 255), 2)
                
        # Draw rectangle candidates
        if 'rect_candidates' in debug_info:
            for rect in debug_info['rect_candidates']:
                x, y, w, h = rect
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 3)
                cv2.putText(debug_img, "Candidate", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Draw detected rectangles
        if 'detected_rects' in debug_info:
            for rect in debug_info['detected_rects']:
                x, y, w, h, text = rect
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(debug_img, f"Detected: {text[:20]}", (x, y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add classification result
    result = "masthead" if debug_info.get('is_masthead', False) else "non-masthead"
    cv2.putText(debug_img, f"Result: {result}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if result == "non-masthead" else (0, 255, 0), 2)
    
    # Save to debug directory
    debug_path = os.path.join(DEBUG_DIR, f"{prefix}_{filename}")
    cv2.imwrite(debug_path, debug_img)
    logger.info(f"Saved debug image: {debug_path}")

def process_image(image_path):
    """Process image and classify as masthead or not"""
    debug_info = {
        'is_masthead': False,
        'rect_candidates': [],
        'detected_rects': [],
        'contours': []
    }
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return False, debug_info
    
    # Preprocessing
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Noise reduction
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=40)  # Contrast enhancement
    
    img_height = img.shape[0]
    top_crop = int(img_height * TOP_CROP_RATIO)
    img_top = img[0:top_crop, :]
    
    # OCR processing
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghilmnopqrstuvz.\'"'
    try:
        data = pytesseract.image_to_data(
            img_top, 
            lang='ita',
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        debug_info['ocr_data'] = data
        
        # Log OCR results
        logger.info(f"OCR results for {os.path.basename(image_path)}:")
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                logger.info(f"  Word: '{data['text'][i]}' at ({data['left'][i]}, {data['top'][i]}) "
                            f"size: {data['width'][i]}x{data['height'][i]}, conf: {data['conf'][i]}")
    except pytesseract.TesseractError as e:
        logger.error(f"OCR failed for {image_path}: {str(e)}")
        return False, debug_info
    
    # Group words by line
    lines = {}
    for i in range(len(data['text'])):
        if not data['text'][i].strip():
            continue
        line_num = data['line_num'][i]
        if line_num not in lines:
            lines[line_num] = []
        lines[line_num].append({
            'text': data['text'][i],
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i]
        })
    
    # Text-based classification
    text_found = False
    for line_num, words in lines.items():
        line_text = ' '.join(w['text'] for w in words).lower()
        if TARGET_TEXT in line_text:
            line_top = min(w['top'] for w in words)
            line_height = max(w['top'] + w['height'] for w in words) - line_top
            
            # Log potential target text
            logger.info(f"Found target text: '{line_text}' in line {line_num} "
                        f"at position ({min(w['left'] for w in words)}, {line_top}), "
                        f"height: {line_height}px")
            
            if line_top <= 0.25 * img_height and line_height >= img_height * TEXT_HEIGHT_RATIO:
                text_found = True
                logger.info(f"Text meets criteria: top position {line_top} <= {0.25 * img_height} "
                            f"and height {line_height} >= {img_height * TEXT_HEIGHT_RATIO}")
                break
            else:
                logger.info(f"Text does NOT meet criteria: top position {line_top} > {0.25 * img_height} "
                            f"or height {line_height} < {img_height * TEXT_HEIGHT_RATIO}")
    
    if text_found:
        debug_info['is_masthead'] = True
        return True, debug_info
    
    # Visual feature detection
    gray = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Store contours for debugging
    debug_info['contours'] = contours
    
    # Log contour information
    logger.info(f"Found {len(contours)} contours in {os.path.basename(image_path)}")
    
    for cnt in contours:
        # Rectangle detection
        if len(cnt) < 4:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Check aspect ratio
        aspect_ratio = w / float(h)
        if aspect_ratio < RECTANGLE_AR_MIN or aspect_ratio > RECTANGLE_AR_MAX:
            logger.info(f"Rectangle aspect ratio {aspect_ratio:.2f} outside range "
                        f"[{RECTANGLE_AR_MIN:.2f}, {RECTANGLE_AR_MAX:.2f}] - skipping")
            continue
            
        # Check text size
        if h < img_height * TEXT_HEIGHT_RATIO:
            logger.info(f"Rectangle height {h} < min required {img_height * TEXT_HEIGHT_RATIO} - skipping")
            continue
            
        # Add to candidates
        debug_info['rect_candidates'].append((x, y, w, h))
        logger.info(f"Rectangle candidate at ({x}, {y}) size {w}x{h}, aspect: {aspect_ratio:.2f}")
        
        # Check for "L." prefix
        roi = img_top[y:y+h, x:x+w]
        try:
            roi_text = pytesseract.image_to_string(roi, lang='ita').lower().strip()
            logger.info(f"  ROI text: '{roi_text}'")
            
            if "l." in roi_text:
                debug_info['detected_rects'].append((x, y, w, h, roi_text))
                logger.info(f"Found 'L.' prefix in rectangle at ({x}, {y})")
                debug_info['is_masthead'] = True
                return True, debug_info
        except pytesseract.TesseractError as e:
            logger.error(f"OCR failed for ROI in {image_path}: {str(e)}")
    
    return False, debug_info

def main():
    """Main classification function"""
    logger.info("Starting masthead classification")
    print(f"Processing images in {IMG_DIR}. Logs: {LOG_DIR}, Output: {OUTPUT_DIR}, Debug: {DEBUG_DIR}")
    
    results = []
    
    for img_file in os.listdir(IMG_DIR):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            img_path = os.path.join(IMG_DIR, img_file)
            logger.info(f"Processing image: {img_file}")
            
            is_masthead, debug_info = process_image(img_path)
            result = "masthead" if is_masthead else "non-masthead"
            
            # Save debug image
            save_debug_image(
                cv2.imread(img_path), 
                img_file, 
                "masthead" if is_masthead else "nonmasthead",
                debug_info
            )
            
            # Add to results list
            results.append({
                "filename": img_file,
                "classification": result
            })
            
            logger.info(f"Classification result for {img_file}: {result}")
            print(f"{img_file}: {result}")
    
    # Save results to JSON
    output_path = os.path.join(OUTPUT_DIR, f"classification_results_2.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nClassification complete! Results saved to: {output_path}")
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()