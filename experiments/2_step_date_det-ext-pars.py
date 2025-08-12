import os

#Step 2: Date Localization with Object Detection
import cv2
import numpy as np
from craft_text_detector import Craft
from 1_step_identify_frontpages import extract_frontpages

def locate_date_region(image_path):
    craft = Craft(crop_type="box", output_dir=None)
    prediction = craft.detect_text(image_path)
    boxes = prediction["boxes"]
    
    # Prioritize top-right regions for date search
    right_boxes = [b for b in boxes if b[0][0] > image.shape[1]*0.7]
    top_right = sorted(right_boxes, key=lambda b: b[0][1])[:5]
    
    return top_right

#Step 3: Text Extraction & Date Parsing
import pytesseract
import re
from dateparser.search import search_dates

def extract_date(image_path, boxes):
    img = cv2.imread(image_path)
    date_candidates = []
    
    for box in boxes:
        x,y,w,h = cv2.boundingRect(box)
        roi = img[y:y+h, x:x+w]
        
        # Preprocessing for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Italian OCR with config
        text = pytesseract.image_to_string(thresh, lang='ita', config='--psm 11')
        date_candidates.append(text)
    
    # Date parsing with language context
    for text in date_candidates:
        dates = search_dates(text, languages=['it'], settings={'PREFER_DAY_OF_MONTH': True})
        if dates:
            return dates[0][1].strftime('%Y-%m-%d')
    
    return None

#Step 4: Pipeline

def process_collection(xml_dir, img_dir):
    frontpages = extract_frontpages(xml_dir)
    catalog = []
    
    for item in frontpages:
        img_path = os.path.join(img_dir, item['file'])
        boxes = locate_date_region(img_path)
        date = extract_date(img_path, boxes)
        
        if date:
            catalog.append({
                'file': item['file'],
                'title': item['title'],
                'date': date,
                'status': 'success'
            })
        else:
            catalog.append({
                'file': item['file'],
                'title': item['title'],
                'date': '',
                'status': 'failed'
            })
    
    return catalog