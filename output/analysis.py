import json
import argparse
import os

def analyze_frontpage_dates(json_path):
    """
    Analyzes the output JSON file to count how many front pages have valid dates
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Create a dictionary to store highest confidence entry for each filename
    unique_pages = {}
    for entry in data:
        filename = entry.get('filename')
        confidence = entry.get('classification_confidence', 0)
        
        # If we haven't seen this filename or if this entry has higher confidence
        if filename not in unique_pages or confidence > unique_pages[filename].get('classification_confidence', 0):
            unique_pages[filename] = entry

    # Convert back to list and filter front pages
    front_pages = [page for page in unique_pages.values() 
                  if page.get("classification") == "Front Page"]
                  #and page.get("classification_confidence", 0) > 0.35]
    
    total_front_pages = len(front_pages)
    
    # Count pages with valid dates
    dated_front_pages = [page for page in front_pages 
                        if page.get("issue_date") and page.get("issue_date") != "N/A"]
    
    valid_dates_count = len(dated_front_pages)
    
    # Calculate success rate
    success_rate = (valid_dates_count / total_front_pages * 100) if total_front_pages > 0 else 0

    print("\nEnhanced Front Page Date Extraction Report")
    print("=" * 50)
    print(f"Total unique front pages (verified): {total_front_pages}")
    print(f"Front pages with valid dates: {valid_dates_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print("=" * 50)
    
    # Print details of undated pages
    if valid_dates_count < total_front_pages:
        undated = []
        for page in front_pages:
            if page.get("issue_date") == "N/A":
                error_msg = page.get("processing_errors", [])
                undated.append(f"{page['filename']} (conf: {page.get('classification_confidence', 0):.2f})")
                if error_msg:
                    undated[-1] += f"\n   Error: {'; '.join(error_msg)}"
        
        print("\nFront pages without dates:")
        for i, entry in enumerate(undated, 1):
            print(f"{i}. {entry}")

if __name__ == "__main__":
    default_json = 'script_lu.json'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_path = os.path.join(script_dir, default_json)

    parser = argparse.ArgumentParser(description='Analyze newspaper JSON output')
    parser.add_argument('--json', default=default_json_path, 
                       help='Path to JSON output file (default: extraction_output.json)')
    args = parser.parse_args()
    
    analyze_frontpage_dates(args.json)