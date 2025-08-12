import json
import argparse
import os
import re

def analyze_frontpage_dates(json_path):
    """
    Analyzes the output JSON from multimodal classification pipeline
    to count how many front pages have valid dates
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Filter only front pages - no deduplication needed since each filename is unique
    front_pages = [page for page in data 
                  if page.get("classification") == "Front Page"]
    
    total_front_pages = len(front_pages)
    
    # Count pages with valid dates (non-empty and not "N/A")
    dated_front_pages = [page for page in front_pages 
                        if page.get("issue_date") and page.get("issue_date") != "N/A"]
    
    valid_dates_count = len(dated_front_pages)
    
    # Calculate success rate
    success_rate = (valid_dates_count / total_front_pages * 100) if total_front_pages > 0 else 0

    print("\nFront Page Date Extraction Report")
    print("=" * 50)
    print(f"Total unique front pages: {total_front_pages}")
    print(f"Front pages with valid dates: {valid_dates_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print("=" * 50)
    
    # Print details of undated pages
    if valid_dates_count < total_front_pages:
        undated = []
        for page in front_pages:
            if not page.get("issue_date") or page.get("issue_date") == "N/A":
                # Extract any available error details
                error_msg = page.get("details", "")
                if not error_msg:
                    # Try to infer from date parsing issues
                    date_candidate = re.search(r'[a-zA-Z0-9]{3,}', page.get("llm_raw_response", "") or "")
                    if date_candidate:
                        error_msg = f"Unparseable date: {date_candidate.group(0)}"
                
                undated.append(page['filename'])
                if error_msg:
                    undated[-1] += f" | {error_msg}"
        
        print("\nFront pages without dates:")
        for i, entry in enumerate(undated, 1):
            print(f"{i}. {entry}")

if __name__ == "__main__":
    default_json = 'paolo_extraction.json'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_path = os.path.join(script_dir, default_json)

    parser = argparse.ArgumentParser(description='Analyze newspaper JSON output')
    parser.add_argument('--json', default=default_json_path, 
                       help='Path to JSON output file (default: output/paolo_extraction.json)')
    args = parser.parse_args()
    
    analyze_frontpage_dates(args.json)