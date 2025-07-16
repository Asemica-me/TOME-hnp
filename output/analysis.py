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

    # Filter front pages
    front_pages = [entry for entry in data 
                   if entry.get("classification") == "Front Page"]
    
    total_front_pages = len(front_pages)
    
    # Count pages with valid dates
    dated_front_pages = [page for page in front_pages 
                        if page.get("issue_date", "N/A") != "N/A"]
    
    valid_dates_count = len(dated_front_pages)
    
    # Calculate success rate
    success_rate = (valid_dates_count / total_front_pages * 100) if total_front_pages > 0 else 0

    print("\nFront Page Date Extraction Report")
    print("=" * 40)
    print(f"Total front pages: {total_front_pages}")
    print(f"Front pages with valid dates: {valid_dates_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print("=" * 40)
    
    # Print details of undated pages
    if valid_dates_count < total_front_pages:
        undated = [page['filename'] for page in front_pages 
                  if page.get("issue_date") == "N/A"]
        print("\nFront pages without dates:")
        for i, filename in enumerate(undated, 1):
            print(f"{i}. {filename}")

if __name__ == "__main__":
    default_json = 'imgs_results.json'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_path = os.path.join(script_dir, default_json)

    parser = argparse.ArgumentParser(description='Analyze newspaper JSON output')
    parser.add_argument('--json', default=default_json_path, 
                       help='Path to JSON output file (default: imgs_results.json)')
    args = parser.parse_args()
    
    analyze_frontpage_dates(args.json)