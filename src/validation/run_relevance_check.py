"""
Batch Web Document Relevance Checker

Processes web documents from data/web/ directory and outputs relevance classifications
to data/valid-web/ directory. Each document is classified as Relevant (R) or Not Relevant (NR).

Usage:
    # Process a single file
    python src/validation/run_relevance_check.py --input web-5.json --limit 3
    
    # Process all web-*.json files
    python src/validation/run_relevance_check.py
    
    # Dry run (process selected queries without saving)
    python src/validation/run_relevance_check.py --input web-5.json --limit 3 --dry-run
"""

import os
import json
import argparse
from pathlib import Path
from relevance_checker import check_relevance


def process_web_file(input_path: str, output_path: str, limit: int = None, 
                     dry_run: bool = False):
    """
    Process a single web document file and output relevance classifications.
    
    Args:
        input_path: Path to input JSON file (e.g., data/web/web-5.json)
        output_path: Path to output JSON file (e.g., data/valid-web/valid-web-5.json)
        limit: Process only first N queries (None = all)
        dry_run: If True, process and print selected queries (respecting limit) without saving
    """
    # Load input data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nProcessing {input_path}")
    print(f"Total queries: {len(data)}")
    if limit:
        print(f"Limiting to first {limit} queries")
    
    # Process queries
    output_data = []
    queries_to_process = data[:limit] if limit else data
    
    for idx, item in enumerate(queries_to_process):
        query = item.get("query", "")
        web_docs = item.get("web_docs", {})
        results = web_docs.get("results", [])
        
        print(f"\n[{idx + 1}/{len(queries_to_process)}] Query: {query}")
        print(f"  Documents: {len(results)}")
        
        # Get relevance classifications
        classifications = check_relevance(query, results)
        
        print(f"  Classifications: {classifications}")
        r_count = sum(1 for v in classifications.values() if v == "R")
        nr_count = sum(1 for v in classifications.values() if v == "NR")
        print(f"  Relevant: {r_count}, Not Relevant: {nr_count}")
        
        # Add relevance field to each document
        for doc in results:
            doc_id = str(doc['id'])
            doc['relevance'] = classifications.get(doc_id, "NR")
        
        # Preserve original structure
        output_data.append(item)
    
    # Dry run: process all selected queries but do not save
    if dry_run:
        print("\nDry run complete. Processed selected queries without saving.")
        print(f"Would save to: {output_path}")
        return

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify web documents as Relevant (R) or Not Relevant (NR) using GPT-5-Nano"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        help="Specific file to process (e.g., web-5.json). If not specified, processes all web-*.json files"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        help="Process only first N queries (useful for testing)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Process and print selected queries without saving (for testing)"
    )
    
    args = parser.parse_args()
    
    web_dir = Path("data/web")
    valid_web_dir = Path("data/valid-web")
    
    # Determine which files to process
    if args.input:
        # Process single file
        input_files = [args.input]
    else:
        # Process all web-*.json files
        input_files = sorted([f.name for f in web_dir.glob("web-*.json")])
    
    if not input_files:
        print("No files to process. Check that data/web/ contains web-*.json files.")
        return
    
    print(f"Files to process: {input_files}")
    
    # Process each file
    for filename in input_files:
        input_path = web_dir / filename
        # Convert web-5.json -> valid-web-5.json
        output_filename = filename.replace("web-", "valid-web-")
        output_path = valid_web_dir / output_filename
        
        try:
            process_web_file(
                str(input_path), 
                str(output_path),
                limit=args.limit,
                dry_run=args.dry_run
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
