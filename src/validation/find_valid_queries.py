"""
Find valid (non-error) queries that appear in both offline and merged summary files.

Identifies queries where is_error_summary returns False for both files,
meaning the summary generation succeeded in both cases.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Union


def is_error_summary(summary: Union[List, Dict], is_merged: bool = False) -> bool:
    """
    Detect error summaries based on error patterns.
    
    Args:
        summary: Summary to check (can be list of claims or dict with summary field)
        is_merged: True if from merged summaries, False if from offline summaries
        
    Returns:
        True if summary contains error patterns, False otherwise
    """
    # Handle different summary formats
    if isinstance(summary, dict):
        summary_list = summary.get("summary", summary.get("summaries", []))
    else:
        summary_list = summary
    
    if not isinstance(summary_list, list):
        return False
    
    # Check each claim's perspectives for error patterns
    for claim in summary_list:
        if not isinstance(claim, dict):
            continue
        perspectives = claim.get("perspectives", [])
        for perspective in perspectives:
            if not isinstance(perspective, dict):
                continue
            text = perspective.get("text", "")
            
            if is_merged:
                # Merged summary error patterns
                if "JSON parse errors 5+ times (prompt too long)" in text:
                    return True
                if "All 10 generation attempts failed" in text:
                    return True
            else:
                # Offline summary error patterns
                if text.startswith("Error generating summary:"):
                    return True
    
    return False


def load_summaries(file_path: str) -> List[Dict]:
    """Load summaries from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_valid_queries_offline(offline_file: str) -> List[Dict]:
    """
    Find valid (non-error) queries from offline summaries.
    
    Returns:
        List of all valid entries (including duplicates) with id and query
    """
    summaries = load_summaries(offline_file)
    valid_queries = []
    
    for entry in summaries:
        query_text = entry.get("query", "").strip()
        if not query_text:
            continue
        
        summary = entry.get("summary", [])
        # Check if this is NOT an error summary
        if not is_error_summary(summary, is_merged=False):
            valid_queries.append({
                "id_offline": entry.get("id", ""),
                "query": query_text
            })
    
    return valid_queries


def find_valid_queries_merged(merged_file: str) -> List[Dict]:
    """
    Find valid (non-error) queries from merged summaries.
    
    Returns:
        List of all valid entries (including duplicates) with id and query
    """
    summaries = load_summaries(merged_file)
    valid_queries = []
    
    for entry in summaries:
        query_text = entry.get("query", "").strip()
        if not query_text:
            continue
        
        summary = entry.get("summary", [])
        # Check if this is NOT an error summary
        if not is_error_summary(summary, is_merged=True):
            valid_queries.append({
                "id_merged": entry.get("id", ""),
                "query": query_text
            })
    
    return valid_queries


def find_intersection(offline_file: str, merged_file: str, deduplicate: bool = True) -> List[Dict]:
    """
    Find queries that are valid (non-error) in BOTH offline and merged files.
    
    Args:
        offline_file: Path to offline summaries file
        merged_file: Path to merged summaries file
        deduplicate: If True, only keep one instance of each duplicate query string.
                    If False, includes all duplicates (original behavior).
    
    Returns:
        List of dicts with id_offline, id_merged, and query fields
    """
    # Get valid queries from each file (as lists to preserve duplicates)
    valid_offline = find_valid_queries_offline(offline_file)
    valid_merged = find_valid_queries_merged(merged_file)
    
    # Group by query text for matching
    from collections import defaultdict
    offline_by_query = defaultdict(list)
    merged_by_query = defaultdict(list)
    
    for entry in valid_offline:
        offline_by_query[entry["query"]].append(entry["id_offline"])
    
    for entry in valid_merged:
        merged_by_query[entry["query"]].append(entry["id_merged"])
    
    # Find all valid pairs
    intersection = []
    all_queries = set(offline_by_query.keys()) & set(merged_by_query.keys())
    
    # Track seen queries if deduplicating
    seen_queries = set() if deduplicate else None
    
    for query_text in all_queries:
        # Skip if we've already seen this query and deduplicating
        if deduplicate and query_text in seen_queries:
            continue
        
        offline_ids = offline_by_query[query_text]
        merged_ids = merged_by_query[query_text]
        
        # Only take the first pair if deduplicating, otherwise take all pairs
        if deduplicate:
            num_pairs = 1
            seen_queries.add(query_text)
        else:
            # Create pairs - only create as many pairs as the minimum count
            # This ensures we only pair entries that actually exist in both files
            num_pairs = min(len(offline_ids), len(merged_ids))
        
        for i in range(num_pairs):
            offline_id = offline_ids[i]
            merged_id = merged_ids[i]
            
            intersection.append({
                "id_offline": offline_id,
                "id_merged": merged_id,
                "query": query_text
            })
    
    return intersection


def main():
    """Main function to find and save valid queries."""
    # File paths
    offline_file = "results/offline-summaries-JSON-enforced/results-10-offline-0-online-tfidf-20251214_222854.json"
    merged_file = "results/merged-summaries/results-merged-10-20251215_082353.json"
    
    # Find intersection of valid queries (deduplicated by default)
    print("Loading offline summaries...")
    valid_queries = find_intersection(offline_file, merged_file, deduplicate=True)
    
    print(f"Found {len(valid_queries)} unique queries that are valid in both files")
    
    # Create output directory if it doesn't exist
    output_dir = "data/valid-queries"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"valid-k-10-queries-{timestamp}.json")
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_queries, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    main()

