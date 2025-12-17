#!/usr/bin/env python3
"""
Simple script to compute basic dataset statistics for ThePerspective dataset.
"""

import sys
from pathlib import Path
from collections import Counter
import statistics

# Add project root to path so imports work when running directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_theperspective_dataset, load_theperspective_evidence


def count_words(text):
    """Simple word count by splitting on whitespace."""
    return len(text.split())


def main():
    # Load dataset
    dataset = load_theperspective_dataset("data/theperspective")
    evidence = load_theperspective_evidence("data/theperspective")
    
    # Setup output file
    output_dir = project_root / "results" / "dataset-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset-metrics.md"
    
    # Basic statistics
    num_queries = len(dataset)
    num_documents = len(evidence)
    
    # Word counts
    total_words = sum(count_words(doc["content"]) for doc in evidence)
    avg_words_per_doc = total_words / num_documents if num_documents > 0 else 0
    
    # Collect output
    output_lines = []
    
    def write(text=""):
        output_lines.append(text)
    
    # Print basic statistics
    write("## Dataset Statistics")
    write()
    write(f"- **Number of queries:** {num_queries}")
    write(f"- **Number of evidence documents:** {num_documents:,}")
    write(f"- **Total words across all documents:** {total_words:,}")
    write(f"- **Average words per document:** {avg_words_per_doc:.1f}")
    write()
    
    # Data Quality Metrics
    write("## Data Quality Metrics")
    write()
    queries = [entry['query'] for entry in dataset]
    unique_queries = set(queries)
    num_unique = len(unique_queries)
    query_counts = Counter(queries)
    duplicates = {q: c for q, c in query_counts.items() if c > 1}
    num_duplicates = len(duplicates)
    
    write(f"- **Unique queries:** {num_unique}")
    write(f"- **Duplicate queries:** {num_duplicates}")
    if num_duplicates > 0:
        write()
        write("**Sample duplicate queries (each appearing 2 times):**")
        for i, (query, count) in enumerate(list(duplicates.items())[:5], 1):
            write(f"{i}. \"{query}\"")
    write()
    
    # Task-Specific Statistics
    write("## Task-Specific Statistics")
    write()
    
    # Perspective statistics
    perspective_counts = []
    perspective_word_counts = []
    for entry in dataset:
        pro_perspectives = entry['perspectives']['pro']
        con_perspectives = entry['perspectives']['con']
        total_perspectives = len(pro_perspectives) + len(con_perspectives)
        perspective_counts.append(total_perspectives)
        
        # Count words in all perspectives
        for persp in pro_perspectives + con_perspectives:
            perspective_word_counts.append(count_words(persp))
    
    avg_perspectives_per_query = statistics.mean(perspective_counts) if perspective_counts else 0
    avg_words_per_perspective = statistics.mean(perspective_word_counts) if perspective_word_counts else 0
    
    # Claim statistics
    claim_word_counts = []
    for entry in dataset:
        claim_word_counts.append(count_words(entry['claims'][0]))
        claim_word_counts.append(count_words(entry['claims'][1]))
    
    avg_words_per_claim = statistics.mean(claim_word_counts) if claim_word_counts else 0
    
    # Document/Evidence statistics
    docs_per_query = []
    for entry in dataset:
        favor_ids = entry['favor_ids']
        against_ids = entry['against_ids']
        total_docs = len(favor_ids) + len(against_ids)
        docs_per_query.append(total_docs)
    
    avg_docs_per_query = statistics.mean(docs_per_query) if docs_per_query else 0
    avg_docs_per_perspective = avg_docs_per_query / avg_perspectives_per_query if avg_perspectives_per_query > 0 else 0
    
    write(f"- **Average perspectives per query:** {avg_perspectives_per_query:.1f}")
    write(f"- **Average words per perspective:** {avg_words_per_perspective:.1f}")
    write(f"- **Average documents per query:** {avg_docs_per_query:.1f}")
    write(f"- **Average words per claim:** {avg_words_per_claim:.1f}")
    write()
    
    # Properties That Make the Task Challenging
    write("## Properties That Make the Task Challenging")
    write()
    write("### 1. Oppositional Structure")
    write()
    write("- All queries require exactly 2 oppositional claims (pro/con)")
    write("- Claims must be directly opposing viewpoints on the same topic")
    write()
    write("### 2. Multi-Perspective Requirement")
    write()
    write(f"- Each claim must have multiple supporting perspectives (avg: {avg_perspectives_per_query:.1f} per query)")
    write("- Perspectives must be distinct and non-overlapping")
    write()
    write("### 3. Evidence Grounding")
    write()
    write("- Each perspective must be supported by document IDs")
    write("- Documents must be retrieved from a corpus of 4,107 evidence documents")
    write("- Requires accurate retrieval and relevance matching")
    write()
    write("### 4. Data Quality Challenges")
    write()
    write(f"- {num_duplicates} duplicate query strings ({num_queries - num_unique} duplicate instances)")
    write("- Requires handling of duplicate queries in evaluation")
    write()
    
    # Extract 2 example queries
    write("## Example Input/Output Pairs")
    write()
    
    for i, example in enumerate(dataset[:2], 1):
        write(f"### Example {i}")
        write()
        write(f"**Query:** {example['query']}")
        write()
        write(f"**Claim 1:** {example['claims'][0]}")
        write()
        write("**Perspectives (pro):**")
        for j, persp in enumerate(example['perspectives']['pro'][:2], 1):
            write(f"{j}. {persp}")
        write()
        write(f"**Supporting document IDs:** `{example['favor_ids'][:3]}`...")
        write()
        
        write(f"**Claim 2:** {example['claims'][1]}")
        write()
        write("**Perspectives (con):**")
        for j, persp in enumerate(example['perspectives']['con'][:2], 1):
            write(f"{j}. {persp}")
        write()
        write(f"**Supporting document IDs:** `{example['against_ids'][:3]}`...")
        write()
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Dataset metrics saved to: {output_file}")


if __name__ == "__main__":
    main()

