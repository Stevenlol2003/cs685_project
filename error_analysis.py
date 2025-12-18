import json
from pathlib import Path
from scipy.stats import pearsonr

# Read ids.txt
with open("ids.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Parse k
if not lines[0].startswith("k="):
    raise ValueError("First line must be in format 'k=<value>'")
k = int(lines[0].split("=", 1)[1])

# Queries to process
query_ids = lines[1:]

# Load results-merged-{k}.json
results_merged_file = Path(f"results-merged-{k}.json")
with open(results_merged_file, "r", encoding="utf-8") as f:
    results_merged = json.load(f)

# Load merged-{k}.json
merged_file = Path(f"merged-{k}.json")
with open(merged_file, "r", encoding="utf-8") as f:
    merged_data = json.load(f)

# Build a mapping from query string to merged entry for fast lookup
query_to_merged = {entry["query"]: entry for entry in merged_data}

# Compute ratios and prepare correlation lists
ratios = {}
query_lengths = []
avg_content_lengths = []

for qid in query_ids:
    # Find the query string from results-merged-{k}.json
    matching_entries = [r for r in results_merged if r["id"] == qid]
    if not matching_entries:
        print(f"Warning: query id {qid} not found in results-merged-{k}.json")
        continue
    query_str = matching_entries[0]["query"]

    # Find the merged entry
    if query_str not in query_to_merged:
        print(f"Warning: query '{query_str}' not found in merged-{k}.json")
        continue

    merged_entry = query_to_merged[query_str]
    merged_contents = merged_entry.get("merged", [])

    # Only include content without a "score" key
    lengths = [len(item["content"]) for item in merged_contents if "content" in item and "score" not in item]
    if not lengths:
        print(f"Warning: no content without 'score' found for query '{query_str}'")
        continue

    avg_content_length = sum(lengths) / len(lengths)
    query_length = len(query_str)

    # Compute ratio
    ratio = query_length / avg_content_length
    ratios[qid] = ratio

    # Prepare lists for correlation
    query_lengths.append(query_length)
    avg_content_lengths.append(avg_content_length)

# Print ratios and average ratio
print("Ratios (query length / avg content length):")
for qid, ratio in ratios.items():
    print(f"{qid}: {ratio:.4f}")

if ratios:
    avg_ratio = sum(ratios.values()) / len(ratios)
    print(f"\nAverage ratio across all queries: {avg_ratio:.4f}")
else:
    print("No valid ratios computed.")

# Compute Pearson correlation
if query_lengths and avg_content_lengths:
    corr, p_value = pearsonr(query_lengths, avg_content_lengths)
    print(f"\nPearson correlation between query length and average content length: {corr:.4f}")
    print(f"P-value: {p_value:.4g}")
else:
    print("No valid data to compute correlation.")
