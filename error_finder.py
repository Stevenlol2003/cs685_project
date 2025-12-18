import json
from collections import defaultdict

INPUT_FILE = "results-merged-20.json"
ERROR_TEXT = "All 10 generation attempts failed"

error_counts = defaultdict(int)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    qid = entry["id"]

    for claim in entry.get("summary", []):
        for perspective in claim.get("perspectives", []):
            text = perspective.get("text", "")
            if ERROR_TEXT in text:
                error_counts[qid] += 1

print("Total IDs with errors:", len(error_counts))
print("IDs with JSON parse errors:\n")
for qid, count in sorted(error_counts.items()):
    print(f"{qid}")
