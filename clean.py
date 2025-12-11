import json
import re

def replace_doc_values(obj):
    """
    Recursively replace values of the form 'Doc <number>' with the number.
    """
    if isinstance(obj, dict):
        return {k: replace_doc_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_doc_values(v) for v in obj]
    elif isinstance(obj, str):
        match = re.fullmatch(r"Doc\s*(\d+)", obj)
        if match:
            return int(match.group(1))
        return obj
    else:
        return obj

# ------ EDIT THESE ------
input_path = "tfidf-20-offline.json"
output_path = "offine-20.json"
# -------------------------

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

updated = replace_doc_values(data)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(updated, f, indent=2, ensure_ascii=False)

print("Done! Updated JSON saved to:", output_path)
