import json
from pathlib import Path


def load_theperspective_dataset(folder_path: str):
    """
    Load theperspective dataset from data/theperspective/

    data.jsonl example:
        {
            "id": "Entertainment_0",
            "response1": [...],
            "response2": [...],
            "favor_ids": [205, 364],
            "against_ids": [1138, 858],
            "t1": "Claim 1 text",
            "t2": "Claim 2 text",
            "title": "Topic title"
        }

    doc_new.jsonl example:
        {
            "id": 0,
            "content": "Document text..."
        }

    Returns:
        list[dict]: Each item formatted for the pipeline as:
        {
            "id": str,
            "query": str,
            "claims": [t1, t2],
            "perspectives": {"pro": response1, "con": response2},
            "favor_ids": [...],
            "against_ids": [...],
        }
    """
    folder = Path(folder_path)
    data_path = folder / "data.jsonl"
    doc_path = folder / "doc_new.jsonl"

    # Evidence documents
    documents = {}
    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents[doc["id"]] = doc["content"]

    # Query, claims, and perspectives
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            entry = {
                "id": ex.get("id"),
                "query": ex.get("title", ""),
                "claims": [ex.get("t1", ""), ex.get("t2", "")],
                "perspectives": {
                    "pro": ex.get("response1", []),
                    "con": ex.get("response2", [])
                },
                "favor_ids": ex.get("favor_ids", []),
                "against_ids": ex.get("against_ids", [])
            }

            dataset.append(entry)

    print(f"Loaded ThePerspective dataset from {folder_path} "
          f"({len(dataset)} entries, {len(documents)} documents).")

    return dataset

def load_theperspective_evidence(folder_path: str):
    """
    Load evidence documents for theperspective dataset.

    Returns:
        list[dict]: Each item formatted as:
        {
            "id": int,
            "content": str
        }
    """
    folder = Path(folder_path)
    doc_path = folder / "doc_new.jsonl"

    evidence = []
    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            evidence.append({
                "id": doc["id"],
                "content": doc["content"]
            })

    return evidence


def load_perspectrumx_dataset(folder_path: str):
    # Placeholder for PerspectrumX dataset
    raise NotImplementedError("PerspectrumX dataset not implemented yet.")
 