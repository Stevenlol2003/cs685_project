from src.utils.io import load_theperspective_dataset
from src.utils.io import load_theperspective_evidence
from src.evaluation.local_metrics import recall_at_k, cover_at_k

from src.retrieval.tfidf_retrieval import retrieve_local_docs

query = "Surrealist Memes: Regression or Progression?"

data = load_theperspective_dataset("data/theperspective")
evidence = load_theperspective_evidence("data/theperspective")

top_docs = retrieve_local_docs(query, evidence, k=5)
print(f"top_docs:\n{top_docs}")

retrieved_ids = [doc.get('id') for doc in top_docs]
print(f"retrieved_ids: {retrieved_ids}")

gold_ids = []

for doc in data:
    if doc.get("query") == query:
        print(doc.get("query"))
        for fid in doc.get("favor_ids"):
            gold_ids.append(fid)
        for aid in doc.get("against_ids"):
            gold_ids.append(aid)

print(f"gold_ids: {gold_ids}")

recall_k = recall_at_k(retrieved_ids, gold_ids, k=5)
print(recall_k)