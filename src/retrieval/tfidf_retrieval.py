from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def retrieve_local_docs(query: str, evidence: list, k: int = 5):
    """
    Retrieve top-k most relevant local TF-IDF documents
    Args:
        query: user/topic query
        evidence: list of dicts with id and content
        k: number of documents to return
    Returns:
        list[dict]: top-k evidence docs
    """
    if not evidence:
        return []
    
    # Extract document contents
    doc_contents = [doc.get("content", "") for doc in evidence]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(lowercase=True)
    
    # Fit on all documents and transform query
    doc_vectors = vectorizer.fit_transform(doc_contents)
    # print(doc_vectors.toarray())
    query_vector = vectorizer.transform([query])
    # print(query_vector.toarray())

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    # print(similarities)
    # print(all(x==0 for x in similarities))

    # Get top-k document indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    # print(top_k_indices)
    
    # Return top-k documents with their similarity scores
    top_docs = []
    for idx in top_k_indices:
        if idx < len(evidence):
            doc = evidence[idx].copy()
            doc["score"] = float(similarities[idx])
            top_docs.append(doc)
    
    return top_docs