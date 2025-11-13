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
    pass