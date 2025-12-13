import json
def merge_documents(offline_docs, web_docs, output_path):
# Load offline summaries (tfidf summaries)
    with open(offline_docs, 'r') as f:
        offline_data = json.load(f)
    
    # Load web retrieval results
    with open(web_docs, 'r') as f:
        web_data = json.load(f)
    
    # Create a dictionary for web docs by query
    web_by_query = {item['query']: item['web_docs']['results'] for item in web_data}
    
    # For each offline summary item
    for item in offline_data:
        query = item['query']
        if query in web_by_query:
            relevant_docs = []
            for doc in web_by_query[query]:
                if doc.get('relevance') == 'R':
                    relevant_docs.append({
                        'content': doc['content'],
                        'url': doc['url']
                    })
            # Add relevant web docs to the item
            item['relevant_web_docs'] = relevant_docs
    
    # Save the merged data to output path
    with open(output_path, 'w') as f:
        json.dump(offline_data, f, indent=2)


def merge_docs_lists(local_docs, web_docs):
    """
    Merge local docs with relevant web docs (relevance == 'R').
    Returns list of merged docs, with web docs transformed to {"id": url, "content": content}
    """
    relevant_web = [doc for doc in web_docs if doc.get("relevance") == "R"]
    transformed_web = [{"id": doc["url"], "content": doc["content"]} for doc in relevant_web]
    return local_docs + transformed_web
