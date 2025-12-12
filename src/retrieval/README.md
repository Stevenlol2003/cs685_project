# Retrieval Module

Implements document retrieval combining local TF-IDF search and web retrieval with Tavily API.

## Components

### `tfidf_retrieval.py`
Local document retrieval using TF-IDF vectorization.

```python
from src.retrieval.tfidf_retrieval import retrieve_local_docs

query = "climate change policy"
local_docs = retrieve_local_docs(query, evidence, k=5)
# Returns: list[dict] with keys: id, content, score
```

### `web_retrieval.py`
Web document retrieval using Tavily API with rate limiting (no on-disk cache).

```python
from src.retrieval.web_retrieval import search_web

web_docs = search_web("climate change policy", k=3)
# Returns: {
#   "num_docs": 3,
#   "api_k": 3,          # actual max_results used (may be > k if retries bumped it)
#   "results": [
#       {"id": 0, "content": "...", "url": "...", "source_type": "web", "title": "...", "domain": "..."},
#       ...
#   ]
# }
```

## Usage

### Combining Local and Web Results

```python
from src.retrieval.tfidf_retrieval import retrieve_local_docs
from src.retrieval.web_retrieval import search_web

query = "climate change policy"
k_local = 5
k_web = 3

# Get both local and web documents
local_docs = retrieve_local_docs(query, evidence, k=k_local)
web_docs = search_web(query, k=k_web)

# Combine for downstream processing
all_docs = local_docs + web_docs
```

## Configuration

### Tavily API Setup
Set your Tavily API key in `.env`:
```env
TAVILY_API_KEY=tvly-your_actual_api_key
```
Get your free API key from https://tavily.com

## Features

### Automatic Retry & Rate Limiting
- Exponential backoff on rate limits (1s → 2s → 4s)
- Automatic detection and retry for rate limit errors
- Max 3 retry attempts per request
- If fewer than k results are returned, the module increments `api_k` and retries (up to k+10)

### Error Handling
- Missing API key: Returns empty list with logged error
- Network errors: Logged and handled gracefully
- Invalid queries: Returns empty list
- Cache errors: Auto-creates cache directory on first write

### Output Format

- TF-IDF retrieval returns a list of documents:

```python
{
    "id": "unique_hash_identifier",
    "content": "cleaned_document_text",
    "source_type": "local",
    "score": 0.87
}
```

- Web retrieval returns a dictionary with metadata and a list of documents:

```python
{
    "num_docs": 3,          # number of docs returned (trimmed to k)
    "api_k": 4,             # actual Tavily max_results used (may exceed k)
    "results": [
        {
            "id": 0,        # zero-based within the result set
            "content": "cleaned_document_text",
            "url": "https://source.com",
            "source_type": "web",
            "title": "Document Title",
            "domain": "source.com"
        },
        ...
    ]
}
```
