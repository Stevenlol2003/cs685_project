"""
Web Document Relevance Checker

Evaluates whether retrieved web documents are relevant to their query using GPT-5-Nano.
Returns classification mapping document IDs to "R" (Relevant) or "NR" (Not Relevant).
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"


def check_relevance(query: str, web_docs: list) -> dict:
    """
    Classify web documents as Relevant (R) or Not Relevant (NR) to the query.
    
    Args:
        query: The search query/question being evaluated.
        web_docs: List of web document dicts with 'id' and 'content' fields.
    
    Returns:
        dict mapping document IDs (as strings) to "R" or "NR".
        Example: {"0": "R", "1": "NR", "2": "R"}
    """
    if not web_docs:
        return {}
    
    # Build prompt with explicit JSON format requirement
    docs_str = "\n".join([
        f"ID {doc['id']}: {doc['content']}"
        for doc in web_docs
    ])
    
    prompt = f"""You are evaluating whether web documents are relevant to a given query.

Query: "{query}"

Documents:
{docs_str}

Task: For each document ID, classify it as either:
- "R" (Relevant): The document content directly addresses or relates to the query
- "NR" (Not Relevant): The document content does not address the query

You MUST respond with a valid JSON object mapping each document ID (as a string) to either "R" or "NR".

Example format:
{{"0": "R", "1": "NR", "2": "R"}}

Provide ONLY the JSON object, no explanation."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    client = OpenAI(api_key=api_key)
    
    # Retry logic for malformed JSON or API errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_text = completion.choices[0].message.content or ""
            
            # Parse JSON response
            classifications = json.loads(response_text)
            
            # Validate and fill missing IDs with "NR"
            result = {}
            for doc in web_docs:
                doc_id = str(doc['id'])
                if doc_id in classifications:
                    value = classifications[doc_id]
                    # Ensure only "R" or "NR" values
                    result[doc_id] = value if value in ["R", "NR"] else "NR"
                else:
                    # Default to "NR" for missing IDs
                    result[doc_id] = "NR"
            
            return result
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                continue
            else:
                # Final fallback: return all "NR"
                print(f"All retries failed: {e}. Defaulting all documents to NR.")
                return {str(doc['id']): "NR" for doc in web_docs}
    
    # Should not reach here, but just in case
    return {str(doc['id']): "NR" for doc in web_docs}