import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from outlines.models import from_transformers

# Module-level cache for model and generator
_model_cache = {}
_tokenizer_cache = {}
_outlines_model_cache = {}


def _sanitize_result_dict(result_dict: Dict[str, Any], available_doc_ids: List[int]) -> Dict[str, Any]:
    """Post-process generated JSON to enforce global doc uniqueness and validity.

    What this sanitizer does now:
    - Keeps the first occurrence of each document ID and drops any duplicates later on.
    - Filters evidence to only IDs present in the provided corpus (`available_doc_ids`).
    - Skips perspectives that lose all valid evidence (no reassignment of random docs).
    - Drops claims that end up with no valid perspectives.

    Rationale:
    Reassigning a random unused corpus doc to perspectives that lost evidence creates
    text-to-doc mismatches and harms citation correctness. We therefore prefer to drop
    such perspectives, preserving semantic integrity over shape completeness.
    Returns an empty dict if nothing valid remains.
    """
    summaries = result_dict.get("summaries", []) or []
    # Only allow citing documents that actually exist in the merged corpus
    available_set = {d for d in available_doc_ids if d is not None}
    # Track global uniqueness: a doc ID may appear only once across all perspectives
    seen_docs = set()
    sanitized_summaries = []

    for claim in summaries:
        claim_text = claim.get("claim") if isinstance(claim, dict) else getattr(claim, "claim", None)
        perspectives = claim.get("perspectives", []) if isinstance(claim, dict) else getattr(claim, "perspectives", [])
        new_perspectives = []

        for p in perspectives:
            p_text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
            p_docs = p.get("evidence_docs", []) if isinstance(p, dict) else getattr(p, "evidence_docs", [])
            # Keep only valid, unseen doc IDs to enforce correctness and uniqueness
            unique_docs = [d for d in p_docs if d in available_set and d not in seen_docs]

            # Skip perspectives that lose all evidence (invalid citations)
            if not unique_docs:
                continue

            seen_docs.update(unique_docs)
            # Preserve the original perspective text with its validated evidence docs
            new_perspectives.append({"text": p_text, "evidence_docs": unique_docs})

        if new_perspectives:
            # Only keep claims that have at least one valid perspective remaining
            sanitized_summaries.append({"claim": claim_text, "perspectives": new_perspectives})

    return {"summaries": sanitized_summaries} if sanitized_summaries else {}

# Define the expected JSON schema using Pydantic
class Perspective(BaseModel):
    text: str = Field(description="One-sentence perspective summary")
    evidence_docs: List[int] = Field(description="List of document IDs supporting this perspective")

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, text):
        """Ensure perspective text is not empty or just whitespace."""
        if not text or not text.strip():
            raise ValueError("Perspective text cannot be empty")
        return text

    @field_validator('evidence_docs')
    @classmethod
    def validate_evidence_docs_not_empty(cls, evidence_docs):
        """Ensure each perspective has at least one evidence document."""
        if not evidence_docs or len(evidence_docs) == 0:
            raise ValueError("Each perspective must have at least one evidence document")
        return evidence_docs

class Claim(BaseModel):
    claim: str = Field(description="The claim in response to the query")
    perspectives: List[Perspective] = Field(description="List of perspectives supporting this claim")

    @field_validator('claim')
    @classmethod
    def validate_claim_not_empty(cls, claim):
        """Ensure claim text is not empty or just whitespace."""
        if not claim or not claim.strip():
            raise ValueError("Claim text cannot be empty")
        return claim

class MultiPerspectiveSummary(BaseModel):
    summaries: List[Claim] = Field(description="List of claims with their perspectives", min_items=2, max_items=2)
    
    @field_validator('summaries')
    @classmethod
    def validate_unique_docs_globally(cls, summaries):
        """Ensure each document ID appears only once across the entire summary."""
        seen_docs = set()
        for claim in summaries:
            for perspective in claim.perspectives:
                for doc_id in perspective.evidence_docs:
                    if doc_id in seen_docs:
                        raise ValueError(f"Document {doc_id} appears in multiple perspectives across the summary. Each document must be used only once.")
                    seen_docs.add(doc_id)
        return summaries

def _load_model(model_name: str, hf_token: str):
    """Load transformers model and tokenizer with caching."""
    if model_name not in _model_cache:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is REQUIRED but not available!")
        
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        _model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return _model_cache[model_name], _tokenizer_cache[model_name]

def _get_outlines_model(model_name: str, hf_token: str):
    """Get outlines model wrapper with caching."""
    if model_name not in _outlines_model_cache:
        hf_model, tokenizer = _load_model(model_name, hf_token)
        # Wrap the transformers model with outlines
        _outlines_model_cache[model_name] = from_transformers(hf_model, tokenizer)
    
    return _outlines_model_cache[model_name]

def summarize_query(query: str, merged_corpus: list, claims: list):
    """
    Generate multi-perspective summary using Llama-3.2-3B-Instruct with constrained JSON decoding.
    
    Args:
        query: the query/topic
        merged_corpus: list of documents with id, content, and score
        claims: list of 2 claims for different perspectives
    
    Returns:
        list: multi-perspective summary with structure:
        [
            {
                "claim": str,
                "perspectives": [
                    {"text": str, "evidence_docs": list of doc ids}
                ]
            }
        ]
    """
    if not merged_corpus or len(claims) < 2:
        return []
    
    # Load outlines model (cached on first call)
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    try:
        model = _get_outlines_model(model_name, HF_TOKEN)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # Format corpus for the prompt
    corpus_text = "\n".join([
        f"[Doc {doc['id']}]: {doc.get('content', '')}"
        for doc in merged_corpus
    ])
    available_doc_ids = [doc.get('id') for doc in merged_corpus]
    
    print("================================ CORPUS TEXT =================================")
    print(corpus_text)
    print("================================ CORPUS TEXT =================================")

    # Create prompt for multi-perspective summarization
    prompt = f"""Given the query and documents, create a multi-perspective summary with exactly 2 claims (one positive, one negative).

Query: {query}

Documents:
{corpus_text}

Rules:
1. Include both a positive claim and a negative claim in response to the query
2. Each perspective should be a one-sentence summary
3. Each perspective MUST reference specific document IDs that support it - DO NOT use empty evidence_docs
4. Group related perspectives under the same claim
5. Ensure all document IDs used are from the provided documents
6. CRITICAL: Each document ID can only be used ONCE across the entire summary. Different documents must support opposing viewpoints.
7. Prefer using multiple distinct documents for each claim; when available, aim for two or more distinct docs per claim, but prioritize validity and relevance.
8. IGNORE any documents that are clearly off-topic or irrelevant to the query - only cite documents that directly address the query's subject matter.

Generate the JSON output now:"""

    last_json = None  # best-effort parsed JSON from latest attempt

    # Retry loop for generation
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use constrained generation to enforce JSON schema
            result = model(prompt, MultiPerspectiveSummary, max_new_tokens=850, temperature=0.1, top_p=0.9)
            
            print("================================ GENERATED RESPONSE =================================")
            print(result)
            print("================================ GENERATED RESPONSE =================================")
            
            # Parse JSON string result
            if isinstance(result, str):
                result_dict = json.loads(result)
                sanitized = _sanitize_result_dict(result_dict, available_doc_ids)
                if sanitized:
                    last_json = sanitized  # keep best-effort parsed JSON
                    # Validate through Pydantic to enforce constraints
                    summary_obj = MultiPerspectiveSummary(**sanitized)
                    return [claim.model_dump() for claim in summary_obj.summaries]
                else:
                    last_json = result_dict
                    raise ValueError("Sanitization removed all perspectives")
            elif hasattr(result, 'summaries'):
                # If already a Pydantic object, sanitize then return
                summaries = [claim.model_dump() for claim in result.summaries]
                sanitized = _sanitize_result_dict({"summaries": summaries}, available_doc_ids)
                if sanitized:
                    last_json = sanitized
                    summary_obj = MultiPerspectiveSummary(**sanitized)
                    return [claim.model_dump() for claim in summary_obj.summaries]
                else:
                    last_json = {"summaries": summaries}
                    raise ValueError("Sanitization removed all perspectives")
            else:
                return []

        except Exception as e:
            print(f"GENERATION ATTEMPT {attempt + 1}/{max_retries} FAILED: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed; if we have a parsed JSON, return it even if it violates constraints
                if last_json and "summaries" in last_json:
                    summaries = last_json.get("summaries", [])
                    normalized = []
                    for claim in summaries:
                        if hasattr(claim, "model_dump"):
                            normalized.append(claim.model_dump())
                        else:
                            normalized.append(claim)
                    return normalized

                # No parsed JSON available, return fallback
                print(f"All {max_retries} attempts failed. Returning fallback summary.")
                fallback_ids = [doc['id'] for doc in merged_corpus[:min(3, len(merged_corpus))]]
                return [
                    {
                        "claim": claims[0] if len(claims) > 0 else "Positive claim",
                        "perspectives": [
                            {
                                "text": f"Error generating summary: {str(e)[:100]}",
                                "evidence_docs": fallback_ids
                            }
                        ]
                    },
                    {
                        "claim": claims[1] if len(claims) > 1 else "Negative claim",
                        "perspectives": [
                            {
                                "text": f"Error generating summary: {str(e)[:100]}",
                                "evidence_docs": fallback_ids
                            }
                        ]
                    }
                ]
            # Continue to next attempt
