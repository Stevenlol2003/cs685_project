import json
import re
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def summarize_query(query: str, merged_corpus: list, claims: list):
    """
    Generate multi-perspective summary using Llama-3.1-8B-Instruct
    
    Args:
        query: the query/topic
        merged_corpus: list of documents with id, content, and score
        claims: list of 2 claims for different perspectives
    
    Returns:
        dict: multi-perspective summary with structure:
        {
            "query": str,
            "perspectives": [
                {
                    "claim": str,
                    "perspective": str,
                    "evidence_docs": list of doc ids
                },
                {
                    "claim": str,
                    "perspective": str,
                    "evidence_docs": list of doc ids
                }
            ]
        }
    """
    if not merged_corpus or len(claims) < 2:
        return {
            "query": query,
            "perspectives": []
        }
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9
    )
    
    # Format corpus for the prompt
    corpus_text = "\n".join([
        f"[Doc {i}]: {doc.get('content', '')[:300]}"  # Limit content length
        for i, doc in enumerate(merged_corpus)
    ])
    
    # Create prompt for multi-perspective summarization
    prompt = f"""Based on the following query and documents, generate a multi-perspective summary with exactly 2 perspectives.

Query: {query}

Claims to consider:
1. {claims[0]}
2. {claims[1]}

Documents:
{corpus_text}

Generate a response in JSON format with exactly this structure:
{{
    "perspectives": [
        {{
            "claim": "First claim",
            "perspective": "A perspective supporting or relating to the first claim, mentioning relevant document numbers in brackets like [Doc 0] [Doc 1]",
            "evidence_docs": [list of document indices used]
        }},
        {{
            "claim": "Second claim",
            "perspective": "A perspective supporting or relating to the second claim, mentioning relevant document numbers in brackets like [Doc 2] [Doc 3]",
            "evidence_docs": [list of document indices used]
        }}
    ]
}}

Only respond with valid JSON, no additional text."""

    try:
        # Generate response
        response = pipe(prompt)
        response_text = response[0]['generated_text']
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            summary_data = json.loads(json_match.group())
        else:
            # Fallback if JSON extraction fails
            summary_data = {
                "perspectives": [
                    {
                        "claim": claims[0],
                        "perspective": response_text,
                        "evidence_docs": list(range(min(len(merged_corpus), 3)))
                    },
                    {
                        "claim": claims[1],
                        "perspective": response_text,
                        "evidence_docs": list(range(min(3, len(merged_corpus))))
                    }
                ]
            }
        
        summary_data["query"] = query
        return summary_data
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Return fallback structure
        return {
            "query": query,
            "perspectives": [
                {
                    "claim": claims[0],
                    "perspective": f"Could not generate perspective due to error: {str(e)}",
                    "evidence_docs": []
                },
                {
                    "claim": claims[1],
                    "perspective": f"Could not generate perspective due to error: {str(e)}",
                    "evidence_docs": []
                }
            ]
        }