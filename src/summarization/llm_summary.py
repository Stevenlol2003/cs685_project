import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    # Load model, token, and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is REQUIRED but not available!")

    # device = 0 if torch.cuda.is_available() else -1
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"

    HF_TOKEN = os.getenv("HF_TOKEN")
    # HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        dtype=torch.float16
    ).to("cuda")

    # Format corpus for the prompt
    corpus_text = "\n".join([
        f"[Doc {doc['id']}]: {doc.get('content', '')[:300]}"  # Limit content length
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
            "perspective": "A perspective supporting or relating to the first claim",
            "evidence_docs": [list of document indices used by perspective used to support the claim]
        }},
        {{
            "claim": "Second claim",
            "perspective": "A perspective supporting or relating to the second claim",
            "evidence_docs": [list of document indices used by perspective used to support the claim]
        }}
    ]
}}

Only respond with valid JSON, no additional text."""

    try:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("================================ RESPONSE =================================")
        print(response_text)
        print("================================ RESPONSE =================================")
        
        # Extract JSON between the last pair of ```
        backticks = re.findall(r'```(.*?)```', response_text, re.DOTALL)

        if backticks:
            json_str = backticks[-1].strip()  # take the last block
            try:
                summary_data = json.loads(json_str)
            except json.JSONDecodeError:
                # fallback if JSON is malformed
                fallback_ids = [doc['id'] for doc in merged_corpus[:3]]
                summary_data = {
                    "perspectives": [
                        {"claim": claims[0], "perspective": response_text, "evidence_docs": fallback_ids},
                        {"claim": claims[1], "perspective": response_text, "evidence_docs": fallback_ids}
                    ]
                }
        else:
            # fallback if no backticks found
            fallback_ids = [doc['id'] for doc in merged_corpus[:3]]
            summary_data = {
                "perspectives": [
                    {"claim": claims[0], "perspective": response_text, "evidence_docs": fallback_ids},
                    {"claim": claims[1], "perspective": response_text, "evidence_docs": fallback_ids}
                ]
            }


        summary_data["query"] = query
        return summary_data
        
    except Exception as e:
        print("GENERATION FAILED: ", e)
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