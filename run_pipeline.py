import argparse
import json
from datetime import datetime
from pathlib import Path
from src.utils.io import load_theperspective_dataset
from src.utils.io import load_theperspective_evidence
from src.retrieval.tfidf_retrieval import retrieve_local_docs
# from src.retrieval.web_retrieval import search_web
# from src.validation.entailment import check_entailment
# from src.summarization.merge import merge_documents
from src.summarization.merge import merge_docs_lists
from src.summarization.llm_summary import summarize_query
from src.summarization.llm_summary_merged import summarize_query as summarize_query_merged
# from src.evaluation.web_metrics import evaluate_all


def main():
    # Two entry points:
    # 1) Standard pipeline (offline/web retrieval + summarization):
    #    python run_pipeline.py --dataset theperspective --offline-k 0 --online-k 10 --method tfidf --limit 5
    #    -> uses llm_summary.py with integer doc IDs from offline corpus and Tavily web docs
    # 2) Merged-corpus pipeline (no retrieval, premerged docs with URL/str IDs):
    #    python run_pipeline.py --merged-file results/merged-5.json --limit 5
    #    -> uses llm_summary_merged.py which accepts mixed int/str IDs (e.g., URLs)
    # GPU + HF_TOKEN are still required for either mode; merged mode simply bypasses retrieval.
    parser = argparse.ArgumentParser(
        description="Web-Augmented Multi-Perspective Summarization Pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["theperspective", "perspectrumx"],
        default="theperspective",
        help="Dataset to use theperspective or perspectrumx"
    )
    parser.add_argument(
        "--offline-k",
        type=int,
        default=0,
        help="Number of top offline (TF-IDF) documents to retrieve."
    )
    parser.add_argument(
        "--online-k",
        type=int,
        default=0,
        help="Number of top online (web) documents to retrieve."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tfidf",
        help="Retrieval method label to include in filename (e.g., tfidf)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (e.g., 10 for a quick test)."
    )
    parser.add_argument(
        "--merged-file",
        type=str,
        default=None,
        help="Optional path to a pre-merged corpus JSON (e.g., results/merged-5.json)."
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    offline_k = args.offline_k
    online_k = args.online_k
    method = args.method
    limit = args.limit
    merged_file = args.merged_file

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If a merged corpus is provided, skip retrieval and summarize directly
    if merged_file:
        merged_path = Path(merged_file)
        with open(merged_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)

        if limit is not None:
            merged_data = merged_data[:limit]
            print(f"Processing first {len(merged_data)} merged queries due to --limit={limit}.")

        output_file = results_dir / (
            f"results-{merged_path.stem}-{timestamp}.json"
        )

        print(f"\nLoaded {len(merged_data)} queries from merged corpus: {merged_path}")
        print(f"Saving results to: {output_file}")

        results = []
        for i, entry in enumerate(merged_data):
            query_text = entry.get("query", "")
            merged_corpus = entry.get("merged") or entry.get("docs") or []

            print("\n")
            print(f"[{i+1}/{len(merged_data)}] Query: {query_text}")
            print("\n")

            summary = summarize_query_merged(query_text, merged_corpus)
            print(f"summary:\n{summary}")

            result_entry = {
                "id": entry.get("id", f"query_{i}"),
                "query": query_text,
                "summary": summary,
                "metrics": None
            }
            results.append(result_entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Pipeline completed (merged corpus mode)")
        print(f"Results saved to: {output_file}")
        return

    output_file = results_dir / (
        f"results-{offline_k}-offline-{online_k}-online-{method}-{timestamp}.json"
    )

    # Load dataset
    if dataset_name == "theperspective":
        dataset = load_theperspective_dataset("data/theperspective")
    else:
        raise NotImplementedError("Perspectrumx not yet added.")

    total_queries = len(dataset)
    print(f"\nLoaded {total_queries} queries from {dataset_name} dataset.")
    print(f"Using top-{online_k} retrieval for web retrieval.")
    print(f"Saving results to: {output_file}")

    # print(dataset)

    # Load evidence depending on dataset
    if dataset_name == "theperspective":
        evidence = load_theperspective_evidence("data/theperspective")
    else:
        raise NotImplementedError("Perspectrumx not yet added.")

    # Load valid-web data for testing (only if online_k > 0)
    if online_k > 0:
        valid_web_path = f"data/valid-web/valid-web-{online_k}.json"
        with open(valid_web_path, 'r', encoding='utf-8') as f:
            valid_web_data = json.load(f)
        web_docs_by_query = {item['query']: item['web_docs']['results'] for item in valid_web_data}
    else:
        web_docs_by_query = {}

    # Optionally limit dataset for quick tests
    if limit is not None:
        dataset = dataset[:limit]
        print(f"Processing first {len(dataset)} queries due to --limit={limit}.")

    # Go over each query, should be from title section for theperspective
    results = []
    for i, entry in enumerate(dataset):
        query_text = entry["query"]
        print("\n")
        # could remove query: text
        print(f"[{i+1}/{len(dataset)}] Query: {query_text}")
        print("\n")

        # TF-IDF document retrieval
        local_docs = retrieve_local_docs(query_text, evidence, k=offline_k)
        # print(len(local_docs))
        # print(local_docs)

        # Web retrieval
        web_docs = web_docs_by_query.get(query_text, [])

        # Entailment and novel perspective validation
        # validated_web_docs = []
        # #validated_web_docs = [
        #   check_entailment(doc, entry["perspectives"]) for doc in web_docs
        # ]

        # Merge local documents + web documents
        merged_corpus = merge_docs_lists(local_docs, web_docs)

        # Summarization: model generates claims; pass only query and docs
        summary = summarize_query(query_text, merged_corpus)

        print(f"summary:\n{summary}")

        # Evaluation - calculate metrics for LLM summary compared to gold data
        # metrics = evaluate_all(summary, entry)
        # Store minimal results: id, query, summary, metrics
        result_entry = {
            "id": entry.get("id", f"query_{i}"),
            "query": query_text,
            "summary": summary,
            "metrics": None
        }
        results.append(result_entry)

        # print(f"Summary metrics: {metrics}\n")

    # Save all results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save all merged results (commented out - using main results file instead)
    # merged_file = results_dir / f"merged-{online_k}.json"
    # with open(merged_file, 'w', encoding='utf-8') as f:
    #     json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"Pipeline completed")
    print(f"Results saved to: {output_file}")
    # print(f"Merged docs saved to: {merged_file}")

if __name__ == "__main__":
    main()

