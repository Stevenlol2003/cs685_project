import json
import random
import math
from collections import defaultdict

RANDOM_SEED = 10
RELEVANCE_SAMPLE_SIZE = 30
SUMMARY_SAMPLE_SIZE = 20

INPUT_JSON = "data/valid-queries/valid-k-10-queries-20251216_210630.json"
RELEVANCE_OUT = "data/valid-queries/relevance_eval_30.json"
SUMMARY_OUT = "data/valid-queries/summary_eval_20.json"

def extract_topic(id_offline: str) -> str:
    """
    Extract topic from id_offline.
    Example: 'politics_12' -> 'politics'
    """
    return id_offline.split("_")[0]


def group_by_topic(queries):
    """
    Group query dicts by topic.
    """
    grouped = defaultdict(list)
    for q in queries:
        topic = extract_topic(q["id_offline"])
        grouped[topic].append(q)
    return grouped


def proportional_sample(grouped, total_k, seed=10):
    """
    Proportionally sample total_k items across topic groups.

    Returns:
        sampled_items (list)
        allocation (dict): topic -> number sampled
    """
    random.seed(seed)

    total_n = sum(len(v) for v in grouped.values())
    if total_k > total_n:
        raise ValueError("Requested sample size larger than population.")

    # 1. proportional allocation
    raw_alloc = {
        topic: (len(items) / total_n) * total_k
        for topic, items in grouped.items()
    }

    # 2. floor allocations
    alloc = {topic: math.floor(v) for topic, v in raw_alloc.items()}
    remaining = total_k - sum(alloc.values())

    # 3. distribute remainder by largest fractional part
    remainders = sorted(
        ((topic, raw_alloc[topic] - alloc[topic]) for topic in raw_alloc),
        key=lambda x: x[1],
        reverse=True
    )

    for topic, _ in remainders:
        if remaining == 0:
            break
        if alloc[topic] < len(grouped[topic]):
            alloc[topic] += 1
            remaining -= 1

    # 4.sample per topic
    sampled = []
    for topic, k in alloc.items():
        sampled.extend(random.sample(grouped[topic], k))

    return sampled, alloc

def main():
    # Load valid k=10 queries
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        valid_queries = json.load(f)

    print(f"Loaded {len(valid_queries)} valid k=10 queries")

    # Group by topic
    grouped_topics = group_by_topic(valid_queries)

    print("\nTopic distribution (all valid queries):")
    for topic, items in sorted(grouped_topics.items()):
        print(f"{topic}: {len(items)}")

    # Sample 30 for relevance eval
    relevance_30, relevance_alloc = proportional_sample(
        grouped_topics,
        total_k=RELEVANCE_SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    print("\nRelevance evaluation 30 sample:")
    for topic, count in sorted(relevance_alloc.items()):
        print(f"{topic}: {count}")

    # Save relevance set
    with open(RELEVANCE_OUT, "w", encoding="utf-8") as f:
        json.dump(relevance_30, f, indent=2)

    # Sample 20 for summary eval
    grouped_relevance = group_by_topic(relevance_30)

    summary_20, summary_alloc = proportional_sample(
        grouped_relevance,
        total_k=SUMMARY_SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    print("\nSummary evaluation sample (20) topic allocation:")
    for topic, count in sorted(summary_alloc.items()):
        print(f"{topic}: {count}")

    # Save summary set
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary_20, f, indent=2)

    print("\nSampling complete.")
    print(f"Saved: {RELEVANCE_OUT}")
    print(f"Saved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
