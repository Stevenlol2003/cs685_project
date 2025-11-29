def recall_at_k(retrieved_ids, gold_ids, k):
    """Compute Recall@k."""
    
    retrieved_top_k = retrieved_ids[:k]

    retrieved_set = set(retrieved_top_k)
    gold_set = set(gold_ids)
    relevant_retrieved = retrieved_set.intersection(gold_set)

    if len(gold_set) == 0:
        return 0.0

    print(f"relevant_retrieved: {relevant_retrieved}")
    print(f"gold_set: {gold_set}")

    return len(relevant_retrieved) / len(gold_set)


def cover_at_k(covered_perspectives, gold_perspectives):
    """Compute Cover@k."""
    
    covered_set = set(covered_perspectives)
    gold_set = set(gold_perspectives)

    covered_gold_set = covered_set.intersection(gold_set)

    if len(gold_set) == 0:
        return 0.0

    print(f"covered_gold_set: {covered_gold_set}")
    print(f"gold_set: {gold_set}")

    return len(covered_gold_set) / len(gold_set)


