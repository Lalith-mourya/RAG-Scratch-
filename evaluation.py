import json
from pathlib import Path

def hit_rate_at_k(retrieved_ids , relevant_ids , k=3):

    top_k_id = retrieved_ids[:k]

    for id in top_k_id:
        if id in relevant_ids:
            return 1 
    return 0

def recall_k(retrieved_ids,relevant_ids,k=3):

    if not relevant_ids:
        return 0.0
    
    top_k_ids = retrieved_ids[:k]
    count =0
    for id in top_k_ids:
        if id in relevant_ids:
            count+=1
    return count/len(relevant_ids)

def precision_k(retrieved_ids , relevant_ids , k=3):
    if k<=0:
        return 0.0
    
    count = 0

    top_k_ids = retrieved_ids[:k]

    for id in top_k_ids:
        if id in relevant_ids:
            count+=1

    return count / k

def reciprocal_rank(retrieved_ids , relevant_ids , k=3):

    top_k_ids = retrieved_ids[:k]

    for rank,id in enumerate(top_k_ids,start=1):

        if id in relevant_ids:
            return 1/rank
        
    return 0.0


def mean_reciprocal_rank(all_retrieved_ids , all_relevant_ids , k):
    if not all_retrieved_ids:
        return 0.0

    if len(all_retrieved_ids) != len(all_relevant_ids):
        raise ValueError(
            f"Number of retrieved query results ({len(all_retrieved_ids)}) "
            f"does not match number of ground-truth queries "
            f"({len(all_relevant_ids)})"
        )
    
    total_reciprocal_rank = 0.0

    for retrieved_ids,relevant_ids in zip(all_retrieved_ids,all_relevant_ids):
        total_reciprocal_rank += reciprocal_rank(retrieved_ids,relevant_ids,3)

    return total_reciprocal_rank/len(all_retrieved_ids)

def evaluate_retriever(embedded_chunks, retrieve_fn , dataset_path="rag_retriever_golden_dataset_v1.json",k=3):
    dataset_path = Path(dataset_path)
    with dataset_path.open("r",encoding="utf-8") as file:
        dataset  = json.load(file)

    queries = dataset["answerable_queries"]

    chunk_ids = [chunk["id"] for chunk in embedded_chunks]
    chunk_ids_set = set(chunk_ids)

    if len(chunk_ids)!= len(chunk_ids_set):
        raise ValueError("Duplicate chunks are found in corpus")

    all_retrieved_ids = []
    all_relevant_ids = []

    hit_scores = []
    recall_scores = []
    precision_scores = []

    for item in queries:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        missing_ids = set(relevant_ids) - set(chunk_ids_set)

        if missing_ids:
            raise ValueError(
                f"{item['id']} references unknown chunk IDs: "
                f"{sorted(missing_ids)}"
            )

        results = retrieve_fn(query,embedded_chunks,k)
        retrieved_ids = [chunk["id"] for chunk in results]

        all_retrieved_ids.append(retrieved_ids)
        all_relevant_ids.append(relevant_ids)

        hit = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_k(retrieved_ids, relevant_ids, k)
        precision = precision_k(retrieved_ids, relevant_ids, k)

        hit_scores.append(hit)
        recall_scores.append(recall)
        precision_scores.append(precision)

        print(f"\n{item['id']}: {query}")
        print(f"Relevant IDs: {relevant_ids}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Hit Rate@{k}: {hit}")
        print(f"Recall@{k}: {recall:.3f}")
        print(f"Precision@{k}: {precision:.3f}")

    if not queries:
        raise ValueError("The golden dataset has no answerable queries.")

    mrr = mean_reciprocal_rank(all_retrieved_ids,all_relevant_ids,k)

    print("\n========== RETRIEVAL EVALUATION SUMMARY ==========")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Hit Rate@{k}: {sum(hit_scores) / len(hit_scores):.3f}")
    print(f"Mean Recall@{k}: {sum(recall_scores) / len(recall_scores):.3f}")
    print(
        f"Mean Precision@{k}: "
        f"{sum(precision_scores) / len(precision_scores):.3f}"
    )
    print(f"MRR@{k}: {mrr:.3f}")

    return {
        "query_count": len(queries),
        f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        f"mean_recall@{k}": sum(recall_scores) / len(recall_scores),
        f"mean_precision@{k}": sum(precision_scores) / len(precision_scores),
        f"mrr@{k}": mrr
    }


