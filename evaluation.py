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


def mean_reciprocal_rank(all_retrieved_ids , all_relevant_ids , k=3):
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

    




