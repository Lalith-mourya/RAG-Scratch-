def hit_rate_at_k(retrieved_ids , relevant_ids , k=3):
    top_k_id = retrieved_ids[:k]
    for i in top_k_id:
        if i in relevant_ids:
            return 1
        
    return 0


retrieved_ids = [5, 2, 8]
relevant_ids = [2, 8, 9, 10]

hit_rate_at_k_value = hit_rate_at_k(retrieved_ids,relevant_ids,3)
print(hit_rate_at_k_value)
