def hit_rate_at_k(retrieved_ids , relevant_ids , k=3):

    top_k_id = retrieved_ids[:k]
    
    for i in top_k_id:
        if i in relevant_ids:
            return 1 
    return 0

def recall_k(retrieved_ids,relevant_ids,k=3):

    if not relevant_ids:
        return 0.0
    
    top_k_ids = retrieved_ids[:k]
    count =0
    for i in top_k_ids:
        if i in relevant_ids:
            count+=1
    return count/len(relevant_ids)








retrieved_ids = [5, 2, 8]
relevant_ids = [2, 8, 9, 10]

print(hit_rate_at_k(retrieved_ids,relevant_ids,3))
print(recall_k(retrieved_ids,relevant_ids,3))

