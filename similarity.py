import numpy as np


def cosine_similarity(vector_a , vector_b):
    dot_product = np.dot(vector_a,vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_b==0 or magnitude_a==0:
        return 0.0

    similarity = dot_product/(magnitude_a*magnitude_b)
    return similarity

