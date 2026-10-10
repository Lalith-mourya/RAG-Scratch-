
import logging
from embedding import model
from similarity import cosine_similarity

logger = logging.getLogger(__name__)


def retriever(query,embedded_chunks,top_k = 3):

    if not query.strip():
        raise ValueError("Query can not be empty")

    if not embedded_chunks:
        logger.warning("Embedded chunks are not loaded")
        return []

    logger.info("Retrieving chunks for query: %s", query)

    query_embedding = model.encode(query,normalize_embeddings=True,show_progress_bar=False)

    results = []
    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding,chunk["embedding"])
        results.append({
            "text":chunk["text"],
            "metadata":chunk["metadata"],
            "score":score
        })

    results.sort(key=lambda item:item["score"],reverse=True)

    return results[:top_k]