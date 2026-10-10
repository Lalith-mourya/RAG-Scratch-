import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def EMBEDDING(chunks):
    if not chunks:
        logger.warning("No chunks received for embedding")
        return []
    embedded_chunks = []
    logger.info("Generating embeddings for %d chunks", len(chunks))

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts,show_progress_bar=False,normalize_embeddings=True)

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": embedding
        }
        embedded_chunks.append(embedded_chunk)

    logger.info("Successfully generated %d embeddings", len(embedded_chunks))
    return embedded_chunks
