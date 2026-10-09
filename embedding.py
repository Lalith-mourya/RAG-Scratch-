import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_chunks(chunks):
    logger.info("Generating embeddings for %d chunks", len(chunks))

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts,show_progress_bar=False,normalize_embeddings=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info("Successfully generated embeddings")
    return chunks
