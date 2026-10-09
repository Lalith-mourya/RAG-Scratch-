from pathlib import Path
import logging

from chunker import chunk_text
from embedding import embed_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR/"data"

def load_docs():
    documents=[]

    logger.info("Started the data loading")
    logger.info("Finding for the text files in the %s",DATA_DIR)

    for file_path in DATA_DIR.glob("*.txt"):

        logger.info("Found a file called %s",file_path.name)
        try:
            with open(file=file_path,mode="r",encoding="utf-8") as f:
                logger.info("Loading the %s file",file_path.name)
                text = f.read()
                text = text.strip()
                documents.append({
                                    "text":text,
                                "metadata":{
                                    "source":file_path.name
                                }
                            })
                logger.info("Loaded and converted the file data to doc of %s file",file_path.name)
            
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path.name, e)
            continue
    logger.info("Loaded %d documents",len(documents))

    for files in documents:
        print(files,end="\n ------------- \n")

    return documents

documents = load_docs()


def chunk_documents(documents, chunk_size=500, overlap=100):
    chunk_docs = []

    logger.info("Started chunking")

    for document in documents:
        text = document["text"]
        source = document["metadata"]["source"]

        text_chunks = chunk_text(text, chunk_size, overlap)

        for text_chunk in text_chunks:
            chunk_docs.append({
                "text": text_chunk,
                "metadata": {
                    "source": source
                }
            })

    logger.info("Successfully created %d chunks", len(chunk_docs))

    return chunk_docs


chunks = chunk_documents(documents)

embeddings = embed_chunks(chunks=chunks)
print(len(embeddings))


    