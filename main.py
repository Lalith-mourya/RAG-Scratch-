from pathlib import Path
import logging

from chunker import chunk_text
from embedding import EMBEDDING
from retriever import retriever
from generator import generate_answer
from evaluation import evaluate_retriever

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

    return documents

documents = load_docs()


def chunk_documents(documents, chunk_size=500, overlap=100):
    chunk_docs = []
    chunk_id = 1
    logger.info("Started chunking")

    for document in documents:
        text = document["text"]
        source = document["metadata"]["source"]
        text_chunks = chunk_text(text, chunk_size, overlap)

        for text_chunk in text_chunks:
            chunk_docs.append({
                "id": chunk_id,
                "text": text_chunk,
                "metadata": {
                    "source": source
                }
            })
            chunk_id +=1

    logger.info("Successfully created %d chunks", len(chunk_docs))

    return chunk_docs

chunks = chunk_documents(documents)
embedded_chunks = EMBEDDING(chunks=chunks)

#                       EVALUATION  OF RETRIEVER
logger.info("Retriever Evaluation getting started with the golden dataset")
evaluate_retriever(
    embedded_chunks=embedded_chunks,
    retrieve_fn=retriever,
    dataset_path=BASE_DIR / "rag_retriever_golden_dataset_v1.json",
    k=3
)

# query = "How is espresso prepared?"
# results = retriever(query, embedded_chunks, top_k=3)
# response = generate_answer(query, results)

# print(response["answer"])
# print("Sources:", response["sources"])


    