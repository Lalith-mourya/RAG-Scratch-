from pathlib import Path
import logging

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



load_docs()


