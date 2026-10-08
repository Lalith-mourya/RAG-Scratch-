from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR/"data"

def load_docs():
    documents=[]
    for file_path in DATA_DIR.glob("*.txt"):
        with open(file=file_path,mode="r",encoding="utf-8") as f:
            text = f.read()
            text = text.strip()
            documents.append({
                "text":text,
                "metadata":{
                    "source":file_path.name
                }
            })


    for files in documents:
        print(files,end="\n ------------- \n")



load_docs()


