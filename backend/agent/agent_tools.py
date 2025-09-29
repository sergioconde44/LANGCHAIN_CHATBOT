import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DATA_DIR = "data"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Create and load RAG Vector Store
if os.path.exists("data/faiss_index"):
    vector_store = FAISS.load_local(
    "data/faiss_index", embeddings, allow_dangerous_deserialization=True
)
else:
    print("No index found")    

@tool(response_format="content")
def retrieve(query: str):
    """Retrieve information related to a query."""
    
    print("retrieving context...")
    adeslas_go_docs = vector_store.similarity_search(query, k=2, filter={"filename": "adeslas-go"})
    adeslas_plena_docs = vector_store.similarity_search(query, k=2, filter={"filename": "adeslas-plena-total-vital"})

    serialized = "ADESLAS GO: \n---\n" + "\n---\n".join(
        (f"{doc.page_content}")
        for doc in adeslas_go_docs
    ) + "\n\nADESLAS PLENA TOTAL VITAL: \n---\n" + "\n---\n".join(
        (f"{doc.page_content}")
        for doc in adeslas_plena_docs
    ) 
    return serialized