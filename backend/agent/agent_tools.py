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

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    
    print("retrieving context...")
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs