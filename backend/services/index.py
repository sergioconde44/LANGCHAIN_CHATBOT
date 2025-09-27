import os
from glob import glob
from dotenv import load_dotenv
import time

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import pdfplumber

load_dotenv()

DATA_DIR = "data"
DOCUMENTS_DIR = "resources"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class BasePreprocessing:
    def __init__(self, path, chunks_size=1024, chunk_overlap=200):
        self.path = path if path is not None else None
        self.chunks_size = chunks_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunks_size, chunk_overlap=chunk_overlap
        )

    def __call__(self, *args, **kwds):
        return self._preprocess(
            chunks_size=self.chunks_size, chunk_overlap=self.chunk_overlap
        )
    
    def _save_extracted_text(self, file_path, text):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_file_path = os.path.join(self.path, f"extractedText/{base_name}.txt")
        
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return ""

    def _extract_text_from_pdf(self) -> str:
        """
        Extract text from a PDF file.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            str: Extracted text from the PDF file.
        """
        pdf_path = os.path.abspath(self.path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        if pdf_path.lower().endswith(".pdf"):
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    print(page.extract_text())
                    all_text += page.extract_text() + "\n"

        else:
            files = glob(pdf_path + "/*.pdf")
            if not files:
                raise FileNotFoundError(
                    f"No PDF files found in the directory {pdf_path}."
                )
            all_text = ""
            for file in files:
                print("Extracting text from ", file)
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        all_text += page.extract_text() + "\n"
                
                all_text = self._save_extracted_text(file, all_text)

        return 

    def _fragmentar_texto(
        self, chunks_size: int = 512, chunk_overlap: int = 100
    ) -> list:
        """
        Fragment the text into smaller chunks.
        Args:
            texto (str): The text to be fragmented.
        Returns:
            list: A list of Document objects containing the fragmented text.
        """
        docs = []
        extracted_text_dir = os.path.join(self.path, "extractedText")
        
        for fname in os.listdir(extracted_text_dir):
            fpath = os.path.join(extracted_text_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(".txt"):
                print("Extracting chunks from ", fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                all_splits = [
                    Document(page_content=t, metadata={"source": f"{fname}_chunk_{i}"}) 
                    for i, t in enumerate(self.splitter.split_text(text))
                    ]
                docs.extend(all_splits)
                
        return docs

    def _index(self, docs):
        if docs:
            print("Indexing...")
            vector_store = FAISS.from_documents(docs[:99], self.embeddings)
            print("Waiting 1 min for the RPM Limit on Google Embedding free tier")
            time.sleep(60)
            vector_store.add_documents(docs[99:])
            vector_store.save_local(os.path.join(DATA_DIR, "index.json"))
            return "Indexed stored" 
        else:
            vector_store = None
            return "Not documents indexed"
    
    def _preprocess(self, chunks_size: int = 512, chunk_overlap: int = 100) -> list:
        """
        Preprocess the PDF file by extracting text and fragmenting it into smaller chunks.
        Args:
            chunks_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between chunks.
        Returns:
            list: A list of Document objects containing the fragmented text.
        """

        self._extract_text_from_pdf()
        
        docs = self._fragmentar_texto(
            chunks_size=chunks_size, chunk_overlap=chunk_overlap
        )
        
        return self._index(docs)


if __name__ == "__main__":
    # Context document vector_store (for RAG retrieval)
    # embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    # vector_store = InMemoryVectorStore(embeddings)
    # text_splitter = MarkdownHeaderTextSplitter([("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),])
    # _load_data(vector_store, text_splitter)

    print(BasePreprocessing("data/docs")._preprocess())
