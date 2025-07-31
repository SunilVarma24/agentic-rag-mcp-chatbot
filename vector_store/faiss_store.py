# vector_store/faiss_store.py

from typing import List
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class FAISSVectorStore:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize FAISS vector store with HuggingFace embeddings.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None

    def build_index(self, documents: List[Document]):
        """
        Build a FAISS index from a list of LangChain Documents.

        Args:
            documents (List[Document]): Chunked documents to index.
        """
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)

    def as_retriever(self, k: int = 5):
        """
        Return retriever interface from current FAISS index.

        Args:
            k (int): Number of top documents to retrieve.

        Returns:
            BaseRetriever: LangChain retriever.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

