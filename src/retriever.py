from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
from utils.logging import setup_logging

logger = setup_logging()

class RetrievalSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def setup_retrievers(self, documents: List[Document]) -> EnsembleRetriever:
        """Set up multiple retrievers with BM25 and vector store"""
        try:
            if not documents:
                raise ValueError("No documents provided for retriever setup")

            # Initialize BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=3
            )
            
            # Initialize vector store retriever
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            vectorstore_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 2}
            )
            
            # Combine retrievers
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vectorstore_retriever],
                weights=[0.5, 0.5]
            )
            
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {str(e)}")
            raise