from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Union, Tuple, Dict
from utils.logging import setup_logging
from dataclasses import dataclass

logger = setup_logging()

@dataclass
class ProcessedDocument:
    content: str
    doc_type: str
    metadata: Dict
    sections: List[str]

class RetrievalSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def _convert_to_documents(self, processed_doc: ProcessedDocument) -> List[Document]:
        """Convert ProcessedDocument to list of LangChain Documents"""
        # Split content into manageable chunks (reusing section splits)
        chunks = processed_doc.sections if processed_doc.sections else [processed_doc.content]
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Create metadata for each chunk
            chunk_metadata = {
                **processed_doc.metadata,  # Include original metadata
                "chunk_index": i,
                "doc_type": processed_doc.doc_type,
                "total_chunks": len(chunks)
            }
            
            # Create LangChain Document
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents

    def prepare_documents(self, 
                         terms_doc: ProcessedDocument, 
                         claim_doc: ProcessedDocument) -> List[Document]:
        """Prepare both documents for retrieval"""
        terms_documents = self._convert_to_documents(terms_doc)
        claim_documents = self._convert_to_documents(claim_doc)
        
        # Combine all documents
        all_documents = terms_documents + claim_documents
        
        if not all_documents:
            logger.warning("No documents were prepared for retrieval")
            
        return all_documents

    def setup_retrievers(self, 
                        documents: ProcessedDocument, 
                        ) -> EnsembleRetriever:
        """Set up multiple retrievers with BM25 and vector store"""
        try:
            # Convert ProcessedDocuments to LangChain Documents
            # documents = self.prepare_documents(terms_doc, claim_doc)
            documents = self._convert_to_documents(documents)
            if not documents:
                raise ValueError("No documents available for retriever setup")

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
            
            logger.info(f"Successfully set up retrieval system with {len(documents)} documents")
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {str(e)}")
            raise

    def retrieve_relevant_context(self, 
                                retriever: EnsembleRetriever, 
                                query: str,
                                k: int = 5) -> List[Document]:
        """Retrieve relevant context using the ensemble retriever"""
        try:
            retrieved_docs = retriever.get_relevant_documents(query, k=k)
            
            # Log retrieval statistics
            doc_types = {}
            for doc in retrieved_docs:
                doc_type = doc.metadata.get('doc_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
            logger.info(f"Retrieved {len(retrieved_docs)} documents: {doc_types}")
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
