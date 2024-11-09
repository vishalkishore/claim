from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from typing import List
import pandas as pd
import os
from utils.text_preprocessing import clean_text, detect_document_type
from utils.logging import setup_logging

logger = setup_logging()

class DocumentProcessor:
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process documents with enhanced preprocessing"""
        processed_docs = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                
                if not docs:
                    logger.warning(f"No content extracted from: {file_path}")
                    continue
                
                metadata = {
                    "file_name": os.path.basename(file_path),
                    "doc_type": detect_document_type(docs[0].page_content),
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "preprocessed": True
                }
                
                for doc in docs:
                    cleaned_content = clean_text(doc.page_content)
                    if not cleaned_content:
                        continue
                        
                    doc.page_content = cleaned_content
                    doc.metadata.update(metadata)
                    doc.metadata["word_count"] = len(cleaned_content.split())
                    doc.metadata["char_count"] = len(cleaned_content)
                    
                    processed_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        if not processed_docs:
            logger.warning("No documents were successfully processed")
            
        return processed_docs