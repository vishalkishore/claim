from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Optional
import pandas as pd
import os
import re
from utils.text_preprocessing import clean_text, detect_document_type
from utils.logging import setup_logging

logger = setup_logging()

class DocumentProcessor:
    def __init__(self):
        self.section_headers = [
            # Common headers in T&C and insurance documents
            r"(?i)^article\s+\d+",
            r"(?i)^section\s+\d+",
            r"(?i)^\d+\.\s+[A-Z]",
            r"(?i)^[A-Z][A-Z\s]+\:",
            r"(?i)^CHAPTER\s+\d+",
            # Insurance specific patterns
            r"(?i)^COVERAGE\s+[A-Z]",
            r"(?i)^EXCLUSIONS?",
            r"(?i)^DEFINITIONS?",
            r"(?i)^CONDITIONS?"
        ]
        self.header_pattern = "|".join(self.section_headers)

    def detect_structure_type(self, text: str) -> str:
        """Detect the type of document structure"""
        header_matches = len(re.findall(self.header_pattern, text, re.MULTILINE))
        if header_matches > 5:
            return "structured"
        return "standard"

    def create_text_splitter(self, structure_type: str) -> RecursiveCharacterTextSplitter:
        """Create appropriate text splitter based on document structure"""
        if structure_type == "structured":
            # For structured documents, split on section headers and preserve them
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                keep_separator=True,
                add_start_index=True,
                strip_whitespace=True
            )
        else:
            # For standard documents, use more aggressive splitting
            return RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ".", " "],
                keep_separator=False
            )

    def split_by_sections(self, text: str) -> List[str]:
        """Split text by section headers while preserving the headers"""
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if re.match(self.header_pattern, line.strip()):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections

    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process documents with enhanced preprocessing and intelligent splitting"""
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
                
                # Basic metadata
                metadata = {
                    "file_name": os.path.basename(file_path),
                    "doc_type": detect_document_type(docs[0].page_content),
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "preprocessed": True
                }
                
                for doc in docs:
                    # Clean and preprocess text
                    cleaned_content = clean_text(doc.page_content)
                    if not cleaned_content:
                        continue
                    
                    # Detect document structure
                    structure_type = self.detect_structure_type(cleaned_content)
                    
                    # Create appropriate splitter
                    text_splitter = self.create_text_splitter(structure_type)
                    
                    if structure_type == "structured":
                        # For structured documents, first split by sections
                        sections = self.split_by_sections(cleaned_content)
                        
                        # Then apply the text splitter to each section
                        for section in sections:
                            splits = text_splitter.split_text(section)
                            
                            for split in splits:
                                # Create new document for each split
                                split_metadata = metadata.copy()
                                split_metadata.update({
                                    "structure_type": structure_type,
                                    "word_count": len(split.split()),
                                    "char_count": len(split),
                                    "section_header": re.match(self.header_pattern, split.strip(), re.MULTILINE)
                                })
                                
                                processed_docs.append(Document(
                                    page_content=split,
                                    metadata=split_metadata
                                ))
                    else:
                        # For standard documents, just use the text splitter
                        splits = text_splitter.split_text(cleaned_content)
                        
                        for split in splits:
                            split_metadata = metadata.copy()
                            split_metadata.update({
                                "structure_type": structure_type,
                                "word_count": len(split.split()),
                                "char_count": len(split)
                            })
                            
                            processed_docs.append(Document(
                                page_content=split,
                                metadata=split_metadata
                            ))
                            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        if not processed_docs:
            logger.warning("No documents were successfully processed")
            
        return processed_docs

    def get_section_titles(self, docs: List[Document]) -> List[str]:
        """Extract section titles from processed documents"""
        titles = []
        for doc in docs:
            content = doc.page_content.strip()
            if re.match(self.header_pattern, content, re.MULTILINE):
                titles.append(content.split('\n')[0])
        return list(set(titles))