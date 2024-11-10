from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import os
import re
from utils.text_preprocessing import clean_text, detect_document_type
from utils.logging import setup_logging
from dataclasses import dataclass
from datetime import datetime

logger = setup_logging()

class DocumentType:
    TERMS = "terms_and_conditions"
    CLAIM = "claim_report"

@dataclass
class ProcessedDocument:
    content: str
    doc_type: str
    metadata: Dict
    sections: List[str]

class DocumentProcessor:
    def __init__(self):
        # Fixed regex patterns with proper anchoring and flags
        self.section_patterns = [
            # Basic numbered sections
            r"^\s*(?:Section|SECTION)\s+\d+(?:\.\d+)*",
            r"^\s*\d+(?:\.\d+)*\s+[A-Z]",
            # Headers in ALL CAPS
            r"^\s*[A-Z][A-Z\s]{2,}(?:\s*\(.*\))?:?$",
            # Insurance specific sections
            r"^\s*(?:COVERAGE|EXCLUSIONS?|CONDITIONS?|DEFINITIONS?)\s*:?$",
            # Articles and chapters
            r"^\s*(?:Article|ARTICLE|Chapter|CHAPTER)\s+\d+(?:\.\d+)*",
            # Common document sections
            r"^\s*(?:PURPOSE|SCOPE|INTRODUCTION|BACKGROUND|SUMMARY|CONCLUSION)s*:?$",
            # Appendices and exhibits
            r"^\s*(?:Appendix|APPENDIX|Exhibit|EXHIBIT)\s+[A-Z\d]+"
        ]
        self.header_pattern = "|".join(self.section_patterns)

    def detect_structure_type(self, text: str) -> str:
        """Detect the type of document structure"""
        header_matches = len(re.findall(self.header_pattern, text, re.MULTILINE))
        if header_matches > 5:
            return "structured"
        return "standard"

    def create_text_splitter(self, structure_type: str) -> RecursiveCharacterTextSplitter:
        """Create appropriate text splitter based on document structure"""
        if structure_type == "structured":
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

    def process_single_document(self, file_path: str) -> List[Document]:
        """Process a single document and return list of processed chunks"""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
        
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            if not docs:
                logger.warning(f"No content extracted from: {file_path}")
                return []
            
            processed_docs = []
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
                
                structure_type = self.detect_structure_type(cleaned_content)
                text_splitter = self.create_text_splitter(structure_type)
                
                if structure_type == "structured":
                    sections = self.split_by_sections(cleaned_content)
                    for section in sections:
                        splits = text_splitter.split_text(section)
                        for split in splits:
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
                        
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def get_section_titles(self, docs: List[Document]) -> List[str]:
        """Extract section titles from processed documents"""
        titles = []
        for doc in docs:
            content = doc.page_content.strip()
            if re.match(self.header_pattern, content, re.MULTILINE):
                titles.append(content.split('\n')[0])
        return list(set(titles))

    def process_documents(self, terms_file: str, claim_file: str) -> Tuple[ProcessedDocument, ProcessedDocument]:
        """Process both terms and claims documents and return as ProcessedDocument objects"""
        # Process terms and conditions
        terms_docs = self.process_single_document(terms_file)
        terms_content = "\n".join([doc.page_content for doc in terms_docs])
        terms_sections = self.get_section_titles(terms_docs)
        
        terms = ProcessedDocument(
            content=terms_content,
            doc_type=DocumentType.TERMS,
            metadata={
                "file_name": os.path.basename(terms_file),
                "processed_at": datetime.now().isoformat(),
                "section_count": len(terms_sections)
            },
            sections=terms_sections
        )
        
        # Process claim report
        claim_docs = self.process_single_document(claim_file)
        claim_content = "\n".join([doc.page_content for doc in claim_docs])
        claim_sections = self.get_section_titles(claim_docs)
        
        claim = ProcessedDocument(
            content=claim_content,
            doc_type=DocumentType.CLAIM,
            metadata={
                "file_name": os.path.basename(claim_file),
                "processed_at": datetime.now().isoformat(),
                "section_count": len(claim_sections)
            },
            sections=claim_sections
        )
        
        return terms, claim