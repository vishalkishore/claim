from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from retriever import RetrievalSystem
from utils.logging import setup_logging

logger = setup_logging()

class ClaimRiskPredictor:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.setup_components()

    def setup_components(self):
        """Initialize components with document type-specific processing"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.google_api_key,
                model=self.model_name,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                google_api_key=self.google_api_key,
                model="models/embedding-001"
            )
            
            self.risk_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert claims analyst. Compare the submitted claim report and estimated bill of repairs
                against the relevant terms and conditions to assess risk and validity.
                
                Important Note: The submitted documents are standardized compilations of the originals. Treat them as accurate and authentic representations of the full original records.

                # Calculate a risk score based on these weights and factors:

                Risk Score = (0.3)*(Policy Compliance) + (0.3)*(Documentation Quality) + (0.4)*(Fraud Risk Indicators)

                Where:
                1. Policy Compliance (0-1):
                - Full compliance = 0
                - Minor violations = 0.3-0.5
                - Major violations = 0.6-1.0

                2. Documentation Quality (0-1):
                - Complete documentation = 0
                - Minor gaps = 0.3-0.5
                - Major gaps = 0.6-1.0

                3. Fraud Risk Indicators (0-1):
                - Calculated based on presence of:
                    * Timing inconsistencies (+0.2)
                    * Unusual claim amounts (+0.2)
                    * Missing third-party verification (+0.2)
                    * Document alterations (+0.2)
                    * Prior suspicious claims (+0.2)

                Final Score Interpretation:
                - 0.0-0.3: Low Risk
                - 0.3-0.6: Medium Risk
                - 0.6-0.8: High Risk
                - 0.8-1.0: Critical Risk
                 
                IMPORTANT: 
                - Do not hallucinate. Do not make up factual information.
                - Return only valid JSON as output, in the format provided below. Avoid any extra text or explanation outside the JSON.
            
                Terms & Conditions Context:
                {terms_and_conditions}
                
                Claim Report Context:
                {claim_report}
                
                Analyze the following aspects:
                1. Compliance with policy terms
                2. Completeness of claim documentation
                3. Consistency between report and policy coverage
                4. Specific risk indicators
                 
                Provide your analysis in the following JSON format:
                {{
                    "risk_score": <float between 0-1>,
                    "policy_violations": ["<list of specific terms & conditions violations>"],
                    "documentation_gaps": ["<list of missing or incomplete documentation>"],
                    "risk_indicators": ["<list of specific risk flags identified>"],
                    "validity_assessment": {{
                        "reasoning": "<detailed explanation>",
                        "policy_references": ["<specific sections from T&C that support the assessment>"]
                    }},
                    "recommended_actions": ["<list of specific next steps>"],
                    "confidence_score": <float between 0-1>
                }}"""),
                ("human", "\nQuestion: {question}")
            ])
            
            self.document_processor = DocumentProcessor()
            self.retrieval_system = RetrievalSystem(self.embedding_model)
            
        except Exception as e:
            logger.error(f"Error setting up components: {str(e)}")
            raise

    def separate_documents(self, documents: List[Any]) -> Dict[str, List[Any]]:
        """Separate documents into terms & conditions and claim reports"""
        separated_docs = {
            "terms_and_conditions": [],
            "claim_reports": []
        }
        
        for doc in documents:
            doc_type = doc.metadata.get('doc_type', '').lower()
            if 'terms' in doc_type or 'conditions' in doc_type or 't&c' in doc_type:
                separated_docs["terms_and_conditions"].append(doc)
            elif 'claim' in doc_type or 'report' in doc_type:
                separated_docs["claim_reports"].append(doc)
                
        return separated_docs

    def format_documents(self, docs: List[Any]) -> str:
        """Format documents with metadata and content"""
        return "\n\n".join([
            f"Section: {doc.metadata.get('section', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        ])

    def count_documents(self, file_paths: List[str]) -> int:
        """Count the number of document files"""
        if isinstance(file_paths, str):
            return 1
        return len(file_paths) if file_paths else 0

    def process_claim(self, terms_file: str, claim_file: str, query: str) -> Dict[str, Any]:
        """Process claims by comparing terms & conditions against claim reports"""
        try:
            # Get initial document counts
            total_files = self.count_documents([terms_file]) + self.count_documents([claim_file])
            
            # Process all documents
            documents = self.document_processor.process_documents(terms_file=terms_file, claim_file=claim_file)
            
            if not documents:
                return {
                    "error": "No documents were successfully processed",
                    "metadata": {"processed_files": total_files, "successful_files": 0}
                }
            
            # Separate documents by type
            separated_docs = {
                "terms_and_conditions": documents[0],
                "claim_reports": documents[1]
            }
            
            if not separated_docs["terms_and_conditions"] or not separated_docs["claim_reports"]:
                return {
                    "error": "Missing required document types. Need both terms & conditions and claim reports.",
                    "metadata": {
                        "found_terms": bool(separated_docs["terms_and_conditions"]),
                        "found_claims": bool(separated_docs["claim_reports"])
                    }
                }
            
            # Set up retrieval for both document types
            tc_retriever = self.retrieval_system.setup_retrievers(separated_docs["terms_and_conditions"])
            claim_retriever = self.retrieval_system.setup_retrievers(separated_docs["claim_reports"])
            
            # Get relevant sections from both document types
            relevant_tc = tc_retriever.get_relevant_documents(query)
            relevant_claims = claim_retriever.get_relevant_documents(query)
            
            rag_chain = (
                {
                    "terms_and_conditions": lambda x: self.format_documents(relevant_tc),
                    "claim_report": lambda x: self.format_documents(relevant_claims),
                    "question": RunnablePassthrough()
                }
                | self.risk_prompt
                | self.llm
                | JsonOutputParser()
            )
            
            response = rag_chain.invoke(query)
            
            # Count sections safely
            tc_sections = 1 if separated_docs["terms_and_conditions"] else 0
            claim_sections = 1 if separated_docs["claim_reports"] else 0
            
            return {
                "analysis": response,
                "metadata": {
                    "processed_files": total_files,
                    "terms_and_conditions": {
                        "total_sections": tc_sections,
                        "relevant_sections": len(relevant_tc) if relevant_tc else 0
                    },
                    "claim_reports": {
                        "total_documents": claim_sections,
                        "relevant_sections": len(relevant_claims) if relevant_claims else 0
                    },
                    "top_referenced_sections": [
                        {
                            "file_name": doc.metadata.get('file_name', 'Unknown'),
                            "section": doc.metadata.get('section', 'Unknown')
                        }
                        for doc in (relevant_tc + relevant_claims)[:5]
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            return {
                "error": str(e),
                "metadata": {
                    "processed_files": total_files if 'total_files' in locals() else 0
                }
            }