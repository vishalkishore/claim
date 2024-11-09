# src/claim_risk_predictor.py
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
        """Initialize components with enhanced retrieval setup"""
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
                ("system", """You are an expert claims analyst. Analyze the provided document context 
                to assess claim risk and validity. Consider:
                1. Policy terms and conditions
                2. Claim details and documentation
                3. Historical claim patterns
                4. Risk factors and red flags
                
                Provide your analysis in the following JSON format:
                {{
                    "risk_score": <float between 0-1>,
                    "risk_factors": [<list of identified risk factors>],
                    "validity_assessment": "<detailed assessment>",
                    "recommended_actions": [<list of recommended actions>],
                    "confidence_score": <float between 0-1>
                }}"""),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])
            
            self.document_processor = DocumentProcessor()
            self.retrieval_system = RetrievalSystem(self.embedding_model)
            
        except Exception as e:
            logger.error(f"Error setting up components: {str(e)}")
            raise

    def process_claim(self, file_paths: List[str], query: str) -> Dict[str, Any]:
        """Process claims with enhanced retrieval and ranking"""
        try:
            documents = self.document_processor.process_documents(file_paths)
            
            if not documents:
                return {
                    "error": "No documents were successfully processed",
                    "metadata": {
                        "processed_files": len(file_paths),
                        "successful_files": 0
                    }
                }
            
            ensemble_retriever = self.retrieval_system.setup_retrievers(documents)
            retrieved_docs = ensemble_retriever.get_relevant_documents(query)
            
            if not retrieved_docs:
                return {
                    "error": "No relevant documents found",
                    "metadata": {
                        "processed_files": len(file_paths),
                        "successful_files": len(documents)
                    }
                }
            
            def format_docs(docs):
                return "\n\n".join([
                    f"Source: {doc.metadata.get('file_name', 'Unknown')}\n"
                    f"Type: {doc.metadata.get('doc_type', 'Unknown')}\n"
                    f"Content: {doc.page_content}"
                    for doc in docs
                ])

            rag_chain = (
                {
                    "context": lambda x: format_docs(retrieved_docs),
                    "question": RunnablePassthrough()
                }
                | self.risk_prompt
                | self.llm
                | JsonOutputParser()
            )
            
            response = rag_chain.invoke(query)
            
            return {
                "analysis": response,
                "metadata": {
                    "processed_files": len(file_paths),
                    "successful_files": len(documents),
                    "retrieved_docs": len(retrieved_docs),
                    "top_documents": [
                        {
                            "file_name": doc.metadata.get('file_name', 'Unknown'),
                            "doc_type": doc.metadata.get('doc_type', 'Unknown')
                        }
                        for doc in retrieved_docs[:3]
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            return {
                "error": str(e),
                "metadata": {
                    "processed_files": len(file_paths),
                    "successful_files": len(documents) if 'documents' in locals() else 0
                }
            }