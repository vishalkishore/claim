import gradio as gr
import json
from claim_risk_predictor import ClaimRiskPredictor  # Import the previous class
import tempfile
import os
import logging
from typing import List, Dict, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimRiskUI:
    def __init__(self):
        self.predictor = ClaimRiskPredictor()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Initialized temporary directory at {self.temp_dir}")

    def save_uploaded_files(self, files: List[tempfile.SpooledTemporaryFile]) -> List[str]:
        """Save uploaded files and return their paths"""
        saved_paths = []
        for file in files:
            if file is not None:
                temp_path = os.path.join(self.temp_dir, file.name)
                with open(temp_path, 'wb') as f:
                    f.write(file.read())
                saved_paths.append(temp_path)
        return saved_paths
    
    def save_uploaded_files(self, files: List[Union[tempfile.SpooledTemporaryFile, str]]) -> List[str]:
        """Save uploaded files and return their paths"""
        saved_paths = []
        
        for file in files:
            if file is not None:
                # Determine the file path and content to save
                if isinstance(file, tempfile.SpooledTemporaryFile):
                    temp_path = os.path.join(self.temp_dir, file.name)
                    content = file.read()
                elif isinstance(file, str):  # If file is a path as a string
                    temp_path = os.path.join(self.temp_dir, os.path.basename(file))
                    with open(file, 'rb') as f:
                        content = f.read()
                else:
                    continue  # Skip if the file type is unexpected
                
                # Write content to temp_path
                with open(temp_path, 'wb') as f:
                    f.write(content)
                saved_paths.append(temp_path)
        
        return saved_paths

    def cleanup_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned up file: {path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {path}: {str(e)}")

    def process_documents(self, 
                        files: List[tempfile.SpooledTemporaryFile], 
                        query: str) -> Dict[str, Any]:
        """Process uploaded documents and return analysis results"""
        try:
            if not files:
                return {
                    "error": "No files uploaded. Please upload at least one document.",
                    "metadata": {}
                }
            
            if not query.strip():
                return {
                    "error": "Please provide a query for analysis.",
                    "metadata": {}
                }

            # Save uploaded files
            file_paths = self.save_uploaded_files(files)
            logger.info(f"Saved {len(file_paths)} files for processing")

            # Process the claim
            result = self.predictor.process_claim(file_paths, query)

            # Cleanup temporary files
            self.cleanup_files(file_paths)

            return result

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return {
                "error": f"Error processing documents: {str(e)}",
                "metadata": {}
            }

    def format_output(self, result: Dict[str, Any]) -> str:
        """Format the analysis result for display"""
        if "error" in result:
            return f"Error: {result['error']}"

        output = []
        
        # Add analysis results
        if "analysis" in result:
            analysis = result["analysis"]
            output.append("📊 Risk Analysis:")
            output.append(f"• Risk Score: {analysis.get('risk_score', 'N/A'):.2f}")
            output.append(f"• Confidence Score: {analysis.get('confidence_score', 'N/A'):.2f}")
            output.append("\n🚩 Risk Factors:")
            for factor in analysis.get("risk_factors", []):
                output.append(f"• {factor}")
            output.append("\n📋 Validity Assessment:")
            output.append(analysis.get("validity_assessment", "N/A"))
            output.append("\n📝 Recommended Actions:")
            for action in analysis.get("recommended_actions", []):
                output.append(f"• {action}")

        # Add metadata
        if "metadata" in result:
            metadata = result["metadata"]
            output.append("\n📌 Processing Details:")
            output.append(f"• Processed Files: {metadata.get('processed_files', 0)}")
            output.append(f"• Successfully Processed: {metadata.get('successful_files', 0)}")
            output.append(f"• Retrieved Documents: {metadata.get('retrieved_docs', 0)}")
            
            if "top_documents" in metadata:
                output.append("\n📚 Top Referenced Documents:")
                for doc in metadata["top_documents"]:
                    output.append(f"• {doc.get('file_name', 'Unknown')} ({doc.get('doc_type', 'Unknown')})")

        return "\n".join(output)

    def create_ui(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Insurance Claim Risk Analyzer", 
                      theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 📄 Insurance Claim Risk Analyzer
            Upload claim-related documents and get an AI-powered risk analysis report.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        file_count="multiple",
                        label="Upload Documents",
                        file_types=[".pdf"],
                        scale=2
                    )
                    query_input = gr.Textbox(
                        label="Analysis Query",
                        placeholder="Enter your analysis query here...",
                        value="Analyze the claim risk considering policy terms and claim history"
                    )
                    analyze_button = gr.Button("🔍 Analyze Documents", variant="primary")

                with gr.Column(scale=3):
                    output_text = gr.Textbox(
                        label="Analysis Results",
                        show_copy_button=True,
                        interactive=False,
                        lines=20
                    )

            # gr.Examples(
            #     examples=[
            #         [["sample_claim.pdf"], "Analyze this claim for potential fraud indicators"],
            #         [["policy.pdf", "claim_form.pdf"], "Evaluate coverage compliance and claim validity"],
            #     ],
            #     inputs=[file_input, query_input],
            #     label="Example Queries"
            # )

            analyze_button.click(
                fn=lambda files, query: self.format_output(self.process_documents(files, query)),
                inputs=[file_input, query_input],
                outputs=output_text
            )

            gr.Markdown("""
            ### 📝 Instructions
            1. Upload one or more PDF documents related to the insurance claim
            2. Enter your analysis query or use the default query
            3. Click 'Analyze Documents' to get the risk assessment report
            
            ### 📊 Output Explanation
            - Risk Score: 0 (Low Risk) to 1 (High Risk)
            - Confidence Score: Indicates the AI's confidence in its analysis
            - Risk Factors: Identified potential issues
            - Recommended Actions: Suggested next steps
            """)

        return interface

def main():
    ui = ClaimRiskUI()
    interface = ui.create_ui()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()