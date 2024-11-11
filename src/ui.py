import gradio as gr
import json
from claim_risk_predictor import ClaimRiskPredictor
import tempfile
import os
import logging
from typing import Dict, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimRiskUI:
    def __init__(self):
        self.predictor = ClaimRiskPredictor()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Initialized temporary directory at {self.temp_dir}")

    def save_uploaded_file(self, file: Union[tempfile.SpooledTemporaryFile, str]) -> str:
        """Save a single uploaded file and return its path"""
        if file is None:
            return None
            
        try:
            if isinstance(file, tempfile.SpooledTemporaryFile):
                temp_path = os.path.join(self.temp_dir, file.name)
                content = file.read()
            elif isinstance(file, str):
                temp_path = os.path.join(self.temp_dir, os.path.basename(file))
                with open(file, 'rb') as f:
                    content = f.read()
            else:
                return None

            with open(temp_path, 'wb') as f:
                f.write(content)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return None

    def cleanup_files(self, *file_paths: str):
        """Clean up temporary files"""
        for path in file_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up file: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {path}: {str(e)}")

    def process_documents(self, 
                        terms_file: tempfile.SpooledTemporaryFile,
                        claim_file: tempfile.SpooledTemporaryFile, 
                        query: str) -> Dict[str, Any]:
        """Process the terms and claim documents and return analysis results"""
        try:
            if not terms_file or not claim_file:
                return {
                    "error": "Please upload both terms and claim documents.",
                    "metadata": {}
                }
            
            if not query.strip():
                return {
                    "error": "Please provide a query for analysis.",
                    "metadata": {}
                }

            # Save uploaded files
            terms_path = self.save_uploaded_file(terms_file)
            claim_path = self.save_uploaded_file(claim_file)

            if not terms_path or not claim_path:
                return {
                    "error": "Error saving uploaded files.",
                    "metadata": {}
                }

            logger.info(f"Processing terms file: {terms_path}")
            logger.info(f"Processing claim file: {claim_path}")

            # Process the claim
            result = self.predictor.process_claim(
                terms_file=terms_path,
                claim_file=claim_path,
                query=query
            )

            # Cleanup temporary files
            self.cleanup_files(terms_path, claim_path)

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
            
            # Convert all values to strings before adding to output
            risk_score = analysis.get('risk_score')
            if isinstance(risk_score, (int, float)):
                output.append(f"• Risk Score: {risk_score:.2f}")
            else:
                output.append(f"• Risk Score: N/A")
                
            confidence_score = analysis.get('confidence_score')
            if isinstance(confidence_score, (int, float)):
                output.append(f"• Confidence Score: {confidence_score:.2f}")
            else:
                output.append(f"• Confidence Score: N/A")

            # Handle risk factors
            output.append("\n🚩 Risk Factors:")
            risk_factors = analysis.get("risk_factors", [])
            if isinstance(risk_factors, list):
                for factor in risk_factors:
                    output.append(f"• {str(factor)}")
            else:
                output.append("• No risk factors identified")

            # Handle validity assessment
            output.append("\n📋 Validity Assessment:")
            validity = analysis.get("validity_assessment")
            output.append(str(validity) if validity else "N/A")

            # Handle recommended actions
            output.append("\n📝 Recommended Actions:")
            actions = analysis.get("recommended_actions", [])
            if isinstance(actions, list):
                for action in actions:
                    output.append(f"• {str(action)}")
            else:
                output.append("• No recommended actions")

        # Add metadata
        if "metadata" in result:
            metadata = result["metadata"]
            
            coverage_match = metadata.get('coverage_match')
            if isinstance(coverage_match, (int, float)):
                output.append(f"• Coverage Match: {coverage_match:.1f}%")
            
            policy_compliance = metadata.get('policy_compliance')
            if isinstance(policy_compliance, (int, float)):
                output.append(f"• Policy Compliance: {policy_compliance:.1f}%")

        return "\n".join(output)

    def create_ui(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Insurance Claim Risk Analyzer", 
                      theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 📄 Insurance Claim Risk Analyzer
            Upload policy terms and claim documents for AI-powered risk analysis.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    terms_file = gr.File(
                        label="Upload Policy Terms Document",
                        file_types=[".pdf"],
                        scale=1
                    )
                    claim_file = gr.File(
                        label="Upload Claim Document",
                        file_types=[".pdf"],
                        scale=1
                    )
                    query_input = gr.Textbox(
                        label="Analysis Query",
                        placeholder="Enter your analysis query here...",
                        value="Analyze the claim risk considering policy terms and claim report containing the estimated bill of repairs"
                    )
                    analyze_button = gr.Button("🔍 Analyze Documents", variant="primary")

                with gr.Column(scale=3):
                    output_text = gr.Textbox(
                        label="Analysis Results",
                        show_copy_button=True,
                        interactive=False,
                        lines=20
                    )

            analyze_button.click(
                fn=lambda terms, claim, query: self.format_output(
                    self.process_documents(terms, claim, query)
                ),
                inputs=[terms_file, claim_file, query_input],
                outputs=output_text
            )

            gr.Markdown("""
            ### 📝 Instructions
            1. Upload the policy terms document (PDF)
            2. Upload the claim document (PDF)
            3. Enter your analysis query or use the default query
            4. Click 'Analyze Documents' to get the risk assessment report
            
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