
def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    text = " ".join(text.split())
    text = ''.join(char for char in text if char.isalnum() or char in '. ,?!-()[]{}:;')
    return text

def detect_document_type(content: str) -> str:
    """Detect document type based on content keywords"""
    content = content.lower()
    
    patterns = {
        "claim_report": ["claim", "incident", "accident", "damage", "loss"],
        "policy_document": ["policy", "terms", "conditions", "coverage", "insurance"],
        "medical_report": ["diagnosis", "treatment", "medical", "physician", "patient"],
        "invoice": ["invoice", "bill", "payment", "amount", "due"],
        "correspondence": ["letter", "email", "correspondence", "regarding", "dear"]
    }
    
    matches = {doc_type: sum(1 for keyword in keywords if keyword in content)
              for doc_type, keywords in patterns.items()}
    
    max_matches = max(matches.values()) if matches else 0
    return "other" if max_matches == 0 else max(matches.items(), key=lambda x: x[1])[0]