from claim_risk_predictor import ClaimRiskPredictor
import json

def main():
    predictor = ClaimRiskPredictor()
    terms_file = "./data/car.pdf"
    claim_file = "./data/car.pdf"
    query = "Analyze the claim risk considering policy terms and claim history"
    
    result = predictor.process_claim(terms_file=terms_file,claim_file=claim_file, query=query)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()