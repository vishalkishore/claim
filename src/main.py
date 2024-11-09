from claim_risk_predictor import ClaimRiskPredictor
import json

def main():
    predictor = ClaimRiskPredictor()
    file_paths = ["./data/car.pdf"]
    query = "Analyze the claim risk considering policy terms and claim history"
    
    result = predictor.process_claim(file_paths, query)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()