import pandas as pd
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import predict, list_available_models

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained ML models")
    parser.add_argument(
        "--model",
        type=str,
        help="Model type to use for inference (e.g., 'randomforestclassifier', 'svc')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Direct path to model file"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available trained models"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to CSV file with input data for prediction"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        if not models:
            print("No trained models found.")
        else:
            print("Available trained models:")
            for model_name, metadata in models.items():
                print(f"  - {model_name}:")
                print(f"    Type: {metadata.get('model_type', 'Unknown')}")
                print(f"    Trained: {metadata.get('training_date', 'Unknown')}")
                print(f"    Accuracy: {metadata.get('metrics', {}).get('accuracy', 'Unknown')}")
        return
    
    # Prepare input data
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            return
        sample = pd.read_csv(args.input_file)
        print(f"Loaded input data from: {args.input_file}")
    else:
        # Use example data
        sample = pd.DataFrame({
            "gene1": [5.5, 6.1],
            "gene2": [3.2, 3.8],
            "gene3": [1.5, 4.7],
            "gene4": [0.3, 1.6],
        })
        print("Using example input data")
    
    try:
        # Run prediction
        if args.model_path:
            preds = predict(sample, model_path=args.model_path)
            print(f"Predictions using {args.model_path}: {preds}")
        elif args.model:
            preds = predict(sample, model_type=args.model)
            print(f"Predictions using {args.model} model: {preds}")
        else:
            preds = predict(sample)
            print(f"Predictions: {preds}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nTip: Use --list-models to see available trained models")

if __name__ == "__main__":
    main()
