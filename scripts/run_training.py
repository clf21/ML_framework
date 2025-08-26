import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import train
from src.models.model_factory import get_supported_models

def main():
    parser = argparse.ArgumentParser(description="Train ML models with different configurations")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["random_forest", "gradient_boosting", "svm", "logistic_regression", "decision_tree"],
        default="random_forest",
        help="Model type to train (default: random_forest)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file (overrides --model)"
    )
    
    args = parser.parse_args()
    
    # Map model names to config files
    model_configs = {
        "random_forest": "configs/default.yaml",
        "gradient_boosting": "configs/gradient_boosting.yaml", 
        "svm": "configs/svm.yaml",
        "logistic_regression": "configs/logistic_regression.yaml",
        "decision_tree": "configs/decision_tree.yaml"
    }
    
    # Use custom config if provided, otherwise use model-specific config
    if args.config:
        config_path = args.config
        print(f"Training with custom config: {config_path}")
    else:
        config_path = model_configs[args.model]
        print(f"Training {args.model} model with config: {config_path}")
    
    # Train the model
    model, metrics = train(config_path)
    print(f"Training completed. Model saved to models/model.pkl")

if __name__ == "__main__":
    main()
