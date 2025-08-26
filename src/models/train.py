import joblib
import yaml
from src.data.dataset import load_data, split_data
from src.features.build_features import preprocess
from src.evaluation.metrics import evaluate
from src.models.model_factory import create_model

def train(config_path: str = "configs/random_forest.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load & preprocess
    df = load_data(config["data"]["raw_path"])
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(
        df, config["training"]["target_column"], config["training"]["test_size"]
    )

    # Init model
    model = create_model(config["model"])
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    print("Evaluation:", metrics)

    # Save model with type-specific filename
    model_type = config["model"]["type"]
    model_filename = f"models/{model_type.lower()}_model.pkl"
    joblib.dump(model, model_filename)
    
    # Also save model metadata
    import json
    from datetime import datetime
    metadata = {
        "model_type": model_type,
        "model_params": config["model"]["params"],
        "training_date": datetime.now().isoformat(),
        "config_path": config_path,
        "metrics": metrics
    }
    metadata_filename = f"models/{model_type.lower()}_metadata.json"
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved as: {model_filename}")
    print(f"Metadata saved as: {metadata_filename}")

    return model, metrics
