import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from src.data.dataset import load_data, split_data
from src.features.build_features import preprocess
from src.evaluation.metrics import evaluate

def train(config_path: str = "configs/default.yaml"):
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
    model = RandomForestClassifier(**config["model"]["params"])
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    print("Evaluation:", metrics)

    # Save model
    joblib.dump(model, "models/model.pkl")

    return model, metrics
