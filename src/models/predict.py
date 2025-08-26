import joblib
import pandas as pd
from src.features.build_features import preprocess

def predict(input_df: pd.DataFrame, model_path: str = "models/model.pkl"):
    """Run inference using a trained model."""
    model = joblib.load(model_path)
    processed = preprocess(input_df)
    return model.predict(processed)
