import joblib
import pandas as pd
import json
import os
from src.features.build_features import preprocess

def predict(input_df: pd.DataFrame, model_path: str = None, model_type: str = None):
    """
    Run inference using a trained model.
    
    Args:
        input_df: Input data for prediction
        model_path: Direct path to model file (takes precedence)
        model_type: Model type name (e.g., 'randomforestclassifier', 'svc')
        
    Returns:
        Predictions array
    """
    if model_path:
        # Use provided model path directly
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
    elif model_type:
        # Use model type to construct path
        model_path = f"models/{model_type.lower()}_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found for type: {model_type}")
        model = joblib.load(model_path)
    else:
        # Try to find any available model, prefer the most recent
        model_files = [f for f in os.listdir("models") if f.endswith("_model.pkl")]
        if not model_files:
            raise FileNotFoundError("No trained models found. Please train a model first.")
        
        # Use the most recently modified model
        model_files.sort(key=lambda f: os.path.getmtime(f"models/{f}"), reverse=True)
        model_path = f"models/{model_files[0]}"
        print(f"Using most recent model: {model_path}")
        model = joblib.load(model_path)
    
    processed = preprocess(input_df)
    return model.predict(processed)

def list_available_models():
    """List all available trained models with their metadata."""
    if not os.path.exists("models"):
        return {}
    
    models = {}
    for filename in os.listdir("models"):
        if filename.endswith("_metadata.json"):
            model_name = filename.replace("_metadata.json", "")
            try:
                with open(f"models/{filename}", "r") as f:
                    metadata = json.load(f)
                models[model_name] = metadata
            except Exception as e:
                print(f"Error reading metadata for {model_name}: {e}")
    
    return models
