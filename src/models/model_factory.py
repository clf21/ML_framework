from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

MODEL_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "KNeighborsClassifier": KNeighborsClassifier
}

def create_model(model_config):
    """
    Create a model instance based on configuration.
    
    Args:
        model_config (dict): Configuration containing 'type' and 'params'
        
    Returns:
        sklearn classifier instance
        
    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_config["type"]
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_type]
    model_params = model_config.get("params", {})
    
    return model_class(**model_params)

def get_supported_models():
    """Return list of supported model types."""
    return list(MODEL_REGISTRY.keys())