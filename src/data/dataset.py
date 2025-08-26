import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    """Load CSV dataset."""
    return pd.read_csv(path)

def split_data(df, target_col: str, test_size: float = 0.2):
    """Split into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)
