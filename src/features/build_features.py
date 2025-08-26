import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Example preprocessing: fill missing values."""
    return df.fillna(0)
