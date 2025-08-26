import pandas as pd
from src.models.predict import predict

if __name__ == "__main__":
    # Example new input (gene expression)
    sample = pd.DataFrame({
        "gene1": [5.5, 6.1],
        "gene2": [3.2, 3.8],
        "gene3": [1.5, 4.7],
        "gene4": [0.3, 1.6],
    })
    preds = predict(sample)
    print("Predictions:", preds)
