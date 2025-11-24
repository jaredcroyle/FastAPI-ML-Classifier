import torch
import pandas as pd
from typing import List, Dict, Union
import numpy as np

def process_tabular_data(data: Dict[str, Union[float, int]]) -> torch.Tensor:
    """
    Process tabular data from API request to tensor.
    Input: Dictionary with keys A0, A1, ..., A180 (Splice-junction Gene Sequence Dataset from UCI Machine Learning Repository)
    Output: PyTorch tensor of shape (1, num_features)
    """
    # converts to list of feature values, order A0 to A180
    features = [data[f"A{i}"] for i in range(180)]
    return torch.FloatTensor([features])

def load_csv_data(filepath: str):
    """
    Load and preprocess CSV dataset
    Returns: (features, labels) as numpy arrays
    """
    df = pd.read_csv(filepath)
    X = df.drop('class', axis=1).values
    y = df['class'].values
    return X, y
