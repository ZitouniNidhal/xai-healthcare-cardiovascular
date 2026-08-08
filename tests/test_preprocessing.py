import pytest
import pandas as pd
import numpy as np
from src.utils.config import Config
from src.data.load_dataset import generate_mock_tabular_data
from src.data.preprocessing import preprocess_tabular_data

def test_tabular_preprocessing():
    # Setup dummy configurations
    config = Config()
    
    # Generate mock dataframe
    df = generate_mock_tabular_data(n_samples=50)
    
    # Apply preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_tabular_data(df, config)
    
    # Check shape constraints
    assert len(X_train) == 40
    assert len(X_test) == 10
    
    # Check that there are no remaining NaNs
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
    
    # Test normalization (mean should be close to 0, std close to 1)
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-1)
