import pytest
import numpy as np
import torch
from src.utils.config import Config
from src.models.xgboost_model import XGBoostModelWrapper
from src.models.cnn_ecg import ECGCNN

def test_xgboost_model_wrapper():
    config = Config()
    wrapper = XGBoostModelWrapper(config)
    
    # Mock data
    X = np.random.normal(0, 1, (20, 5))
    y = np.random.choice([0, 1], size=20)
    
    wrapper.train(X, y)
    
    preds = wrapper.predict(X)
    probs = wrapper.predict_proba(X)
    
    assert len(preds) == 20
    assert probs.shape == (20, 2)
    assert np.all((probs >= 0) & (probs <= 1))

def test_ecg_cnn_forward():
    model = ECGCNN(in_channels=12, num_classes=2)
    # Batch = 4, Leads = 12, Length = 1000
    dummy_input = torch.randn(4, 12, 1000)
    
    output = model(dummy_input)
    
    assert output.shape == (4, 2)
