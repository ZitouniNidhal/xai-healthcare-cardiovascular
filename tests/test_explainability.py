import pytest
import numpy as np
import pandas as pd
import torch
from src.utils.config import Config
from src.models.xgboost_model import XGBoostModelWrapper
from src.models.cnn_ecg import ECGCNN
from src.models.gradcam_explainer import GradCAMExplainer
from src.explainability.shap_analysis import calculate_shap_values
from src.explainability.generate_reports import generate_patient_diagnostic_report

def test_shap_values():
    config = Config()
    wrapper = XGBoostModelWrapper(config)
    
    # Generate mock features
    features = ["f1", "f2", "f3"]
    X_train = pd.DataFrame(np.random.normal(0, 1, (10, 3)), columns=features)
    y_train = np.random.choice([0, 1], size=10)
    X_test = pd.DataFrame(np.random.normal(0, 1, (5, 3)), columns=features)
    
    wrapper.train(X_train, y_train)
    shap_values, explainer = calculate_shap_values(wrapper.model, X_train, X_test)
    
    # Tree explainer output values size
    assert shap_values.values.shape == (5, 3)

def test_gradcam_explainer():
    model = ECGCNN(in_channels=12, num_classes=2)
    explainer = GradCAMExplainer(model)
    
    # Formulate sample tensor
    input_tensor = torch.randn(1, 12, 500, requires_grad=True)
    heatmap = explainer.generate_heatmap(input_tensor, target_class=1)
    
    assert len(heatmap) == 500
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

def test_diagnostic_report_explanation_card():
    report = generate_patient_diagnostic_report(
        patient_id="P001",
        prediction_score=0.82,
        shap_contributions={"troponin": 0.6, "age": 0.2, "hdl": -0.4, "bmi": -0.1},
        lime_explanations=[("troponin", 0.5)],
        text_explanation="High risk requires clinical review.",
        tabular_score=0.9,
        ecg_score=0.7,
    )

    card = report["clinical_explanation_card"]
    assert card["risk_band"] == "HIGH"
    assert card["top_positive_drivers"] == ["troponin", "age"]
    assert card["top_protective_drivers"] == ["hdl", "bmi"]
    assert card["modality_agreement"] == {
        "score": 0.8,
        "label": "HIGH",
        "tabular_risk_score": 0.9,
        "ecg_risk_score": 0.7,
    }
