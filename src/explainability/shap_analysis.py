import shap
import pandas as pd
import numpy as np

def calculate_shap_values(model, X_train, X_test):
    """
    Fits and calculates SHAP value components using TreeExplainer.
    
    Args:
        model: Trained tree classifier (XGBoost/RandomForest).
        X_train (pd.DataFrame): Training feature dataset.
        X_test (pd.DataFrame): Test dataset.
        
    Returns:
        tuple: (shap_values, explainer)
    """
    # Initialize tree explainer
    explainer = shap.TreeExplainer(model, data=X_train)
    shap_values = explainer(X_test)
    
    return shap_values, explainer

def get_feature_importance_summary(shap_values, feature_names):
    """
    Computes average absolute SHAP values for global feature ranking.
    
    Args:
        shap_values: Explainer output values.
        feature_names (list): Feature names.
        
    Returns:
        pd.DataFrame: Global feature importance data frame.
    """
    # Extract absolute values
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": vals
    }).sort_values(by="mean_absolute_shap", ascending=False)
    
    return importance_df
