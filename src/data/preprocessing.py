import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_tabular_data(df, config):
    """
    Clean, impute, and scale structured clinical features.
    
    Args:
        df (pd.DataFrame): Raw patient data.
        config (Config): Project configurations.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor_dict)
    """
    # Extract config parameters
    prep_params = config.get("preprocessing")
    test_size = prep_params.get("test_size", 0.2)
    random_state = prep_params.get("random_state", 42)
    imputation_strategy = prep_params.get("imputation_strategy", "median")
    
    # Define features and target
    target_col = "cvd_target"
    id_col = "patient_id"
    
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split dataset before fitting imputer/scaler to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Impute missing values
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Convert back to DataFrame to preserve feature names for SHAP/LIME
    X_train_proc = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_proc = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    # Save processed dataset to files
    processed_dir = config.get_path("data")["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save train/test datasets
    train_df = X_train_proc.copy()
    train_df[target_col] = y_train.values
    test_df = X_test_proc.copy()
    test_df[target_col] = y_test.values
    
    processed_tabular_path = config.get_path("data")["processed_tabular"]
    # Save a merged copy of all processed tabular details
    all_processed = pd.concat([train_df, test_df]).sort_index()
    all_processed.to_csv(processed_tabular_path, index=False)
    
    preprocessor = {
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": feature_cols
    }
    
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor

def preprocess_ecg_data(ecg_signals, config):
    """
    Preprocess raw ECG signal matrices.
    Applies standard normalization and filters out potential NaN readings.
    
    Args:
        ecg_signals (np.ndarray): Shape (n_samples, n_leads, signal_length)
        config (Config): Configurations.
        
    Returns:
        np.ndarray: Normalized and processed ECG signals.
    """
    # Simple Z-score normalization per channel per patient
    n_samples, n_leads, signal_len = ecg_signals.shape
    processed_ecg = np.zeros_like(ecg_signals)
    
    for i in range(n_samples):
        for lead in range(n_leads):
            lead_signal = ecg_signals[i, lead, :]
            mean = np.mean(lead_signal)
            std = np.std(lead_signal) + 1e-8
            processed_ecg[i, lead, :] = (lead_signal - mean) / std
            
    # Save processed signal
    processed_ecg_path = config.get_path("data")["processed_ecg"]
    os.makedirs(os.path.dirname(processed_ecg_path), exist_ok=True)
    np.save(processed_ecg_path, processed_ecg)
    
    return processed_ecg
