import os
import argparse
import numpy as np
import pandas as pd
import torch

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_predictions

from src.data.load_dataset import load_or_create_datasets
from src.data.preprocessing import preprocess_tabular_data, preprocess_ecg_data

from src.models.xgboost_model import XGBoostModelWrapper
from src.models.cnn_ecg import ECGCNN, train_ecg_cnn
from src.models.gradcam_explainer import GradCAMExplainer
from src.models.lime_explainer import LIMEExplainerWrapper

from src.explainability.shap_analysis import calculate_shap_values, get_feature_importance_summary
from src.explainability.visualize_explanations import (
    plot_shap_summary,
    plot_lime_explanation,
    plot_ecg_gradcam
)
from src.explainability.generate_reports import generate_patient_diagnostic_report


def run_pipeline(args):
    # 1. Configuration & Logger Initialization
    config = Config()
    log_file = os.path.join(config.get_path("outputs")["logs_dir"], "pipeline.log")
    logger = setup_logger(log_file=log_file)
    logger.info("Initializing XAI Healthcare Cardiovascular Diagnostics Pipeline")
    
    # 2. Loading Datasets
    logger.info("Loading patient and ECG datasets...")
    df_raw, ecg_signals = load_or_create_datasets(config)
    
    # 3. Data Preprocessing
    logger.info("Preprocessing datasets...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_tabular_data(df_raw, config)
    processed_ecg = preprocess_ecg_data(ecg_signals, config)
    
    # 4. Model Training & Prediction: XGBoost (Tabular)
    logger.info("Training XGBoost Classifier...")
    xgb_wrapper = XGBoostModelWrapper(config)
    xgb_wrapper.train(X_train, y_train)
    xgb_wrapper.save()
    
    # Test tabular predictions
    xgb_preds = xgb_wrapper.predict(X_test)
    xgb_probs = xgb_wrapper.predict_proba(X_test)[:, 1]
    
    tabular_metrics = evaluate_predictions(y_test, xgb_preds, xgb_probs)
    logger.info(f"XGBoost Classifier Tabular Evaluation Metrics: {tabular_metrics}")
    
    # 5. Model Training & Prediction: ResNet 1D CNN (ECG)
    logger.info("Training 1D CNN model on ECG signals...")
    # Reshape ECG dataset to fit model dimensions (batch, channel, len)
    # y_train and y_test indices match the splitting of the tabular dataset
    train_indices = X_train.index.values
    test_indices = X_test.index.values
    
    X_ecg_train = processed_ecg[train_indices]
    y_ecg_train = y_train.values
    X_ecg_test = processed_ecg[test_indices]
    y_ecg_test = y_test.values
    
    cnn_model = ECGCNN(in_channels=12, num_classes=2)
    train_ecg_cnn(cnn_model, X_ecg_train, y_ecg_train, config)
    
    # Test CNN predictions
    cnn_model.eval()
    with torch.no_grad():
        ecg_test_tensor = torch.tensor(X_ecg_test, dtype=torch.float32)
        cnn_logits = cnn_model(ecg_test_tensor)
        cnn_probs = torch.softmax(cnn_logits, dim=1).numpy()
        cnn_preds = np.argmax(cnn_probs, axis=1)
        
    ecg_metrics = evaluate_predictions(y_ecg_test, cnn_preds, cnn_probs[:, 1])
    logger.info(f"ECG 1D CNN Classifier Evaluation Metrics: {ecg_metrics}")
    
    # 6. Ensemble late fusion risk calculation
    logger.info("Evaluating hybrid ensemble predictions...")
    hybrid_probs = 0.5 * xgb_probs + 0.5 * cnn_probs[:, 1]
    hybrid_preds = (hybrid_probs > 0.5).astype(int)
    hybrid_metrics = evaluate_predictions(y_test, hybrid_preds, hybrid_probs)
    logger.info(f"Hybrid Ensemble Evaluation Metrics: {hybrid_metrics}")
    
    # Save overall performance statistics
    metrics_path = config.get_path("outputs")["metrics_csv_path"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df = pd.DataFrame([
        {"model": "XGBoost Tabular", **tabular_metrics},
        {"model": "CNN ECG", **ecg_metrics},
        {"model": "Hybrid Ensemble", **hybrid_metrics}
    ])
    # Exclude confusion matrix from CSV structure for simplicity
    metrics_df.drop(columns=["confusion_matrix"]).to_csv(metrics_path, index=False)
    logger.info(f"Performance statistics saved to {metrics_path}")
    
    # 7. Explainability Analyses & Visualizations
    logger.info("Generating XAI explainability metrics and visualizations...")
    
    # 7.1 SHAP
    logger.info("Calculating SHAP feature importance...")
    shap_values, explainer = calculate_shap_values(xgb_wrapper.model, X_train, X_test)
    shap_plot_path = config.get_path("outputs")["shap_plot_path"]
    plot_shap_summary(shap_values, X_test, save_path=shap_plot_path)
    
    # 7.2 LIME & Diagnostic Report (for a selected single test patient)
    sample_idx = 0
    sample_patient_id = df_raw.iloc[test_indices[sample_idx]]["patient_id"]
    logger.info(f"Generating local LIME explanation & diagnostic report for patient: {sample_patient_id}")
    
    lime_wrapper = LIMEExplainerWrapper(
        training_data=X_train,
        feature_names=X_train.columns.tolist()
    )
    
    # Prepare predict wrapper returning 2D probability matrices
    predict_fn = lambda x: xgb_wrapper.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    lime_exp = lime_wrapper.explain_instance(X_test.iloc[sample_idx], predict_fn)
    
    # Create text narrative
    prob_percentage = hybrid_probs[sample_idx] * 100
    top_factors_desc = ", ".join([f"{feat}: {val:.2f}" for feat, val in lime_exp[:3]])
    text_narrative = (
        f"The model predicts a high risk of CVD at {prob_percentage:.1f}% for patient {sample_patient_id}. "
        f"Key risk factors driving this prediction include: {top_factors_desc}."
    )
    
    # Save structured clinical report
    reports_dir = os.path.join(config.get_path("outputs")["outputs_dir"], "reports")
    generate_patient_diagnostic_report(
        patient_id=sample_patient_id,
        prediction_score=hybrid_probs[sample_idx],
        shap_contributions={f: float(v) for f, v in zip(X_test.columns, shap_values[sample_idx].values)},
        lime_explanations=lime_exp,
        text_explanation=text_narrative,
        save_dir=reports_dir
    )
    
    # 7.3 Grad-CAM for the sample patient's ECG signal
    logger.info(f"Generating Grad-CAM visualization for patient: {sample_patient_id}")
    gradcam_explainer = GradCAMExplainer(cnn_model)
    
    # Format sample tensor
    sample_ecg_signal = X_ecg_test[sample_idx:sample_idx+1] # Shape (1, leads, length)
    sample_ecg_tensor = torch.tensor(sample_ecg_signal, dtype=torch.float32, requires_grad=True)
    
    # Generate heatmap for target prediction category
    target_class = int(y_ecg_test[sample_idx])
    gradcam_heatmap = gradcam_explainer.generate_heatmap(sample_ecg_tensor, target_class=target_class)
    
    gradcam_plot_path = config.get_path("outputs")["gradcam_plot_path"]
    plot_ecg_gradcam(sample_ecg_signal[0], gradcam_heatmap, save_path=gradcam_plot_path, lead_idx=0)
    
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Healthcare Cardiovascular Diagnostics Pipeline")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "preprocess", "train", "explain"],
                        help="Select pipeline stage execution mode")
    args = parser.parse_args()
    run_pipeline(args)
