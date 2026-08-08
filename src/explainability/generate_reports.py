import os
import json

def generate_patient_diagnostic_report(patient_id, prediction_score, shap_contributions, lime_explanations, text_explanation, save_dir=None):
    """
    Generates a structured medical diagnostic report in JSON format.
    
    Args:
        patient_id (str): Patient identification.
        prediction_score (float): Predicted risk score.
        shap_contributions (dict): Key-value pairs of feature importance.
        lime_explanations (list): LIME local weights list.
        text_explanation (str): Auto-generated natural language report.
        save_dir (str): Location folder to save report file.
    """
    risk_level = "LOW"
    if prediction_score > 0.7:
        risk_level = "HIGH"
    elif prediction_score > 0.35:
        risk_level = "MEDIUM"
        
    report = {
        "report_id": f"REP-{patient_id}",
        "patient_id": patient_id,
        "diagnostic_summary": {
            "cardiovascular_disease_risk_score": f"{prediction_score * 100:.1f}%",
            "risk_assessment_level": risk_level,
            "automated_narrative": text_explanation
        },
        "explainability_data": {
            "shap_contributions": shap_contributions,
            "lime_local_weights": lime_explanations
        }
    }
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f"patient_report_{patient_id}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Saved structured clinical report to {report_path}")
        
    return report
