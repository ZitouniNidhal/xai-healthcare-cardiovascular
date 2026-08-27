import os
import json

def _build_explanation_card(prediction_score, shap_contributions, lime_explanations,
                            tabular_score=None, ecg_score=None):
    """Build a compact, clinician-readable summary of explanation agreement."""
    positive_drivers = sorted(
        ((feature, float(value)) for feature, value in shap_contributions.items() if value > 0),
        key=lambda item: item[1], reverse=True
    )[:3]
    negative_drivers = sorted(
        ((feature, float(value)) for feature, value in shap_contributions.items() if value < 0),
        key=lambda item: item[1]
    )[:3]

    card = {
        "risk_band": "HIGH" if prediction_score > 0.7 else "MEDIUM" if prediction_score > 0.35 else "LOW",
        "top_positive_drivers": [feature for feature, _ in positive_drivers],
        "top_protective_drivers": [feature for feature, _ in negative_drivers],
        "review_note": "Use this explanation to support clinical review; it is not a diagnosis.",
    }

    if tabular_score is not None and ecg_score is not None:
        agreement = round(1.0 - abs(float(tabular_score) - float(ecg_score)), 3)
        card["modality_agreement"] = {
            "score": round(max(0.0, min(1.0, agreement)), 3),
            "label": "HIGH" if agreement >= 0.8 else "MODERATE" if agreement >= 0.5 else "LOW",
            "tabular_risk_score": round(float(tabular_score), 3),
            "ecg_risk_score": round(float(ecg_score), 3),
        }

    return card


def generate_patient_diagnostic_report(patient_id, prediction_score, shap_contributions,
                                       lime_explanations, text_explanation, save_dir=None,
                                       tabular_score=None, ecg_score=None):
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
        },
        "clinical_explanation_card": _build_explanation_card(
            prediction_score,
            shap_contributions,
            lime_explanations,
            tabular_score=tabular_score,
            ecg_score=ecg_score,
        ),
    }
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f"patient_report_{patient_id}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Saved structured clinical report to {report_path}")
        
    return report
