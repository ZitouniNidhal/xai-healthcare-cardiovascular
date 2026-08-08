# API Documentation

## Overview

The XAI-Healthcare API provides endpoints for cardiovascular disease prediction with explainable AI. The API accepts patient data (structured features and/or ECG signals) and returns risk predictions along with multi-modal explanations.

**Base URL**: `http://localhost:5000/api/v1`

---

## Authentication

> **Note**: Authentication is not implemented in the prototype. For production deployments, implement OAuth 2.0 or API key authentication.

---

## Endpoints

### 1. Health Check

```
GET /health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "timestamp": "2026-01-15T10:30:00Z"
}
```

---

### 2. Predict CVD Risk

```
POST /predict
```

**Request Body**:
```json
{
  "patient_data": {
    "age": 65,
    "sex": "male",
    "systolic_bp": 150,
    "diastolic_bp": 95,
    "heart_rate": 88,
    "cholesterol_ldl": 190,
    "cholesterol_hdl": 38,
    "troponin": 0.15,
    "bmi": 29.5,
    "smoking": true,
    "diabetes": true,
    "family_cvd_history": true,
    "hba1c": 7.2,
    "creatinine": 1.3
  },
  "ecg_data": null,
  "explain": true,
  "explanation_type": ["shap", "lime"]
}
```

**Response** (200 OK):
```json
{
  "prediction": {
    "risk_score": 0.85,
    "risk_level": "HIGH",
    "confidence": 0.92
  },
  "explanations": {
    "shap": {
      "feature_contributions": [
        {"feature": "troponin", "contribution": 0.42, "direction": "positive"},
        {"feature": "cholesterol_ldl", "contribution": 0.28, "direction": "positive"},
        {"feature": "age", "contribution": 0.15, "direction": "positive"},
        {"feature": "cholesterol_hdl", "contribution": -0.10, "direction": "negative"}
      ],
      "base_value": 0.35
    },
    "lime": {
      "top_features": [
        {"feature": "troponin > 0.12", "weight": 0.38},
        {"feature": "cholesterol_ldl > 160", "weight": 0.25},
        {"feature": "age > 60", "weight": 0.18}
      ]
    },
    "text_explanation": "The model predicts a HIGH CVD risk (85%) primarily due to: (1) elevated troponin level of 0.15 ng/mL (contribution: +42%), (2) high LDL cholesterol at 190 mg/dL (contribution: +28%), and (3) patient age of 65 years (contribution: +15%)."
  },
  "timestamp": "2026-01-15T10:30:05Z"
}
```

---

### 3. Predict with ECG Data

```
POST /predict/ecg
Content-Type: multipart/form-data
```

**Request Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ecg_file` | file | Yes | ECG signal file (CSV, WFDB, or EDF format) |
| `patient_data` | JSON | No | Structured patient features |
| `explain` | boolean | No | Include Grad-CAM explanations (default: true) |

**Response** (200 OK):
```json
{
  "prediction": {
    "risk_score": 0.78,
    "risk_level": "HIGH",
    "ecg_findings": ["ST-segment elevation", "T-wave inversion"],
    "confidence": 0.88
  },
  "explanations": {
    "gradcam": {
      "heatmap_url": "/api/v1/outputs/gradcam_12345.png",
      "highlighted_regions": [
        {"region": "ST-segment", "lead": "V2", "importance": 0.92},
        {"region": "T-wave", "lead": "aVL", "importance": 0.78}
      ]
    },
    "text_explanation": "The ECG analysis detected ST-segment elevation in leads V1-V4, which is highly suggestive of an acute myocardial infarction. The T-wave inversion in aVL further supports this finding."
  }
}
```

---

### 4. Batch Prediction

```
POST /predict/batch
```

**Request Body**:
```json
{
  "patients": [
    {
      "patient_id": "P001",
      "patient_data": { "age": 55, "sex": "female", ... }
    },
    {
      "patient_id": "P002",
      "patient_data": { "age": 72, "sex": "male", ... }
    }
  ],
  "explain": false
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {"patient_id": "P001", "risk_score": 0.32, "risk_level": "LOW"},
    {"patient_id": "P002", "risk_score": 0.71, "risk_level": "HIGH"}
  ],
  "processing_time_ms": 245
}
```

---

### 5. Get Explanation Report

```
GET /report/{patient_id}
```

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `json` | Output format: `json`, `html`, `pdf` |
| `include_plots` | boolean | `true` | Include SHAP/Grad-CAM visualizations |

---

### 6. Similar Patients

```
POST /similar
```

**Request Body**:
```json
{
  "patient_data": { "age": 65, "sex": "male", ... },
  "top_k": 5,
  "similarity_metric": "cosine"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request — Invalid input data |
| 404 | Not Found — Resource not found |
| 422 | Unprocessable Entity — Valid JSON but invalid field values |
| 500 | Internal Server Error |

## Rate Limiting

- **Prototype**: No rate limiting
- **Production**: 100 requests/minute per API key

---

## Data Formats

### ECG File Formats Supported
- **CSV**: Columns = leads (I, II, III, aVR, aVL, aVF, V1-V6), rows = samples at 500 Hz
- **WFDB**: PhysioNet waveform database format
- **EDF**: European Data Format

### HL7/FHIR Integration
The API supports HL7 FHIR R4 resources for interoperability with Electronic Medical Records:
- **Observation**: For lab results and vital signs
- **DiagnosticReport**: For prediction results and explanations
- **Patient**: For demographic information
