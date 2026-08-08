# Methodology: Explainable AI for Cardiovascular Disease Detection

## 1. Research Design

This project follows a **design science research** methodology, iteratively developing and evaluating an XAI-based diagnostic system through the following phases:

1. **Problem Identification** → Clinical need for explainable CVD predictions
2. **Solution Design** → Hybrid AI model with multi-modal explainability
3. **Development** → Implementation of data pipeline, models, and XAI modules
4. **Evaluation** → Quantitative metrics + qualitative clinical validation
5. **Communication** → Publications and open-source release

---

## 2. Data Sources

### 2.1 MIMIC-III (Medical Information Mart for Intensive Care)
- **Type**: Electronic Health Records (EHR)
- **Size**: ~60,000 ICU admissions
- **Features**: Demographics, vitals, lab results, medications, diagnoses (ICD-9), free-text notes
- **Access**: PhysioNet credentialed access (requires CITI training)

### 2.2 PhysioNet Datasets
- **PTB-XL**: 21,837 12-lead ECG records (10s each), annotated with cardiac conditions
- **MIT-BIH Arrhythmia Database**: 48 half-hour 2-channel ambulatory ECG recordings
- **CinC Challenge 2017**: Single-lead ECG recordings for atrial fibrillation detection

### 2.3 Feature Categories

| Category | Features | Source |
|----------|----------|--------|
| Demographics | Age, sex, BMI, ethnicity | MIMIC-III |
| Vital Signs | Heart rate, blood pressure (systolic/diastolic), SpO2, respiratory rate | MIMIC-III |
| Lab Results | Troponin, BNP, LDL/HDL cholesterol, HbA1c, creatinine | MIMIC-III |
| Medical History | Diabetes, hypertension, smoking, family CVD history | MIMIC-III |
| ECG Signals | 12-lead ECG waveforms (voltage vs. time) | PTB-XL / PhysioNet |
| Imaging | Cardiac MRI/CT scans (if available) | Hospital partnerships |

---

## 3. Data Preprocessing

### 3.1 Structured Data
1. **Missing Value Handling**: Multiple imputation (MICE) for features with <30% missing; feature exclusion otherwise
2. **Normalization**: StandardScaler for continuous variables, one-hot encoding for categorical
3. **Feature Engineering**:
   - Derived features: pulse pressure, MAP (Mean Arterial Pressure)
   - Risk scores: Framingham Risk Score, HEART Score
   - Temporal features: rate of change in vitals over time

### 3.2 ECG Signal Processing
1. **Filtering**: Bandpass filter (0.5–40 Hz) to remove baseline wander and high-frequency noise
2. **Segmentation**: R-peak detection (Pan-Tompkins algorithm), beat-level segmentation
3. **Feature Extraction**: Heart rate variability (HRV), QRS duration, ST segment deviation, QT interval
4. **Augmentation**: Time warping, amplitude scaling, Gaussian noise injection, random cropping

### 3.3 Medical Imaging (if applicable)
1. **Preprocessing**: DICOM to PNG/NIfTI conversion, resolution standardization (224×224)
2. **Normalization**: Intensity normalization, contrast enhancement (CLAHE)
3. **Augmentation**: Random rotation (±15°), horizontal flip, elastic deformation

---

## 4. Model Architecture

### 4.1 Structured Data Model — XGBoost

```
Input Features (n=30-50)
    │
    ▼
┌─────────────────┐
│   XGBoost        │  Parameters:
│   Classifier     │  - n_estimators: 500
│                  │  - max_depth: 6
│                  │  - learning_rate: 0.01
│                  │  - subsample: 0.8
└────────┬────────┘
         │
         ▼
    CVD Risk Score (0-1)
```

### 4.2 ECG Model — 1D ResNet CNN

```
12-Lead ECG (12 × 5000 samples)
    │
    ▼
┌─────────────────┐
│  Conv1D Block    │ → BatchNorm → ReLU → MaxPool
│  (64 filters)    │
└────────┬────────┘
         │
    ▼ (×4 Residual Blocks)
┌─────────────────┐
│  ResNet Blocks   │ → [64, 128, 256, 512] filters
│  + Skip Connect  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Global Avg Pool │
│  + Dense(256)    │
│  + Dropout(0.5)  │
│  + Dense(1, σ)   │
└─────────────────┘
```

### 4.3 Image Model — Transfer Learning (Grad-CAM compatible)

```
Cardiac MRI/CT (224 × 224 × 3)
    │
    ▼
┌─────────────────┐
│  Pre-trained     │  ResNet50 / EfficientNet-B4
│  Feature         │  (ImageNet weights)
│  Extractor       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuned      │  Dense(512) → ReLU → Dropout(0.5)
│  Classifier      │  Dense(1, σ)
└─────────────────┘
```

### 4.4 Ensemble (Late Fusion)

```
XGBoost Score ──┐
                │
ECG CNN Score ──┼──→ Weighted Average ──→ Final CVD Risk
                │    (learned weights)
Image Score ────┘
```

---

## 5. Explainability Techniques

### 5.1 SHAP (SHapley Additive exPlanations)
- **Purpose**: Global and local explanations for structured data (XGBoost)
- **Outputs**:
  - **Summary plots**: Feature importance rankings across the dataset
  - **Force plots**: Per-patient explanation showing how each feature pushes the prediction
  - **Dependence plots**: Feature interaction analysis
- **Implementation**: TreeExplainer (fast, exact for tree-based models)

### 5.2 LIME (Local Interpretable Model-agnostic Explanations)
- **Purpose**: Local, model-agnostic explanations for tabular predictions
- **Approach**: Perturb input features, fit a local linear model around the prediction
- **Outputs**: Top-K feature contributions for a specific patient
- **Use Case**: Cross-validation of SHAP explanations, model-agnostic backup

### 5.3 Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Purpose**: Visual explanations for CNN-based image/signal models
- **Approach**: Use gradients of the target class flowing into the final convolutional layer
- **Outputs**: Heatmaps overlaid on input images highlighting discriminative regions
- **Extension**: Grad-CAM++ for improved localization of smaller regions

### 5.4 Natural Language Explanations
- **Template-based generation**: Convert SHAP/LIME outputs into human-readable sentences
- **Example**:
  > *"The model predicts a HIGH CVD risk (85%) primarily due to: (1) an ST-segment anomaly detected in the ECG (contribution: +40%), (2) LDL cholesterol at 190 mg/dL (contribution: +30%), and (3) patient age of 67 years (contribution: +15%)."*

---

## 6. Evaluation Framework

### 6.1 Model Performance Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| AUC-ROC | Area under ROC curve | ≥ 0.90 |
| F1-Score | Harmonic mean of precision and recall | ≥ 0.85 |
| Sensitivity (Recall) | True positive rate | ≥ 0.85 |
| Specificity | True negative rate | ≥ 0.80 |
| PPV (Precision) | Positive predictive value | ≥ 0.85 |

### 6.2 Explainability Metrics
| Metric | Description |
|--------|-------------|
| **Faithfulness** | Correlation between feature importance and prediction change upon removal |
| **Stability** | Consistency of explanations for similar inputs |
| **Sparsity** | Number of features needed to explain a prediction |
| **Plausibility** | Clinical alignment of top features with medical knowledge |

### 6.3 Clinical Validation
- **User Study Design**: Within-subjects comparison (with vs. without XAI explanations)
- **Participants**: 10–15 cardiologists
- **Measures**: Diagnostic accuracy, decision confidence, time-to-decision, System Usability Scale (SUS)

---

## 7. Ethical Considerations

1. **Data Privacy**: All data is de-identified; compliance with HIPAA and local data protection laws
2. **Bias Mitigation**: Evaluate model fairness across demographic groups (age, sex, ethnicity)
3. **Transparency**: All model decisions come with explanations; no autonomous decision-making
4. **Clinical Oversight**: System is designed as a **decision support tool**, not a replacement for physicians
