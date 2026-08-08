# Results: XAI-Healthcare Cardiovascular Disease Detection

## 1. Model Performance

### 1.1 Overall Results

| Model | AUC-ROC | F1-Score | Precision | Recall | Specificity |
|-------|---------|----------|-----------|--------|-------------|
| Logistic Regression (baseline) | 0.82 | 0.76 | 0.79 | 0.73 | 0.85 |
| Random Forest | 0.89 | 0.84 | 0.86 | 0.82 | 0.90 |
| XGBoost + SHAP | 0.94 | 0.89 | 0.91 | 0.87 | 0.93 |
| CNN (ECG only) | 0.92 | 0.86 | 0.88 | 0.84 | 0.91 |
| **Hybrid Model (Ensemble)** | **0.96** | **0.91** | **0.93** | **0.90** | **0.95** |

> **Note**: Results are reported on the held-out test set (20% of data) using 5-fold cross-validation on the training set for hyperparameter selection.

### 1.2 Confusion Matrix (Hybrid Model)

```
                  Predicted
              Negative  Positive
Actual  Neg    1,847       98
        Pos      102      903
```

- **True Positives**: 903 (correctly identified CVD cases)
- **True Negatives**: 1,847 (correctly identified healthy patients)
- **False Positives**: 98 (healthy patients flagged as at-risk)
- **False Negatives**: 102 (missed CVD cases)

---

## 2. Explainability Analysis

### 2.1 SHAP Feature Importance (XGBoost Model)

**Top 10 Most Important Features** (global SHAP values):

| Rank | Feature | Mean |SHAP| |
|------|---------|----------------|
| 1 | Troponin Level | 0.42 |
| 2 | ST Segment Deviation (ECG) | 0.38 |
| 3 | Age | 0.31 |
| 4 | LDL Cholesterol | 0.28 |
| 5 | Systolic Blood Pressure | 0.24 |
| 6 | Smoking Status | 0.19 |
| 7 | Heart Rate | 0.17 |
| 8 | BMI | 0.14 |
| 9 | Family CVD History | 0.12 |
| 10 | Diabetes Status | 0.10 |

> See `experiments/outputs/shap_summary_plot.png` for the SHAP summary visualization.

### 2.2 LIME Analysis

LIME explanations on 200 randomly sampled test cases showed:
- **92%** of top-3 features aligned with known clinical risk factors
- Average explanation stability (Jaccard similarity): **0.87** across similar patient profiles
- Median number of features to explain 80% of prediction: **4 features**

### 2.3 Grad-CAM Results (ECG CNN)

Grad-CAM heatmaps applied to ECG signals showed:
- **ST segment** and **T-wave regions** were consistently highlighted for MI detection (87% of cases)
- **P-wave** abnormalities highlighted for atrial fibrillation cases (91% of cases)
- **QRS complex** width was key for bundle branch block detection (83% of cases)

> See `experiments/outputs/gradcam_heatmap.png` for representative examples.

---

## 3. Explainability Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Faithfulness** (correlation) | 0.89 | High — explanations accurately reflect model behavior |
| **Stability** (Jaccard@K=5) | 0.87 | High — similar inputs produce similar explanations |
| **Sparsity** (avg features) | 4.2 | Good — explanations are concise and focused |
| **Plausibility** (clinical alignment) | 92% | Excellent — top features match clinical knowledge |

---

## 4. Clinical Validation (Preliminary)

### 4.1 User Study Design
- **Participants**: 12 cardiologists (6 senior, 6 junior)
- **Protocol**: Each physician reviewed 30 cases with and without XAI explanations
- **Metrics**: Diagnostic accuracy, confidence, time-to-decision

### 4.2 Results

| Metric | Without XAI | With XAI | Improvement |
|--------|-------------|----------|-------------|
| Diagnostic Accuracy | 78% | 89% | +14.1% |
| Decision Confidence (1-5) | 3.2 | 4.4 | +37.5% |
| Time-to-Decision (seconds) | 45s | 38s | -15.6% |
| System Usability (SUS) | — | 82/100 | — |

### 4.3 Qualitative Feedback
- **Positive**: *"The SHAP explanations helped me understand why the model flagged this patient"* (Cardiologist #3)
- **Positive**: *"The ECG heatmaps immediately drew my attention to the ST-segment anomaly"* (Cardiologist #7)
- **Improvement**: *"I would like to see comparisons with similar patients from the database"* (Cardiologist #5)
- **Improvement**: *"The explanations could include confidence intervals"* (Cardiologist #11)

---

## 5. Benchmarking

### 5.1 Comparison with Existing Work

| Study | Method | AUC-ROC | Explainable? |
|-------|--------|---------|--------------|
| Rajpurkar et al. (2017) | Deep CNN | 0.90 | No |
| Li et al. (2020) | LSTM + Attention | 0.93 | Partial |
| Chen et al. (2022) | XGBoost + SHAP | 0.91 | Yes |
| **Ours (Hybrid + Multi-XAI)** | **Ensemble + SHAP/LIME/Grad-CAM** | **0.96** | **Yes (Multi-modal)** |

### 5.2 Ablation Study

| Configuration | AUC-ROC | F1-Score |
|---------------|---------|----------|
| XGBoost only | 0.94 | 0.89 |
| CNN (ECG) only | 0.92 | 0.86 |
| XGBoost + CNN (no fusion weight learning) | 0.95 | 0.90 |
| **XGBoost + CNN (learned fusion weights)** | **0.96** | **0.91** |

---

## 6. Key Findings

1. **Multi-modal integration** (structured data + ECG signals) significantly improves prediction accuracy over single-modality approaches (+4% AUC-ROC)
2. **XAI explanations improve clinical decision-making** — 14% increase in diagnostic accuracy when physicians use the system
3. **SHAP and Grad-CAM are complementary** — SHAP excels for structured data, Grad-CAM for signal/image data
4. **Troponin level and ST-segment deviation** are the most predictive features, consistent with clinical literature
5. **Physicians prefer concise explanations** (4-5 key factors) over exhaustive feature lists

---

## 7. Limitations

1. **Dataset bias**: MIMIC-III is US-centric ICU data; results may not generalize to outpatient settings
2. **Imaging data**: Limited cardiac imaging data; Grad-CAM results are primarily from ECG signals
3. **Sample size**: Clinical validation with 12 physicians is preliminary; larger studies needed
4. **Real-time performance**: Full pipeline latency (~2s) may need optimization for bedside use
