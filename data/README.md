# Data Directory Structure

This directory stores patient records and ECG signals. Note that raw patient medical datasets should never be committed to repository version control.

## Layout

- `raw/`: Stores the original data files.
  - `mimic_patients.csv`: Tabular clinical records containing age, vital statistics, troponin levels, and disease status.
  - `ecg_signals.npy`: Numpy array storing ECG voltage values with dimensions `(n_patients, n_leads, samples)`.
- `processed/`: Processed, imputed, and normalized train/test subsets ready for model input.

## Data Dictionary (mimic_patients.csv)

| Feature | Type | Description |
|---|---|---|
| `patient_id` | Text | Unique identifier of the patient (e.g. `P00001`) |
| `age` | Integer | Patient age in years |
| `sex` | Binary | Patient biological sex (0 = Female, 1 = Male) |
| `systolic_bp` | Integer | Systolic Blood Pressure (mmHg) |
| `diastolic_bp` | Integer | Diastolic Blood Pressure (mmHg) |
| `heart_rate` | Integer | Resting heart rate (beats per minute) |
| `cholesterol_ldl` | Float | Low-Density Lipoprotein cholesterol (mg/dL) |
| `cholesterol_hdl` | Float | High-Density Lipoprotein cholesterol (mg/dL) |
| `troponin` | Float | Troponin biomarker level (ng/mL) |
| `bmi` | Float | Body Mass Index ($kg/m^2$) |
| `smoking` | Binary | Active smoking history (0 = No, 1 = Yes) |
| `diabetes` | Binary | History of diabetes mellitus (0 = No, 1 = Yes) |
| `family_cvd_history` | Binary | Family history of cardiovascular diseases (0 = No, 1 = Yes) |
| `cvd_target` | Binary | Target classification label (0 = No CVD, 1 = Cardivascular Disease) |
