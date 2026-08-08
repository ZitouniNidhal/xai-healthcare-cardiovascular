import os
import numpy as np
import pandas as pd

def generate_mock_tabular_data(n_samples=1000):
    """
    Generates a realistic mock tabular dataset representing clinical patients.
    Includes features like age, blood pressure, cholesterol, troponin levels, and CVD diagnosis.
    """
    np.random.seed(42)
    
    # Generate structured parameters
    age = np.random.normal(62, 12, n_samples).clip(30, 95).astype(int)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.48, 0.52])  # 0=female, 1=male
    systolic_bp = np.random.normal(132, 18, n_samples).clip(90, 200).astype(int)
    diastolic_bp = np.random.normal(82, 10, n_samples).clip(50, 120).astype(int)
    heart_rate = np.random.normal(76, 14, n_samples).clip(45, 140).astype(int)
    cholesterol_ldl = np.random.normal(125, 30, n_samples).clip(50, 240).astype(int)
    cholesterol_hdl = np.random.normal(48, 12, n_samples).clip(20, 90).astype(int)
    bmi = np.random.normal(28.2, 5.4, n_samples).clip(15.0, 48.0)
    smoking = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    diabetes = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])
    family_cvd_history = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    
    # Troponin level (highly predictive of MI) - mostly low with some elevated spikes
    troponin = np.random.exponential(0.04, n_samples)
    troponin[np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)] += np.random.uniform(0.1, 1.5, int(n_samples * 0.15))
    
    # Risk factor combination to determine target cardiovascular disease (CVD)
    risk_score = (
        (age - 50) * 0.02 +
        sex * 0.2 +
        (systolic_bp - 120) * 0.01 +
        (cholesterol_ldl - 100) * 0.005 +
        (35 - cholesterol_hdl) * 0.01 +
        smoking * 0.5 +
        diabetes * 0.6 +
        family_cvd_history * 0.4 +
        troponin * 2.5 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert risk_score to binary outcome (CVD target)
    cvd_probability = 1 / (1 + np.exp(-risk_score))
    cvd_target = (cvd_probability > 0.5).astype(int)

    # Introduce some artificial NaNs to simulate real raw clinic data
    missing_mask = np.random.rand(n_samples, 2) < 0.05
    cholesterol_ldl = cholesterol_ldl.astype(float)
    bmi = bmi.astype(float)
    cholesterol_ldl[missing_mask[:, 0]] = np.nan
    bmi[missing_mask[:, 1]] = np.nan
    
    df = pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(5)}" for i in range(n_samples)],
        "age": age,
        "sex": sex,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "cholesterol_ldl": cholesterol_ldl,
        "cholesterol_hdl": cholesterol_hdl,
        "troponin": troponin,
        "bmi": bmi,
        "smoking": smoking,
        "diabetes": diabetes,
        "family_cvd_history": family_cvd_history,
        "cvd_target": cvd_target
    })
    
    return df

def generate_mock_ecg_data(n_samples=1000, signal_len=5000, n_leads=12):
    """
    Generates simulated multi-lead ECG signal waveforms.
    A simple synthetic waveform composed of baseline sine waves + heart beats.
    """
    np.random.seed(42)
    t = np.linspace(0, 10, signal_len)
    
    # Initialize array
    ecg_signals = np.zeros((n_samples, n_leads, signal_len))
    
    for i in range(n_samples):
        # Base frequency varies slightly per patient (heart rate variability)
        freq = np.random.uniform(1.0, 1.8)
        
        for lead in range(n_leads):
            # Phase shifts and noise to differentiate leads
            lead_phase = np.random.uniform(0, np.pi/4)
            # Baseline noise
            noise = np.random.normal(0, 0.05, signal_len)
            
            # Simple ECG approximation: periodic QRS complex using sum of sines/gaussians
            # Let's create periodic beats
            beat_positions = np.arange(0, 10, 1.0/freq)
            signal = np.sin(2 * np.pi * freq * t + lead_phase) * 0.1 # general baseline
            
            # Add QRS peaks
            for beat in beat_positions:
                signal += 1.0 * np.exp(-((t - beat - 0.1) / 0.05)**2)  # R peak
                signal -= 0.25 * np.exp(-((t - beat - 0.05) / 0.03)**2) # Q peak
                signal -= 0.15 * np.exp(-((t - beat - 0.15) / 0.03)**2) # S peak
                signal += 0.2 * np.exp(-((t - beat - 0.3) / 0.1)**2)    # T wave
                
            ecg_signals[i, lead] = signal + noise
            
    return ecg_signals

def load_or_create_datasets(config):
    """
    Loads patient and ECG datasets. If files are not present in raw directory,
    creates realistic mock data to facilitate complete pipeline execution.
    """
    raw_dir = config.get_path("data")["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    
    tabular_path = config.get_path("data")["raw_tabular"]
    ecg_path = config.get_path("data")["raw_ecg"]
    
    # Tabular Data Loading / Generation
    if os.path.exists(tabular_path):
        print(f"Loading tabular data from {tabular_path}")
        df = pd.read_csv(tabular_path)
    else:
        print(f"Tabular data file not found at {tabular_path}. Generating mock data.")
        df = generate_mock_tabular_data(n_samples=1000)
        df.to_csv(tabular_path, index=False)
        print(f"Saved mock tabular data to {tabular_path}")
        
    # ECG Data Loading / Generation
    if os.path.exists(ecg_path):
        print(f"Loading ECG signals from {ecg_path}")
        ecg_signals = np.load(ecg_path)
    else:
        print(f"ECG signals file not found at {ecg_path}. Generating mock ECG dataset.")
        ecg_signals = generate_mock_ecg_data(n_samples=1000)
        np.save(ecg_path, ecg_signals)
        print(f"Saved mock ECG data to {ecg_path}")
        
    return df, ecg_signals
