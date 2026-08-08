import numpy as np

def add_gaussian_noise(signal, snr_db=20):
    """Add Gaussian noise to the ECG signal to simulate clinical recording environment."""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def time_warp(signal, warp_factor=0.9):
    """Slightly speed up or slow down the signal waveform (simulating heart rate shifts)."""
    n_samples = len(signal)
    indices = np.linspace(0, n_samples - 1, int(n_samples * warp_factor))
    warped = np.interp(np.linspace(0, n_samples - 1, n_samples), indices, signal[:len(indices)])
    return warped

def amplitude_scale(signal, scale_min=0.8, scale_max=1.2):
    """Scales ECG signal amplitude arbitrarily to reflect varying electrode contact quality."""
    factor = np.random.uniform(scale_min, scale_max)
    return signal * factor

def augment_ecg_lead(signal, random_state=None):
    """Wrapper function that applies a random subset of ECG augmentations."""
    if random_state is not None:
        np.random.seed(random_state)
        
    augmented = signal.copy()
    choice = np.random.choice(["noise", "warp", "scale", "none"], size=2, replace=False)
    
    for action in choice:
        if action == "noise":
            augmented = add_gaussian_noise(augmented, snr_db=np.random.uniform(15, 25))
        elif action == "warp":
            augmented = time_warp(augmented, warp_factor=np.random.uniform(0.85, 1.15))
        elif action == "scale":
            augmented = amplitude_scale(augmented, scale_min=0.8, scale_max=1.2)
            
    return augmented
