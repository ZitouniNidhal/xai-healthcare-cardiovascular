import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

def plot_shap_summary(shap_values, X_test, save_path=None):
    """Generates and saves SHAP beeswarm/summary plots."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved SHAP summary plot to {save_path}")
    plt.close()

def plot_lime_explanation(lime_exp_list, save_path=None):
    """Plots LIME local contributions bar plot."""
    features = [x[0] for x in lime_exp_list]
    weights = [x[1] for x in lime_exp_list]
    colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=weights, y=features, palette=colors, hue=features, legend=False)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("LIME Local Feature Contribution")
    plt.xlabel("Contribution Weight")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_ecg_gradcam(ecg_signal, heatmap, save_path=None, lead_idx=0):
    """
    Plots ECG raw signal with overlaid Grad-CAM heatmap highlighting predictive segments.
    
    Args:
        ecg_signal (np.ndarray): Shape (leads, length)
        heatmap (np.ndarray): Shape (length,)
        save_path (str): Saving output filepath.
        lead_idx (int): Lead index to show.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot raw signal
    ax1.plot(ecg_signal[lead_idx, :], color="black", label=f"Lead {lead_idx+1}")
    ax1.set_title("ECG Waveform Signal")
    ax1.set_ylabel("Normalized Voltage")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot heatmap
    time_axis = np.arange(len(heatmap))
    ax2.fill_between(time_axis, heatmap, color="red", alpha=0.4, label="Grad-CAM activation")
    ax2.plot(time_axis, heatmap, color="darkred", linewidth=1.5)
    ax2.set_title("Grad-CAM Focus Heatmap")
    ax2.set_ylabel("Activation Intensity")
    ax2.set_xlabel("Time Samples")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved Grad-CAM heatmap visualization to {save_path}")
    plt.close()
