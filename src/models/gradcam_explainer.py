import numpy as np
import torch

class GradCAMExplainer:
    """Grad-CAM explainer class tailored for 1D time-series convolutional networks like ECGCNN."""
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generates Grad-CAM heatmap values indicating local importance of parts of the 1D signal.
        
        Args:
            input_tensor (torch.Tensor): ECG input signal, shape (1, channels, length)
            target_class (int): Predict target category of interest.
            
        Returns:
            np.ndarray: Normalized 1D heatmap score.
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target score for classification output
        target = output[0, target_class]
        target.backward()
        
        # Extract gradients & feature activations from PyTorch model hooks
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations(input_tensor)
        
        # Pool gradients across signal temporal dimensions
        # Shape: (batch, channels, length)
        pooled_gradients = torch.mean(gradients, dim=2)
        
        # Weight channels by their respective gradients
        for i in range(activations.size(1)):
            activations[:, i, :] *= pooled_gradients[0, i]
            
        # Sum channel maps
        heatmap = torch.sum(activations, dim=1).squeeze()
        
        # Relu on heatmap to keep positive contributions, convert to numpy
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap.detach().cpu().numpy()
        
        # Normalize between 0 and 1
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        # Upsample heatmap to match original signal shape
        original_length = input_tensor.size(2)
        upsampled_heatmap = np.interp(
            np.linspace(0, len(heatmap) - 1, original_length),
            np.arange(len(heatmap)),
            heatmap
        )
        
        return upsampled_heatmap
