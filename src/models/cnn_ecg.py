import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ResidualBlock1D(nn.Module):
    """1D Residual convolutional block for temporal ECG signal analysis."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ECGCNN(nn.Module):
    """
    1D ResNet CNN architecture for ECG classification.
    Allows easy hook activation for Grad-CAM explanation visualizations.
    """
    def __init__(self, in_channels=12, num_classes=2):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet Blocks
        self.block1 = ResidualBlock1D(64, 64, stride=1)
        self.block2 = ResidualBlock1D(64, 128, stride=2)
        self.block3 = ResidualBlock1D(128, 256, stride=2)
        self.block4 = ResidualBlock1D(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
        # Gradients/Activations storage for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Save activations for Grad-CAM in the last layer block
        x = self.block4(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            self.activations = x
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activations


def train_ecg_cnn(model, X_train, y_train, config):
    """Training helper loop for ECGCNN PyTorch architecture."""
    cnn_params = config.get("ecg_cnn")
    epochs = cnn_params.get("epochs", 10)
    batch_size = cnn_params.get("batch_size", 32)
    lr = cnn_params.get("learning_rate", 0.001)
    
    # Prepare PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(X_train)
        print(f"ECG CNN Train Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    # Save the PyTorch weights
    model_path = config.get_path("models")["ecg_cnn_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved ECG CNN model to {model_path}")
