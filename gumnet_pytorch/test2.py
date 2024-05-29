import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralPooling(nn.Module):
    def __init__(self, input_size, output_size, homomorphic=False):
        super(SpectralPooling, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.homomorphic = homomorphic

    def _dct3D(self, x):
        # Apply 3D DCT using FFT
        N = x.shape[-3:]
        X = torch.fft.fftn(x, dim=(-3, -2, -1))
        for i in range(3):
            n = N[i]
            k = torch.arange(n, device=x.device).unsqueeze(0)
            X = X * torch.exp(-1j * np.pi * k / (2 * n))
        return X.real

    def _idct3D(self, x):
        # Apply 3D inverse DCT using FFT
        N = x.shape[-3:]
        X = x.clone().to(torch.cfloat)
        for i in range(3):
            n = N[i]
            k = torch.arange(n, device=x.device).unsqueeze(0)
            X = X * torch.exp(1j * np.pi * k / (2 * n))
        return torch.fft.ifftn(X, dim=(-3, -2, -1)).real

    def _cropping3D(self, x):
        # Calculate crop indices
        crop_start = [(i - o) // 2 for i, o in zip(self.input_size, self.output_size)]
        crop_end = [start + o for start, o in zip(crop_start, self.output_size)]
        
        # Crop the DCT coefficients
        return x[..., crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    def forward(self, x):
        x = x.to(torch.float32)
        
        if self.homomorphic:
            x = torch.log(x + 1e-6)  # Adding a small constant to prevent log(0)

        # Perform the 3D DCT, cropping, and then the 3D inverse DCT
        x_dct = self._dct3D(x)
        x_crop = self._cropping3D(x_dct)
        x_idct = self._idct3D(x_crop)

        if self.homomorphic:
            x_idct = torch.exp(x_idct)

        return x_idct

# Testing Gradient Flow with Updated SpectralPooling
import torch
import torch.nn as nn
import torch.optim as optim

# Define a small network with the SpectralPooling layer for testing
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.spectral_pool = SpectralPooling((32, 32, 32), (16, 16, 16))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 16 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.spectral_pool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model, optimizer, and loss function
model = TestNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Create dummy data
input_data = torch.randn(8, 1, 32, 32, 32).cuda()
target_data = torch.randn(8, 10).cuda()

# Forward pass
output = model(input_data)
loss = criterion(output, target_data)

# Backward pass
optimizer.zero_grad()
loss.backward()

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Gradients for {name}: {param.grad.norm()}")
    else:
        print(f"No gradients for {name}")
