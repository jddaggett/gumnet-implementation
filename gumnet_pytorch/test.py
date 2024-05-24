import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
    
    def _ft3d(self, x):
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=(2, 3, 4)), dim=(2, 3, 4), norm="ortho"), dim=(2, 3, 4))
    
    def _ift3d(self, x):
        return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(x, dim=(2, 3, 4)), dim=(2, 3, 4), norm="ortho"), dim=(2, 3, 4))

    def forward(self, x):
        x_ft = self._ft3d(x)
        x_ift = self._ift3d(x_ft)
        return x_ift

# Example usage:
model = ExampleModel()
input_tensor = torch.randn(2, 1, 32, 32, 32)  # Example input
output_tensor = model(input_tensor)

# Extract the real part for comparison
output_tensor_real = output_tensor.real

# Check if the tensors are close
if torch.allclose(input_tensor, output_tensor_real, atol=1e-6):
    print("The tensors are the same within the tolerance.")
else:
    print("The tensors are different.")

# For a detailed comparison:
print("Max difference:", (input_tensor - output_tensor_real).abs().max().item())
