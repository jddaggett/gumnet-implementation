import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dctn, idctn

# @TODO check if the 3d dct and idct functions are working with the CUDA tensors
# when the GPU is not at nearly 100% memory usage
class SpectralPooling(nn.Module):
    """
    This class implements spectral pooling and filtering using 3D DCT, designed to work with noisy 3D image data.
    """
    def __init__(self, output_size, truncation, homomorphic=False):
        super(SpectralPooling, self).__init__()
        self.output_size = output_size  # Expected output dimensions after pooling
        self.truncation = truncation  # Limits for truncating the high-frequency components
        self.homomorphic = homomorphic  # If True, apply logarithmic and exponential transformations

    def forward(self, x):
        # Apply logarithmic transformation if homomorphic processing is enabled.
        if self.homomorphic:
            x = torch.log(x + 1e-6)  # Adding a small constant to prevent log(0)

        # Perform the 3D DCT, cropping, and then the 3D inverse DCT
        x_dct = self._dct3D(x)
        x_crop = self._cropping3D(x_dct)
        x_idct = self._idct3D(x_crop)

        # Apply exponential transformation if homomorphic processing was done
        if self.homomorphic:
            x_idct = torch.exp(x_idct)

        return x_idct
    
    def _dct3D(self, x):
        x_np = x.cpu().detach().numpy()
        x_dct = dctn(x_np, norm='ortho')
        return torch.tensor(x_dct, device=x.device)

    def _idct3D(self, x):
        x_np = x.cpu().detach().numpy()
        x_idct = idctn(x_np, norm='ortho')
        return torch.tensor(x_idct, device=x.device)

    def _cropping3D(self, x):
        # Crop the high-frequency components based on the truncation settings
        x_trunc = x[:, :, :self.truncation[0], :self.truncation[1], :self.truncation[2]]
        # Calculate padding needed to achieve the output size
        pad = [0, self.output_size[2] - self.truncation[2],
               0, self.output_size[1] - self.truncation[1],
               0, self.output_size[0] - self.truncation[0],
               0, 0]  # Zero padding for the batch and channel dimensions
        x_pad = F.pad(x_trunc, pad, "constant", 0)  # Apply padding
        return x_pad
