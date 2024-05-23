import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct, idct

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
        device = x.device  # Save the device of the input tensor

        # Move the tensor to the CPU for DCT and IDCT operations (slows network)
        # @TODO find a way to do DCT on GPU
        x = x.cpu()

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

        # Move the tensor back to the original device
        x_idct = x_idct.to(device)

        return x_idct

    def _dct3D(self, x):
        # Applying DCT along each dimension
        x = x.detach().numpy()
        x = dct(dct(dct(x, axis=2, norm='ortho'), axis=3, norm='ortho'), axis=4, norm='ortho')
        return torch.tensor(x)

    def _idct3D(self, x):
        # Applying IDCT along each dimension
        x = x.detach().numpy()
        x = idct(idct(idct(x, axis=2, norm='ortho'), axis=3, norm='ortho'), axis=4, norm='ortho')
        return torch.tensor(x)

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
