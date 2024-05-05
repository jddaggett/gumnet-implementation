import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import dctn, idctn

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
        # Apply logarithmic transformation if homomorphic processing is enabled xx
        # hgg
        if self.homomorphic:
            x = torch.log(x + 1e-6)  # Adding a small constant to prevent log(0)

        # Perform the 3D DCT, cropping, and then the 3D inverse DCT, DCT
        x_dct = self._dct3D(x)
        x_crop = self._cropping3D(x_dct)
        x_idct = self._idct3D(x_crop)

        # Apply exponential transformation if homomorphic processing was done
        if self.homomorphic:
            x_idct = torch.exp(x_idct)

        return x_idct

    def _dct3D(self, x):
        # Applying DCT along each dimension requires reordering of dimensions for compatibility with torch.fft.dctn
        x = x.permute(0, 4, 1, 2, 3)  # Move the channel to the first dimension (after batch)
        x = dctn(x, dim=[2, 3, 4], norm='ortho')  # Apply DCT across spatial dimensions
        return x.permute(0, 2, 3, 4, 1)  # Reorder dimensions back to original

    def _idct3D(self, x):
        # Applying IDCT similar to DCT, with the same dimension reordering
        x = x.permute(0, 4, 3, 2, 1)  # Move the channel to the first dimension and reverse other dimensions
        x = idctn(x, dim=[2, 3, 4], norm='ortho')  # Apply IDCT across spatial dimensions
        return x.permute(0, 2, 3, 4, 1)  # Restore original dimension order

    def _cropping3D(self, x):
        # Crop the high-frequency components based on the truncation settings
        x_trunc = x[:, :self.truncation[0], :self.truncation[1], :self.truncation[2]]
        # Calculate padding needed to achieve the output size
        pad = [(0, self.output_size[i] - self.truncation[i]) for i in range(3)]
        pad = [(0, 0)] + pad  # Add zero padding for the batch and channel dimensions
        x_pad = F.pad(x_trunc, pad, "constant", 0)  # Apply padding
        return x_pad
