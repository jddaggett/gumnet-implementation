import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        x = x.to(torch.float32)

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
        return self._apply_dct(x, type='dct')

    def _idct3D(self, x):
        return self._apply_dct(x, type='idct')

    # @TODO computationally complex, may needto find an optimized alternative 
    def _apply_dct(self, x, type='dct'):
        batch_size, channels, depth, height, width = x.shape

        # DCT basis matrices
        dct_mat_depth = self._create_dct_matrix(depth, type)
        dct_mat_height = self._create_dct_matrix(height, type)
        dct_mat_width = self._create_dct_matrix(width, type)

        dct_mat_depth = dct_mat_depth.to(x.device)
        dct_mat_height = dct_mat_height.to(x.device)
        dct_mat_width = dct_mat_width.to(x.device)

        x = x.permute(0, 1, 4, 3, 2)  # reshape to (batch, channel, width, height, depth)
        x = torch.matmul(dct_mat_width, x)
        x = x.permute(0, 1, 4, 3, 2)
        x = torch.matmul(dct_mat_height, x)
        x = x.permute(0, 1, 4, 3, 2)
        x = torch.matmul(dct_mat_depth, x)
        x = x.permute(0, 1, 4, 3, 2)

        return x

    def _create_dct_matrix(self, N, type='dct'):
        if type not in ['dct', 'idct']:
            raise ValueError("Type must be 'dct' or 'idct'")
        mat = torch.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if type == 'dct':
                    if k == 0:
                        mat[k, n] = 1 / np.sqrt(N)
                    else:
                        mat[k, n] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
                elif type == 'idct':
                    if n == 0:
                        mat[k, n] = 1 / np.sqrt(N)
                    else:
                        mat[k, n] = np.sqrt(2 / N) * np.cos(np.pi * (2 * k + 1) * n / (2 * N))
        return mat

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
