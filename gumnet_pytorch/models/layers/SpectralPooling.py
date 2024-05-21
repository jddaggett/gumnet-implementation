import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct, idct # use scipy.fftpack instead of torch.fft

class SpectralPooling(nn.Module):
    """
    This class implements spectral pooling and filtering using 3D DCT, designed to work with noisy 3D image data.
    """
    def __init__(self, output_size, truncation, homomorphic=False):
        super(SpectralPooling, self).__init__()
        self.output_size = output_size
        self.truncation = truncation
        self.homomorphic = homomorphic

    def forward(self, x):
        if self.homomorphic:
            x = torch.log(x + 1e-6)
        x_dct = self._dct3D(x)
        x_crop = self._cropping3D(x_dct)
        x_idct = self._idct3D(x_crop)
        if self.homomorphic:
            x_idct = torch.exp(x_idct)
        return x_idct

    def _dct3D(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.cpu().detach().numpy()
        x = dct(dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
        return torch.tensor(x).permute(0, 4, 1, 2, 3).contiguous()

    def _idct3D(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.cpu().detach().numpy()
        x = idct(idct(idct(x, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
        return torch.tensor(x).permute(0, 4, 1, 2, 3).contiguous()

    def _cropping3D(self, x):
        x_trunc = x[:, :, :self.truncation[0], :self.truncation[1], :self.truncation[2]]
        pad = [0, self.output_size[2] - self.truncation[2],
               0, self.output_size[1] - self.truncation[1],
               0, self.output_size[0] - self.truncation[0],
               0, 0]
        x_pad = F.pad(x_trunc, pad, "constant", 0)
        return x_pad
