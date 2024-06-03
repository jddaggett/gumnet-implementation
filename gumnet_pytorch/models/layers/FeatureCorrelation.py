import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCorrelation(nn.Module):
    """
    This layer computes the correlation between feature maps from two images.
    """
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, f_A, f_B):
        b, c, l0, h0, w0 = f_A.size()
        f_A = f_A.view(b, c, -1) # multiply spatial dims
        f_B = f_B.view(b, c, -1)

        # Compute and reshape correlation matrix between feature maps
        corr = torch.matmul(f_A.transpose(1, 2), f_B)
        return corr.view(b, l0, h0, w0, -1)
