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
        """
        Arguments:
        f_A, f_B -- feature maps from two different images or transformations
                   shapes (batch, channels, depth, height, width)
        """
        # Get the shapes of each feature map
        b, c, l0, h0, w0 = f_A.size()
        _, _, l, h, w = f_B.size()

        # Reshape feature maps to flatten spatial dimensions
        f_A = f_A.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, c)  # Shape: (b, l0*h0*w0, c)
        f_B = f_B.view(b, c, -1)  # Shape: (b, c, l*h*w)

        # Compute the correlation by matrix multiplication
        f_mul = torch.bmm(f_A, f_B)  # Shape: (b, l0*h0*w0, l*h*w)
        norm = torch.sqrt((f_A * f_A).sum(dim=-1, keepdim=True))  # Sum across channels
        correlation_tensor = (f_mul / norm).view(b, l, h, w, l0 * h0 * w0)  # Reshape to include spatial dimensions

        return correlation_tensor

    def extra_repr(self):
        # Optional: Provides extra information about the module, helpful for debugging
        return 'FeatureCorrelation layer to compute spatial correlations between feature maps'
