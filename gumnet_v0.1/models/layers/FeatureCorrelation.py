import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCorrelation(nn.Module):
    """
    This layer computes the correlation between feature maps from two images.
    """
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, x, y):
        b, c, d, h, w = x.size()
        x = x.view(b, c, d * h * w).transpose(1, 2)
        y = y.view(b, c, d * h * w)
        correlation = torch.bmm(x, y)
        return correlation.view(b, d, h, w, d * h * w)
