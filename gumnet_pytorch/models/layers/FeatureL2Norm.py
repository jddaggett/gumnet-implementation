import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureL2Norm(nn.Module):
    """
    Normalizing features using l2 norm

    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """
    
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6
