import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureL2Norm(nn.Module):
    """
    Normalizing features using l2 norm
    Modified from https://github.com/jaehyunnn/cnngeometric_tensorflow

    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """
    
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, x):
        epsilon = 1e-6
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + epsilon)
        normalized_output = x / norm
        return normalized_output
