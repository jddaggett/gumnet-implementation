import numpy as np
import torch
import math
from scipy.ndimage import rotate
from skimage.io import imsave
from skimage import img_as_ubyte


def get_initial_weights(output_size):
    b = np.random.random((6, )) - 0.5
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [torch.tensor(W), torch.tensor(b.flatten(), dtype=torch.float32)]

    return weights


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x - mx, y - my
    print("shape of xm", xm.shape)
    print("shape of ym", ym.shape)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2))
    r = r_num / r_den
    r = torch.max(torch.min(r, torch.tensor(1.0)), torch.tensor(-1.0))

    return 1 - r**2