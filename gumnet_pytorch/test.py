import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.gumnet_v2 import FeatureCorrelation, FeatureL2Norm, SpectralPooling, RigidTransformation3DImputation

## THIS CODE TESTS VARIOUS ASPECTS OF THE NETWORK BY PRINTING TO THE TERMINAL 
## EVERYTHING TESTED HERE SEEMS TO BE WORKING AS INTENDED

def apply_padding_and_transform(X, theta, padding_method="fill"):
    if padding_method == "fill":
        X = F.pad(X, (1, 1, 1, 1, 1, 1), "constant", 0)
        X_t = batch_affine_warp3d(X, theta)
        X_t = X_t[:, :, 1:-1, 1:-1, 1:-1]
    elif padding_method == "replicate":
        X_t = batch_affine_warp3d(X, theta)
    else:
        raise NotImplementedError(f"Padding method {padding_method} not implemented")
    return X_t

def batch_affine_warp3d(X, theta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid = affine_grid(theta, X.size()).to(device)
    #print("Affine grid:\n", grid)
    X_t = F.grid_sample(X, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return X_t

def affine_grid(theta, size):
    batch_size = theta.size(0)
    theta_matrix = compute_theta_matrix(theta).to(theta.device)
    print("Theta matrix:\n", theta_matrix)
    grid = F.affine_grid(theta_matrix[:, :3, :], [batch_size, *size[1:]], align_corners=True)
    return grid

def compute_theta_matrix(theta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = theta.size(0)
    rotation_matrices = torch.stack([rotation_matrix_zyz(theta[i, 3:]) for i in range(batch_size)]).to(device)
    translation_vectors = theta[:, :3].unsqueeze(-1).to(device)
    # Combine rotation matrix and translation vector to form a 3x4 affine transformation matrix
    theta_matrix = torch.cat((rotation_matrices, translation_vectors), dim=2)
    return theta_matrix

def rotation_matrix_zyz(params):
    device = params.device
    phi, theta, psi = params * 2 * np.pi
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)

    # R_z(phi)
    r1 = torch.tensor([[cos_phi, -sin_phi, 0],
                       [sin_phi, cos_phi, 0],
                       [0, 0, 1]], device=device)
    
    # R_y(theta)
    r2 = torch.tensor([[cos_theta, 0, sin_theta],
                       [0, 1, 0],
                       [-sin_theta, 0, cos_theta]], device=device)
    
    # R_z(psi)
    r3 = torch.tensor([[cos_psi, -sin_psi, 0],
                       [sin_psi, cos_psi, 0],
                       [0, 0, 1]], device=device)

    rotation_matrix = r3 @ r2 @ r1
    return rotation_matrix

def generate_synthetic_data(batch_size, channels, depth, height, width, translation_axis=None, translation_value=0):
    X = torch.rand(batch_size, channels, depth, height, width, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    theta = torch.zeros(batch_size, 6, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Identity transformation

    if translation_axis is not None:
        theta[:, translation_axis] = translation_value

    return X, theta

def generate_synthetic_pool_data(batch_size, channels, depth, height, width):
    X = torch.rand(batch_size, channels, depth, height, width, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return X

def test_transformation():
    batch_size = 1
    channels = 1
    depth = 3
    height = 3
    width = 3
    
    # Test translations along x, y, and z axes
    for axis, name in zip([0, 1, 2], ['x', 'y', 'z']):
        print(f"\nTesting translation along {name}-axis")
        X, theta = generate_synthetic_data(batch_size, channels, depth, height, width, translation_axis=axis, translation_value=0.5)
        print("Input tensor:\n", X)
        
        X_t = apply_padding_and_transform(X, theta)
        print("Apply transformation via:\n", theta)
        
        print("Output shape:", X_t.shape)
        print("Output tensor:\n", X_t)

def test_pooling():
    batch_size = 1
    channels = 1
    depth = 8
    height = 8
    width = 8
    output_size = (4, 4, 4)
    truncation = (4, 4, 4)

    spectral_pooling = SpectralPooling(output_size, truncation, homomorphic=False)

    X = generate_synthetic_pool_data(batch_size, channels, depth, height, width)
    print("Input tensor:\n", X)

    X_pooled = spectral_pooling(X)
    print("Output shape:", X_pooled.shape)
    print("Output tensor:\n", X_pooled)

    # Ensure the DCT and iDCT process maintains the shape and truncates as expected
    assert X_pooled.shape == (batch_size, channels, *output_size), "Output shape is incorrect."



def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=(1, 2, 3, 4), keepdim=True)
    my = torch.mean(y, dim=(1, 2, 3, 4), keepdim=True)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym, dim=(1, 2, 3, 4))
    r_den = torch.sqrt(torch.sum(xm**2, dim=(1, 2, 3, 4)) * torch.sum(ym**2, dim=(1, 2, 3, 4)))
    r = r_num / (r_den + 1e-8)
    r = torch.clamp(r, -1.0, 1.0)
    return torch.mean(1 - r**2)

def test_loss_fn():

    # Test with perfect correlation
    y_true = torch.rand((2, 3, 4, 4, 4))
    y_pred = y_true.clone()  # Perfect correlation
    loss = correlation_coefficient_loss(y_true, y_pred)
    print(f"Loss with perfect correlation: {loss.item()}")

    # Test with random data
    y_true = torch.rand((2, 3, 4, 4, 4))
    y_pred = torch.rand((2, 3, 4, 4, 4))
    loss = correlation_coefficient_loss(y_true, y_pred)
    print(f"Loss with random data: {loss.item()}")

    # Test with zero tensors
    y_true = torch.zeros((2, 3, 4, 4, 4))
    y_pred = torch.zeros((2, 3, 4, 4, 4))
    loss = correlation_coefficient_loss(y_true, y_pred)
    print(f"Loss with zero tensors: {loss.item()}")

    # Test with constant tensors
    y_true = torch.ones((2, 3, 4, 4, 4))
    y_pred = torch.ones((2, 3, 4, 4, 4)) * 2
    loss = correlation_coefficient_loss(y_true, y_pred)
    print(f"Loss with constant tensors: {loss.item()}")

    # Test with extreme values
    y_true = torch.full((2, 3, 4, 4, 4), 1e9)
    y_pred = torch.full((2, 3, 4, 4, 4), -1e9)
    loss = correlation_coefficient_loss(y_true, y_pred)
    print(f"Loss with extreme values: {loss.item()}")

def main():
    #test_transformation()
    #test_pooling()
    #test_loss_fn()
    return

if __name__ == "__main__":
    main()