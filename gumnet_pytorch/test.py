import torch
import torch.nn.functional as F
import numpy as np

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
    print("Affine grid:\n", grid)
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

def generate_synthetic_data(batch_size, channels, depth, height, width):
    X = torch.rand(batch_size, channels, depth, height, width, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    Y = torch.rand(batch_size, channels, depth, height, width, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    m1 = torch.randint(0, 2, (batch_size, depth, height, width), dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    m2 = torch.randint(0, 2, (batch_size, depth, height, width), dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    theta = torch.zeros(batch_size, 6, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Identity transformation
    return X, Y, m1, m2, theta

def main():
    batch_size = 1
    channels = 1
    depth = 3
    height = 3
    width = 3
    
    # Generate synthetic data
    X, Y, m1, m2, theta = generate_synthetic_data(batch_size, channels, depth, height, width)
    print("Input tensor:\n", X)
    
    X_t = apply_padding_and_transform(X, theta)
    print("Apply transformation via:\n", theta)
    
    # Check the output
    print("Output shape:", X_t.shape)
    print("Output tensor:\n", X_t)

    # Ensure the transformation is an identity (i.e., output should be very close to input)
    assert torch.allclose(X, X_t, atol=1e-5), "The transformation should be an identity transformation."

if __name__ == "__main__":
    main()
