import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size

    def forward(self, x, y, m1, m2, theta):

        # Transform the input tensor
        x_transformed = self.apply_affine_transform(x, theta)

        # Transform the masks without translation
        m1_transformed = self.apply_affine_transform(m1, theta, only_rotation=True)
        m2_transformed = self.apply_affine_transform(m2, theta, only_rotation=True)

        # Apply fast-fourier transform on spatial dimensions
        X_fft = fft.fftn(x_transformed, dim=(2, 3, 4))
        Y_fft = fft.fftn(y, dim=(2, 3, 4))

        # Impute the missing coefficients from the transformation target
        X_fft_observed = X_fft * m1_transformed
        Y_fft_missing = Y_fft * m2_transformed

        X_fft_imputed = X_fft_observed + Y_fft_missing

        # Inverse fourier to get result
        x_imputed = fft.ifftn(X_fft_imputed, dim=(2, 3, 4)).real
        return x_imputed

    def rescale_theta(self, theta, x):
        # Rescales theta which has been scaled to [0,1] from sigmoid activation,
        # which is not symmetric about 0
        shape = torch.tensor(x.shape[2:], dtype=torch.float32, device=x.device)
        theta_scaled = theta.clone()

        # Rescale rotation angles to [-π, π]
        theta_scaled[:, :3] = (theta[:, :3] - 0.5) * 2 * np.pi

        # Rescale translation values to [-shape/2, shape/2]
        theta_scaled[:, 3:] = (theta[:, 3:] - 0.5) * shape

        return theta_scaled

    def euler_to_rot_matrix(self, theta):
        # Gets rotation matrix from ZYZ rotation parameters
    
        phi, theta, psi = theta[:, 0], theta[:, 1], theta[:, 2]
        ca, cb, cc = torch.cos(phi), torch.cos(theta), torch.cos(psi)
        sa, sb, sc = torch.sin(phi), torch.sin(theta), torch.sin(psi)

        # Z rotation matrix
        R_z1 = torch.stack([
            cc, -sc, torch.zeros_like(cc),
            sc, cc, torch.zeros_like(cc),
            torch.zeros_like(cc), torch.zeros_like(cc), torch.ones_like(cc)
        ], dim=-1).reshape(-1, 3, 3)

        # Y rotation matrix
        R_y = torch.stack([
            cb, torch.zeros_like(cb), sb,
            torch.zeros_like(cb), torch.ones_like(cb), torch.zeros_like(cb),
            -sb, torch.zeros_like(cb), cb
        ], dim=-1).reshape(-1, 3, 3)

        # Z rotation matrix
        R_z2 = torch.stack([
            ca, -sa, torch.zeros_like(ca),
            sa, ca, torch.zeros_like(ca),
            torch.zeros_like(ca), torch.zeros_like(ca), torch.ones_like(ca)
        ], dim=-1).reshape(-1, 3, 3)

        R = R_z2 @ R_y @ R_z1

        # Transpose result because of right matrix multiplication
        R = R.transpose(1, 2)
        return R

    def apply_affine_transform(self, x, theta, only_rotation=False):
        B, C, D, H, W = x.shape

        # Rescale theta and build affine warp matrix components
        theta_rescaled = self.rescale_theta(theta, x)
        R = self.euler_to_rot_matrix(theta_rescaled)
        T = theta_rescaled[:, 3:]

        # Zero out translation if we don't need it (for masks)
        if only_rotation:
            T = torch.zeros_like(T)

        # Build affine warp matrix
        affine_matrix = torch.eye(3, 4, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        affine_matrix[:, :3, :3] = R
        affine_matrix[:, :3, 3] = T

        # Create an affine grid and sample it with bilinear interpolation
        grid = nn.functional.affine_grid(affine_matrix, [B, C, D, H, W], align_corners=True)
        x_transformed = nn.functional.grid_sample(x, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        return x_transformed
