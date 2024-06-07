import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

# @TODO UNFINISHED LAYER, WILL IMPROVE SOON

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size

    def forward(self, x, y, m1, m2, theta):
        x_transformed = self.apply_affine_transform(x, theta)

        X_fft = self.fourier_transform(x_transformed)
        Y_fft = self.fourier_transform(y)

        X_fft_observed = X_fft * m1
        Y_fft_missing = Y_fft * m2

        X_fft_imputed = X_fft_observed + Y_fft_missing

        x_imputed = self.inverse_fourier_transform(X_fft_imputed)
        return x_imputed

    def rescale_theta(self, theta):
        theta[:, :3] = theta[:, :3] * 2 * np.pi - np.pi
        theta[:, 3:] = theta[:, 3:] * 2 - 1
        return theta

    def euler_to_rot_matrix(self, theta):
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sa, sb, sg = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

        R_x = torch.stack([torch.ones_like(ca), torch.zeros_like(ca), torch.zeros_like(ca),
                           torch.zeros_like(ca), ca, -sa,
                           torch.zeros_like(ca), sa, ca], dim=-1).reshape(-1, 3, 3)
        
        R_y = torch.stack([cb, torch.zeros_like(cb), sb,
                           torch.zeros_like(cb), torch.ones_like(cb), torch.zeros_like(cb),
                           -sb, torch.zeros_like(cb), cb], dim=-1).reshape(-1, 3, 3)
        
        R_z = torch.stack([cg, -sg, torch.zeros_like(cg),
                           sg, cg, torch.zeros_like(cg),
                           torch.zeros_like(cg), torch.zeros_like(cg), torch.ones_like(cg)], dim=-1).reshape(-1, 3, 3)
        
        R = R_z @ R_y @ R_x
        return R

    def apply_affine_transform(self, x, theta):
        B, C, D, H, W = x.shape
        theta = self.rescale_theta(theta)
        R = self.euler_to_rot_matrix(theta)
        T = theta[:, 3:]

        if torch.all(theta == 0):
            return x

        affine_matrix = torch.zeros(B, 3, 4, device=x.device)
        affine_matrix[:, :3, :3] = R
        affine_matrix[:, :3, 3] = T

        grid = nn.functional.affine_grid(affine_matrix, x.size(), align_corners=False)
        x_transformed = nn.functional.grid_sample(x, grid, align_corners=False)
        return x_transformed

    def fourier_transform(self, x):
        return fft.fftn(x, dim=(2, 3, 4))

    def inverse_fourier_transform(self, x):
        return fft.ifftn(x, dim=(2, 3, 4)).real