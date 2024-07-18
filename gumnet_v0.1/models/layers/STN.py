# Simple spatial transformer network adapted from:
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from STN in RigidTransformation3DImputation.py
def euler_to_rot_matrix(theta):
    phi, theta, psi = theta[:, 0], theta[:, 1], theta[:, 2]
    ca, cb, cc = torch.cos(phi), torch.cos(theta), torch.cos(psi)
    sa, sb, sc = torch.sin(phi), torch.sin(theta), torch.sin(psi)

    R_z1 = torch.stack([
        cc, -sc, torch.zeros_like(cc),
        sc, cc, torch.zeros_like(cc),
        torch.zeros_like(cc), torch.zeros_like(cc), torch.ones_like(cc)
    ], dim=-1).reshape(-1, 3, 3)

    R_y = torch.stack([
        cb, torch.zeros_like(cb), sb,
        torch.zeros_like(cb), torch.ones_like(cb), torch.zeros_like(cb),
        -sb, torch.zeros_like(cb), cb
    ], dim=-1).reshape(-1, 3, 3)

    R_z2 = torch.stack([
        ca, -sa, torch.zeros_like(ca),
        sa, ca, torch.zeros_like(ca),
        torch.zeros_like(ca), torch.zeros_like(ca), torch.ones_like(ca)
    ], dim=-1).reshape(-1, 3, 3)

    R = R_z2 @ R_y @ R_z1
    R = R.transpose(1, 2)
    return R 

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

    def forward(self, sa, c):
        B, C, D, H, W = sa.shape

        # Build affine transformation matrix
        R = euler_to_rot_matrix(c)
        T = c[:, 3:]
        affine_matrix = torch.eye(4, device=sa.device).unsqueeze(0).repeat(B, 1, 1)
        affine_matrix[:, :3, :3] = R
        affine_matrix[:, 0, 3] = T[:, 0] / W * 2  # Normalize by width
        affine_matrix[:, 1, 3] = T[:, 1] / H * 2  # Normalize by height
        affine_matrix[:, 2, 3] = T[:, 2] / D * 2  # Normalize by depth
        affine_matrix = affine_matrix[:, :3]
        
        # Generate grid
        grid = F.affine_grid(affine_matrix, [B, C, D, H, W], align_corners=False)
        
        # Apply the grid to sa using trilinear interpolation
        sa_transformed = F.grid_sample(sa, grid, mode='bilinear', align_corners=True)

        return sa_transformed
