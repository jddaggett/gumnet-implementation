import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size, padding_method="fill"):
        super(RigidTransformation3DImputation, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_size = output_size
        self.padding_method = padding_method

    def forward(self, X, Y, m1, m2, theta):
        X, Y, m1, m2, theta = [tensor.to(self.device) for tensor in (X, Y, m1, m2, theta)]

        # Warp masks with rotation params 
        M1_t = self._mask_batch_affine_warp3d(m1, theta)
        M2_t = self._mask_batch_affine_warp3d(m2, theta)

        # Pad and 3D transform X
        X_t = self._apply_padding_and_transform(X, theta)

        # Apply Fourier transform and masks to get missing pieces from Y and non-missing pieces from X
        FT_X_t_masked = self._ft3d(X_t) * M1_t.to(dtype=torch.complex64)
        FT_Y_masked = self._ft3d(Y) * M2_t.to(dtype=torch.complex64)

        # Combine and inverse Fourier to obtain imputed result
        IFT_result = self._ift3d(FT_X_t_masked + FT_Y_masked).real.float()
        return IFT_result

    def _apply_padding_and_transform(self, X, theta):
        if self.padding_method == "fill":
            # Add constant zero padding
            X = F.pad(X, (1, 1, 1, 1, 1, 1), "constant", 0)
            # Apply 3D affine warp
            X_t = self._batch_affine_warp3d(X, theta)
            # Remove the padding
            X_t = X_t[:, :, 1:-1, 1:-1, 1:-1]
        
        # Handle other methods of padding
        elif self.padding_method == "replicate":
            X_t = self._batch_affine_warp3d(X, theta)
        else:
            raise NotImplementedError(f"Padding method {self.padding_method} not implemented")
        return X_t

    def _batch_affine_warp3d(self, X, theta):
        # Generate affine grid with transformation parameters theta
        grid = self._affine_grid(theta, X.size())

        # Sample input tensor X with grid
        X_t = F.grid_sample(X, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return X_t

    def _mask_batch_affine_warp3d(self, masks, theta):
        theta_rot = theta.clone()
        theta_rot[:, :3] = 0  # Zero out translation part

        # Generate affine grid 
        grid = self._affine_grid(theta_rot, masks.size())

        # Sample mask with grid
        masks_t = F.grid_sample(masks, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return masks_t

    def _affine_grid(self, theta, size):
        batch_size = theta.size(0)

        # Compute the transformation matrix
        theta_matrix = self._compute_theta_matrix(theta)

        # Generate affine grid with transformation matrix
        grid = F.affine_grid(theta_matrix[:, :3, :], [batch_size, *size[1:]], align_corners=True)
        return grid

    def _compute_theta_matrix(self, theta):
        batch_size = theta.size(0)

        # Compute the 3D ZYZ rotation matrix with rotation params
        rotation_matrices = torch.stack([self._rotation_matrix_zyz(theta[i, 3:]) for i in range(batch_size)])

        # Compute the translation vector from the translation params
        translation_vectors = theta[:, :3].unsqueeze(-1)

        # Build the transformation matrix (don't need homogenous fourth row for torch.nn.functional)
        theta_matrix = torch.cat((rotation_matrices, translation_vectors), dim=2)
        return theta_matrix

    def _rotation_matrix_zyz(self, params):
        phi, theta, psi = params * 2 * np.pi # Rescale theta to radians

        # Pre-compute the sines and cosines of angles
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)

        # 3D otation matrix about Z-axis
        r1 = torch.tensor([[cos_phi, -sin_phi, 0],
                           [sin_phi, cos_phi, 0],
                           [0, 0, 1]], device=params.device)

        # 3D rotation matrix about Y-axis
        r2 = torch.tensor([[cos_theta, 0, sin_theta],
                           [0, 1, 0],
                           [-sin_theta, 0, cos_theta]], device=params.device)

        # 3D otation matrix about Z-axis
        r3 = torch.tensor([[cos_psi, -sin_psi, 0],
                           [sin_psi, cos_psi, 0],
                           [0, 0, 1]], device=params.device)

        # ZYZ rotation matrix is the product
        rotation_matrix = r3 @ r2 @ r1
        return rotation_matrix

    # torch.fft 3D fourier transform of spatial dims
    def _ft3d(self, x):
        return torch.fft.fftn(x, dim=(-3, -2, -1))

    # torch.fft 3D inverse fourier transform of spatial dims
    def _ift3d(self, x):
        return torch.fft.ifftn(x, dim=(-3, -2, -1))