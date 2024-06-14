import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size, padding_method="zeros"):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size
        self.padding_method = padding_method

    def forward(self, x, y, m1, m2, theta):
        # Apply padding and affine transformations
        if self.padding_method == "zeros":
            x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
            x_transformed = self.apply_affine_transform(x_padded, theta)
            x_transformed = x_transformed[:, :, 1:-1, 1:-1, 1:-1]
        elif self.padding_method == "replicate":
            x_transformed = self.apply_affine_transform(x, theta)
        else:
            raise NotImplementedError("Padding method not implemented")

        # Transform masks without translation
        m1_transformed = self.apply_affine_transform(m1, theta, only_rotation=True)
        m2_transformed = self.apply_affine_transform(m2, theta, only_rotation=True)

        # Impute missing fourier coefficients in x from observed coefficients from y
        output = fft.ifftn(
            torch.mul(fft.fftn(x_transformed, dim=(2, 3, 4)), m1_transformed) +
            torch.mul(fft.fftn(y, dim=(2, 3, 4)), m2_transformed), dim=(2, 3, 4)
        ).float()

        return output

    def rescale_theta(self, theta):
        """
        Scales theta to be symmetric about 0, assuming theta is given
        to be symmetric about 0.5 from sigmoid activation.

        Parameters:
        theta (torch tensor): 6D transformation parameters.

        Returns:
        theta_rescaled (torch tensor): Tensor of same size as input with proper scaling.
        """
        theta_scaled = theta.clone() # Avoid in-place operations
        theta_scaled[:, :3] = (theta[:, :3] - 0.5) * 2 * torch.pi # Scale to [-pi, pi]
        theta_scaled[:, 3:] = (theta[:, 3:] - 0.5) * 2 # Scale to [-1, 1]
        return theta_scaled

    def euler_to_rot_matrix(self, theta):
        """
        Generates a rotation matrix from ZYZ Euler angles.

        Parameters:
        theta (torch tensor): The Euler angles (and other transformation parameters).

        Returns:
        R (torch tensor): The rotation matrix.
        """
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
        R = R.transpose(1, 2) # Transpose to account for right matrix-multiplication
        return R 

    def apply_affine_transform(self, x, theta, only_rotation=False):
        """
        Applies an affine transformation to 3D volume x by 6D transformation 
        parameters theta.

        Parameters:
        x (torch tensor): The input tensor
        theta (torch tensor): 6D transformation parameters.
        [optional] only_rotation (Bool): True if translation should be ignored.

        Returns:
        R (torch tensor): The rotation matrix.
        """
        B, C, D, H, W = x.shape

        # Get rotation matrix R and translation vector T from rescaled theta
        theta_rescaled = self.rescale_theta(theta)
        R = self.euler_to_rot_matrix(theta_rescaled)
        T = theta_rescaled[:, 3:]

        # Zero out translation vector if we don't need it (for masks)
        if only_rotation:
            T = torch.zeros_like(T)

        # Create an affine transformation matrix from R and T
        affine_matrix = torch.eye(4, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        affine_matrix[:, :3, :3] = R
        affine_matrix[:, 0, 3] = T[:, 0] / W * 2  # Normalize by width
        affine_matrix[:, 1, 3] = T[:, 1] / H * 2  # Normalize by height
        affine_matrix[:, 2, 3] = T[:, 2] / D * 2  # Normalize by depth
        affine_matrix = affine_matrix[:, :3]
        
        # Generate and sample affine grid to obtain transformed tensor
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        x_t = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='zeros')

        return x_t
