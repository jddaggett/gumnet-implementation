import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size, padding_method="fill"):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size
        self.padding_method = padding_method

    def forward(self, X, Y, m1, m2, theta):
        # Ensure all tensors are on the correct device (GPU)
        X, Y, m1, m2, theta = map(lambda t: t.cuda(), [X, Y, m1, m2, theta])

        # Apply affine transformation to masks m1 and m2
        M1_t = self._mask_batch_affine_warp3d(m1, theta)
        M2_t = self._mask_batch_affine_warp3d(m2, theta)

        # Apply padding if the padding method is "fill"
        if self.padding_method == "fill":
            X = F.pad(X, (1, 1, 1, 1, 1, 1), "constant", 0)
            X_t = self._batch_affine_warp3d(X, theta)
            X_t = X_t[:, :, 1:-1, 1:-1, 1:-1]
        elif self.padding_method == "replicate":
            X_t = self._batch_affine_warp3d(X, theta)
        else:
            raise NotImplementedError

        # Combine transformed X and Y in the frequency domain
        output = self._ift3d(self._ft3d(X_t) * M1_t + self._ft3d(Y) * M2_t).real
        return output

    def _ft3d(self, x):
        # Perform 3D Fourier transform and shift zero-frequency component to the center
        x = torch.fft.fftn(x, dim=(2, 3, 4))
        x = torch.fft.fftshift(x, dim=(2, 3, 4))
        return x

    def _ift3d(self, x):
        # Perform inverse 3D Fourier transform and shift zero-frequency component back
        x = torch.fft.ifftshift(x, dim=(2, 3, 4))
        x = torch.fft.ifftn(x, dim=(2, 3, 4))
        return x

    def _rotation_matrix_zyz(self, params):
        # Generate rotation matrix from ZYZ Euler angles and translation parameters
        batch_size = params.size(0)
        phi = params[:, 0] * 2 * np.pi - np.pi
        theta = params[:, 1] * 2 * np.pi - np.pi
        psi_t = params[:, 2] * 2 * np.pi - np.pi

        # Normalize translation parameters to be within [-1, 1]
        loc_r = params[:, 3:6] * 2 - 1

        # Compute rotation matrices for each axis
        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = torch.bmm(torch.bmm(a3, a2), a1)

        # Transpose and compute final translation
        rm = rm.permute(0, 2, 1)
        c = torch.bmm(-rm, loc_r.unsqueeze(-1)).squeeze(-1)

        # Concatenate rotation and translation to form transformation matrix
        theta = torch.cat([rm[:, 0, :], c[:, 0].unsqueeze(-1), 
                           rm[:, 1, :], c[:, 1].unsqueeze(-1),
                           rm[:, 2, :], c[:, 2].unsqueeze(-1)], dim=1)

        return theta

    def _mask_rotation_matrix_zyz(self, params):
        # Similar to _rotation_matrix_zyz but for masks (no translation)
        batch_size = params.size(0)
        phi = params[:, 0] * 2 * np.pi - np.pi
        theta = params[:, 1] * 2 * np.pi - np.pi
        psi_t = params[:, 2] * 2 * np.pi - np.pi

        loc_r = params[:, 3:6] * 0

        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = torch.bmm(torch.bmm(a3, a2), a1)

        rm = rm.permute(0, 2, 1)
        c = torch.bmm(-rm, loc_r.unsqueeze(-1)).squeeze(-1)

        theta = torch.cat([rm[:, 0, :], c[:, 0].unsqueeze(-1), 
                           rm[:, 1, :], c[:, 1].unsqueeze(-1),
                           rm[:, 2, :], c[:, 2].unsqueeze(-1)], dim=1)

        return theta

    def _rotation_matrix_axis(self, dim, theta):
        # Compute rotation matrix for a given axis
        batch_size = theta.size(0)
        if dim == 0:  # Rotation around x-axis
            rm = torch.stack([
                torch.stack([torch.ones(batch_size, device=theta.device), torch.zeros(batch_size, device=theta.device), torch.zeros(batch_size, device=theta.device)], dim=1),
                torch.stack([torch.zeros(batch_size, device=theta.device), torch.cos(theta), -torch.sin(theta)], dim=1),
                torch.stack([torch.zeros(batch_size, device=theta.device), torch.sin(theta), torch.cos(theta)], dim=1)
            ], dim=1)
        elif dim == 1:  # Rotation around y-axis
            rm = torch.stack([
                torch.stack([torch.cos(theta), torch.zeros(batch_size, device=theta.device), torch.sin(theta)], dim=1),
                torch.stack([torch.zeros(batch_size, device=theta.device), torch.ones(batch_size, device=theta.device), torch.zeros(batch_size, device=theta.device)], dim=1),
                torch.stack([-torch.sin(theta), torch.zeros(batch_size, device=theta.device), torch.cos(theta)], dim=1)
            ], dim=1)
        elif dim == 2:  # Rotation around z-axis
            rm = torch.stack([
                torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros(batch_size, device=theta.device)], dim=1),
                torch.stack([torch.sin(theta), torch.cos(theta), torch.zeros(batch_size, device=theta.device)], dim=1),
                torch.stack([torch.zeros(batch_size, device=theta.device), torch.zeros(batch_size, device=theta.device), torch.ones(batch_size, device=theta.device)], dim=1)
            ], dim=1)
        else:
            raise ValueError("Invalid axis for rotation matrix")

        return rm

    def _interpolate3d(self, imgs, x, y, z):
        n_batch, n_channel, xlen, ylen, zlen = imgs.size()

        # Convert coordinates to float
        x = x.float()
        y = y.float()
        z = z.float()

        # Scale coordinates from [-1, 1] to [0, xlen/ylen/zlen]
        x = (x + 1.) * (xlen - 1.) * 0.5
        y = (y + 1.) * (ylen - 1.) * 0.5
        z = (z + 1.) * (zlen - 1.) * 0.5

        # Calculate floor and ceil indices for interpolation
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        # Clamp indices to be within valid range
        x0 = torch.clamp(x0, 0, xlen - 1)
        x1 = torch.clamp(x1, 0, xlen - 1)
        y0 = torch.clamp(y0, 0, ylen - 1)
        y1 = torch.clamp(y1, 0, ylen - 1)
        z0 = torch.clamp(z0, 0, zlen - 1)
        z1 = torch.clamp(z1, 0, zlen - 1)

        # Calculate base index for batch
        base = torch.arange(n_batch, device=imgs.device) * (xlen * ylen * zlen)
        base = base.view(n_batch, 1).repeat(1, x0.size(1))

        # Calculate indices for interpolation
        base_x0 = base + x0 * (ylen * zlen)
        base_x1 = base + x1 * (ylen * zlen)
        base00 = base_x0 + y0 * zlen
        base01 = base_x0 + y1 * zlen
        base10 = base_x1 + y0 * zlen
        base11 = base_x1 + y1 * zlen
        index000 = base00 + z0
        index001 = base00 + z1
        index010 = base01 + z0
        index011 = base01 + z1
        index100 = base10 + z0
        index101 = base10 + z1
        index110 = base11 + z0
        index111 = base11 + z1

        # Reshape images to flat tensor and gather values for interpolation
        imgs_flat = imgs.view(n_batch * xlen * ylen * zlen, n_channel)
        imgs_flat = imgs_flat.float()

        I000 = imgs_flat[index000.view(-1), :]
        I001 = imgs_flat[index001.view(-1), :]
        I010 = imgs_flat[index010.view(-1), :]
        I011 = imgs_flat[index011.view(-1), :]
        I100 = imgs_flat[index100.view(-1), :]
        I101 = imgs_flat[index101.view(-1), :]
        I110 = imgs_flat[index110.view(-1), :]
        I111 = imgs_flat[index111.view(-1), :]

        # Compute weights for interpolation
        x0 = x0.view(n_batch, -1)
        x1 = x1.view(n_batch, -1)
        y0 = y0.view(n_batch, -1)
        y1 = y1.view(n_batch, -1)
        z0 = z0.view(n_batch, -1)
        z1 = z1.view(n_batch, -1)

        dx = (x - x0.float()).view(n_batch, 1, -1)
        dy = (y - y0.float()).view(n_batch, 1, -1)
        dz = (z - z0.float()).view(n_batch, 1, -1)

        w000 = (1. - dx) * (1. - dy) * (1. - dz)
        w001 = (1. - dx) * (1. - dy) * dz
        w010 = (1. - dx) * dy * (1. - dz)
        w011 = (1. - dx) * dy * dz
        w100 = dx * (1. - dy) * (1. - dz)
        w101 = dx * (1. - dy) * dz
        w110 = dx * dy * (1. - dz)
        w111 = dx * dy * dz

        I000 = I000.view(n_batch, n_channel, -1)
        I001 = I001.view(n_batch, n_channel, -1)
        I010 = I010.view(n_batch, n_channel, -1)
        I011 = I011.view(n_batch, n_channel, -1)
        I100 = I100.view(n_batch, n_channel, -1)
        I101 = I101.view(n_batch, n_channel, -1)
        I110 = I110.view(n_batch, n_channel, -1)
        I111 = I111.view(n_batch, n_channel, -1)

        # Compute interpolated values
        output = w000 * I000 + w001 * I001 + w010 * I010 + w011 * I011 + w100 * I100 + w101 * I101 + w110 * I110 + w111 * I111
        output = output.view(n_batch, n_channel, xlen, ylen, zlen)

        return output

    def _batch_warp3d(self, imgs, mappings):
        # Perform batch-wise 3D warping
        n_batch = imgs.size(0)
        coords = mappings.view(n_batch, 3, -1)
        x_coords = coords[:, 0, :]
        y_coords = coords[:, 1, :]
        z_coords = coords[:, 2, :]

        output = self._interpolate3d(imgs, x_coords, y_coords, z_coords)
        return output

    def _repeat(self, base_indices, n_repeats):
        return base_indices.repeat_interleave(n_repeats)

    def _mgrid(self, *args, **kwargs):
        # Create orthogonal grid similar to np.mgrid
        low = kwargs.pop("low", -1)
        high = kwargs.pop("high", 1)
        device = kwargs.get('device', 'cpu')
        coords = [torch.linspace(low, high, steps=arg, device=device) for arg in args]
        grid = torch.stack(torch.meshgrid(*coords, indexing='ij'))
        return grid

    def _batch_mgrid(self, n_batch, *args, **kwargs):
        # Create batch of orthogonal grids
        grid = self._mgrid(*args, **kwargs)
        grid = grid.unsqueeze(0)
        grids = grid.repeat(n_batch, 1, 1, 1, 1)
        return grids

    def _batch_affine_warp3d(self, imgs, theta):
        # Perform affine transformation on batch of 3D images
        n_batch, channels, xlen, ylen, zlen = imgs.size()

        c = self._rotation_matrix_zyz(theta)
        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen, device=imgs.device)
        grids = grids.view(n_batch, 3, -1)
        T_g = torch.bmm(matrix, grids) + t
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)

        output = self._batch_warp3d(imgs, T_g)

        return output

    def _mask_batch_affine_warp3d(self, masks, theta):
        # Perform affine transformation on batch of 3D masks
        n_batch, channels, xlen, ylen, zlen = masks.size()

        c = self._mask_rotation_matrix_zyz(theta)
        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen, device=masks.device)
        grids = grids.view(n_batch, 3, -1)
        T_g = torch.bmm(matrix, grids) + t
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)

        output = self._batch_warp3d(masks, T_g)

        return output