import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size, padding_method="fill"):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size
        self.padding_method = padding_method

    def forward(self, X, Y, theta):
        M1_t = self._mask_batch_affine_warp3d(Y, theta)
        M2_t = self._mask_batch_affine_warp3d(Y, theta)

        if self.padding_method == "fill":
            X_t = self._batch_affine_warp3d(X, theta)
        elif self.padding_method == "replicate":
            X_t = self._batch_affine_warp3d(X, theta)
        else:
            raise NotImplementedError

        output = self._ift3d(
            self._ft3d(X_t) * M1_t + self._ft3d(Y) * M2_t
        ).real

        return [output, M1_t, M2_t]

    def _ft3d(self, x):
        return torch.fft.fftn(x, dim=(2, 3, 4), norm="ortho")

    def _ift3d(self, x):
        return torch.fft.ifftn(x, dim=(2, 3, 4), norm="ortho")

    def _fftshift(self, x, dim):
        shift = [s // 2 for s in x.shape[dim]]
        return torch.roll(x, shift, dim)

    def _ifftshift(self, x, dim):
        shift = [-s // 2 for s in x.shape[dim]]
        return torch.roll(x, shift, dim)

    def _rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        loc_r = params[3:6] * 2 - 1

        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = torch.mm(torch.mm(a3, a2), a1).t()

        c = torch.mm(-rm, loc_r.view(-1, 1)).view(-1)

        rm = rm.flatten()

        theta = torch.cat([rm[:3], c[0:1], rm[3:6], c[1:2], rm[6:9], c[2:]])

        return theta

    def _mask_rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        loc_r = params[3:6] * 0

        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = torch.mm(torch.mm(a3, a2), a1).t()

        c = torch.mm(-rm, loc_r.view(-1, 1)).view(-1)

        rm = rm.flatten()

        theta = torch.cat([rm[:3], c[0:1], rm[3:6], c[1:2], rm[6:9], c[2:]])

        return theta

    def _rotation_matrix_axis(self, dim, theta):
        if dim == 0:
            rm = torch.tensor([[1.0, 0.0, 0.0],
                               [0.0, torch.cos(theta), -torch.sin(theta)],
                               [0.0, torch.sin(theta), torch.cos(theta)]])
        elif dim == 1:
            rm = torch.tensor([[torch.cos(theta), 0.0, torch.sin(theta)],
                               [0.0, 1.0, 0.0],
                               [-torch.sin(theta), 0.0, torch.cos(theta)]])
        elif dim == 2:
            rm = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                               [torch.sin(theta), torch.cos(theta), 0.0],
                               [0.0, 0.0, 1.0]])
        else:
            raise ValueError("dim must be 0, 1, or 2")

        return rm

    def _batch_affine_warp3d(self, imgs, theta):
        n_batch, xlen, ylen, zlen = imgs.shape[:4]

        c = torch.stack([self._rotation_matrix_zyz(theta[i]) for i in range(n_batch)])

        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen).view(n_batch, 3, -1)

        T_g = torch.bmm(matrix, grids) + t
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)

        output = self._batch_warp3d(imgs, T_g)

        return output

    def _mask_batch_affine_warp3d(self, masks, theta):
        n_batch, xlen, ylen, zlen = masks.shape[:4]

        c = torch.stack([self._mask_rotation_matrix_zyz(theta[i]) for i in range(n_batch)])

        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen).view(n_batch, 3, -1)

        T_g = torch.bmm(matrix, grids) + t
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)

        output = self._batch_warp3d(masks, T_g)

        return output

    def _batch_mgrid(self, n_batch, *args):
        grid = self._mgrid(*args)
        grid = grid.unsqueeze(0)
        grids = grid.repeat(n_batch, 1, 1, 1, 1)

        return grids

    def _mgrid(self, *args, low=-1, high=1):
        coords = [torch.linspace(low, high, steps=arg) for arg in args]
        grid = torch.stack(torch.meshgrid(*coords, indexing='ij'))

        return grid

    def _batch_warp3d(self, imgs, mappings):
        n_batch, xlen, ylen, zlen = imgs.shape[:4]
        coords = mappings.view(n_batch, 3, -1)
        x_coords = coords[:, 0, :]
        y_coords = coords[:, 1, :]
        z_coords = coords[:, 2, :]

        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        z_coords_flat = z_coords.flatten()

        output = self._interpolate3d(imgs, x_coords_flat, y_coords_flat, z_coords_flat)

        return output

    def _interpolate3d(self, imgs, x, y, z):
        n_batch, xlen, ylen, zlen, n_channel = imgs.shape

        x = (x + 1.0) * (xlen - 1.0) * 0.5
        y = (y + 1.0) * (ylen - 1.0) * 0.5
        z = (z + 1.0) * (zlen - 1.0) * 0.5

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, xlen - 1)
        x1 = torch.clamp(x1, 0, xlen - 1)
        y0 = torch.clamp(y0, 0, ylen - 1)
        y1 = torch.clamp(y1, 0, ylen - 1)
        z0 = torch.clamp(z0, 0, zlen - 1)
        z1 = torch.clamp(z1, 0, zlen - 1)

        base = torch.arange(n_batch) * xlen * ylen * zlen
        base = base.repeat_interleave(xlen * ylen * zlen)
        base = base.view(n_batch, xlen, ylen, zlen).flatten()

        base_x0 = base + x0 * ylen * zlen
        base_x1 = base + x1 * ylen * zlen
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

        imgs_flat = imgs.view(-1, n_channel)

        I000 = imgs_flat[index000]
        I001 = imgs_flat[index001]
        I010 = imgs_flat[index010]
        I011 = imgs_flat[index011]
        I100 = imgs_flat[index100]
        I101 = imgs_flat[index101]
        I110 = imgs_flat[index110]
        I111 = imgs_flat[index111]

        dx = x - x0.float()
        dy = y - y0.float()
        dz = z - z0.float()
        w000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        w001 = (1.0 - dx) * (1.0 - dy) * dz
        w010 = (1.0 - dx) * dy * (1.0 - dz)
        w011 = (1.0 - dx) * dy * dz
        w100 = dx * (1.0 - dy) * (1.0 - dz)
        w101 = dx * (1.0 - dy) * dz
        w110 = dx * dy * (1.0 - dz)
        w111 = dx * dy * dz

        output = w000.unsqueeze(1) * I000 + w001.unsqueeze(1) * I001 + w010.unsqueeze(1) * I010 + w011.unsqueeze(1) * I011 + w100.unsqueeze(1) * I100 + w101.unsqueeze(1) * I101 + w110.unsqueeze(1) * I110 + w111.unsqueeze(1) * I111

        output = output.view(n_batch, xlen, ylen, zlen, n_channel)

        return output