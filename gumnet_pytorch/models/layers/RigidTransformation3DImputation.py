import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

# @TODO a lot of testing and debugging still to do. 3d fourier transforms and inverse transforms
# seem to be properly working on randomized test tensors but 3d interpolation may have issues
# need to check affine warp code and also make sure parameters are what I expect
class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size, padding_method="fill"):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size
        self.padding_method = padding_method

    def _rescale_theta(self, tensors, device):
        theta, X = tensors
        shape = torch.tensor(X.shape[2:5], dtype=torch.float32, device=device)
        shift = 0.5
        ones = torch.ones(3, device=device)
        scale_factor = torch.cat((ones, shape / (shape + 2.)))
        corrected_theta = (theta - shift) * scale_factor + shift
        return corrected_theta

    def forward(self, X, Y, m1, m2, theta):
        device = X.device
        M1_t = self._mask_batch_affine_warp3d(m1, theta, device)
        M2_t = self._mask_batch_affine_warp3d(m2, theta, device)

        if self.padding_method == "fill":
            paddings = (0, 0, 1, 1, 1, 1, 1, 1) 
            rescale_theta = torch.stack([self._rescale_theta((t, X), device) for t in theta])
            X = F.pad(X, paddings, "constant", 0)
            X_t = self._batch_affine_warp3d(X, rescale_theta, device)
            X_t = X_t[:, :, 1:-1, 1:-1, 1:-1]
        elif self.padding_method == "replicate":
            X_t = self._batch_affine_warp3d(X, theta, device)
        else:
            raise NotImplementedError

        output = torch.fft.ifftn(
            torch.mul(self._ft3d(X_t, device), M1_t.to(torch.complex64)) +
            torch.mul(self._ft3d(Y, device), M2_t.to(torch.complex64)),
            dim=(2, 3, 4)
        ).real

        return [output, M1_t, M2_t]

    def _ft3d(self, x, device):
        output = fft.fftshift(fft.fftn(fft.fftshift(x, dim=(2, 3, 4)), dim=(2, 3, 4)), dim=(2, 3, 4))
        return output

    def _ift3d(self, x, device):
        output = fft.ifftshift(fft.ifftn(fft.ifftshift(x, dim=(2, 3, 4)), dim=(2, 3, 4)), dim=(2, 3, 4))
        return output

    def _fftshift(self, x, axes=None):
        shift = [dim // 2 for dim in x.shape[axes]] if axes else [dim // 2 for dim in x.shape]
        return torch.roll(x, shift, axes)

    def _ifftshift(self, x, axes=None):
        shift = [-(dim // 2) for dim in x.shape[axes]] if axes else [-(dim // 2) for dim in x.shape]
        return torch.roll(x, shift, axes)

    def _rotation_matrix_axis(self, dim, theta, device):
        if dim == 0:
            rm = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, torch.cos(theta), -torch.sin(theta)],
                [0.0, torch.sin(theta), torch.cos(theta)]
            ], device=device)
        elif dim == 1:
            rm = torch.tensor([
                [torch.cos(theta), 0.0, torch.sin(theta)],
                [0.0, 1.0, 0.0],
                [-torch.sin(theta), 0.0, torch.cos(theta)]
            ], device=device)
        elif dim == 2:
            rm = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0.0],
                [torch.sin(theta), torch.cos(theta), 0.0],
                [0.0, 0.0, 1.0]
            ], device=device)
        else:
            raise ValueError("Invalid axis for rotation matrix")
        return rm

    def _rotation_matrix_zyz(self, params, device):
        phi, theta, psi_t = params[:3] * 2 * torch.pi - torch.pi
        loc_r = params[3:6] * 2 - 1

        a1 = self._rotation_matrix_axis(2, psi_t, device)
        a2 = self._rotation_matrix_axis(1, theta, device)
        a3 = self._rotation_matrix_axis(2, phi, device)
        rm = torch.mm(torch.mm(a3, a2), a1).T

        c = torch.mm(-rm, loc_r.view(-1, 1))
        rm = rm.flatten()

        return torch.cat([rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2]])

    def _mask_rotation_matrix_zyz(self, params, device):
        phi, theta, psi_t = params[:3] * 2 * torch.pi - torch.pi
        loc_r = params[3:6] * 0

        a1 = self._rotation_matrix_axis(2, psi_t, device)
        a2 = self._rotation_matrix_axis(1, theta, device)
        a3 = self._rotation_matrix_axis(2, phi, device)
        rm = torch.mm(torch.mm(a3, a2), a1).T

        c = torch.mm(-rm, loc_r.view(-1, 1))
        rm = rm.flatten()

        return torch.cat([rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2]])

    def _batch_affine_warp3d(self, imgs, theta, device):
        n_batch = imgs.shape[0]
        xlen, ylen, zlen = imgs.shape[2:5]

        c = torch.stack([self._rotation_matrix_zyz(t, device) for t in theta])
        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen, device=device)
        grids = grids.view(n_batch, 3, -1)

        T_g = torch.bmm(matrix, grids) + t.unsqueeze(-1)
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)
        output = self._batch_warp3d(imgs, T_g, device)

        return output

    def _mask_batch_affine_warp3d(self, masks, theta, device):
        n_batch = masks.shape[0]
        xlen, ylen, zlen = masks.shape[2:5]

        c = torch.stack([self._mask_rotation_matrix_zyz(t, device) for t in theta])
        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen, device=device)
        grids = grids.view(n_batch, 3, -1)

        T_g = torch.bmm(matrix, grids) + t.unsqueeze(-1)
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)
        output = self._batch_warp3d(masks, T_g, device)

        return output

    def _batch_warp3d(self, imgs, mappings, device):
        n_batch = imgs.shape[0]
        coords = mappings.view(n_batch, 3, -1)
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]

        output = self._interpolate3d(imgs, x_coords, y_coords, z_coords, device)
        return output

    def _interpolate3d(self, imgs, x, y, z, device):
        n_batch = imgs.shape[0]
        xlen, ylen, zlen = imgs.shape[2:5]
        n_channel = imgs.shape[1]

        x, y, z = x.float(), y.float(), z.float()
        xlen_f, ylen_f, zlen_f = float(xlen), float(ylen), float(zlen)
        zero = torch.zeros([], dtype=torch.int32, device=device)
        max_x, max_y, max_z = xlen - 1, ylen - 1, zlen - 1

        x = (x + 1) * (xlen_f - 1) * 0.5
        y = (y + 1) * (ylen_f - 1) * 0.5
        z = (z + 1) * (zlen_f - 1) * 0.5

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        z0 = torch.floor(z).int()
        z1 = z0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        z0 = torch.clamp(z0, zero, max_z)
        z1 = torch.clamp(z1, zero, max_z)

        base = self._repeat(torch.arange(n_batch, device=device) * xlen * ylen * zlen, xlen * ylen * zlen)
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

        imgs_flat = imgs.view(-1, n_channel).float()
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
        w000 = ((1. - dx) * (1. - dy) * (1. - dz)).unsqueeze(1)
        w001 = ((1. - dx) * (1. - dy) * dz).unsqueeze(1)
        w010 = ((1. - dx) * dy * (1. - dz)).unsqueeze(1)
        w011 = ((1. - dx) * dy * dz).unsqueeze(1)
        w100 = (dx * (1. - dy) * (1. - dz)).unsqueeze(1)
        w101 = (dx * (1. - dy) * dz).unsqueeze(1)
        w110 = (dx * dy * (1. - dz)).unsqueeze(1)
        w111 = (dx * dy * dz).unsqueeze(1)

        output = (w000 * I000 + w001 * I001 + w010 * I010 + w011 * I011 +
                  w100 * I100 + w101 * I101 + w110 * I110 + w111 * I111)

        output = output.view(n_batch, n_channel, xlen, ylen, zlen)
        return output

    def _repeat(self, base_indices, n_repeats):
        return base_indices.repeat_interleave(n_repeats)

    def _mgrid(self, *args, low=-1, high=1, device=None):
        coords = [torch.linspace(low, high, steps=arg, device=device) for arg in args]
        return torch.stack(torch.meshgrid(*coords, indexing='ij'))

    def _batch_mgrid(self, n_batch, *args, low=-1, high=1, device=None):
        grid = self._mgrid(*args, low=low, high=high, device=device)
        grid = grid.unsqueeze(0)
        return grid.repeat(n_batch, *([1] * (len(args) + 1)))
