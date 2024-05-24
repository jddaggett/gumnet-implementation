import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RigidTransformation3DImputation(nn.Module):
    def __init__(self, output_size, padding_method="fill"):
        super(RigidTransformation3DImputation, self).__init__()
        self.output_size = output_size
        self.padding_method = padding_method

    def compute_output_shape(self, input_shapes):
        length, height, width = self.output_size
        num_channels = input_shapes[0][1]
        return [(None, num_channels, length, height, width)] * 3

    def _rescale_theta(self, tensors):
        theta, X = tensors
        shape = X.shape
        shift = 0.5
        scale_factor_translation = torch.tensor([shape[0] / (shape[0] + 2.0),
                                                 shape[1] / (shape[1] + 2.0),
                                                 shape[2] / (shape[2] + 2.0)]).to(theta.device)
        corrected_theta = theta.clone()
        corrected_theta[3:6] = (theta[3:6] - shift) * scale_factor_translation + shift
        return corrected_theta

    def forward(self, X, Y, m1, m2, theta):
        # Apply masks directly
        M1_t = self._mask_batch_affine_warp3d(m1, theta)
        M2_t = self._mask_batch_affine_warp3d(m2, theta)

        if self.padding_method == "fill":
            paddings = (1, 1, 1, 1, 1, 1)
            rescale_theta = torch.stack([self._rescale_theta((theta[i], X[i])) for i in range(theta.size(0))])
            X = F.pad(X, paddings, "constant", 0)
            X_t = self._batch_affine_warp3d(X, rescale_theta)
            X_t = X_t[:, :, 1:-1, 1:-1, 1:-1]  # Crop back to original size

        elif self.padding_method == "replicate":
            rescale_theta = torch.stack([self._rescale_theta((theta[i], X[i])) for i in range(theta.size(0))])
            X_t = self._batch_affine_warp3d(X, rescale_theta)
        else:
            raise NotImplementedError

        # Apply Fourier transform and masks
        FT_X_t = self._ft3d(X_t)
        FT_Y = self._ft3d(Y)
        FT_X_t_masked = torch.mul(FT_X_t, M1_t.to(dtype=torch.complex64))
        FT_Y_masked = torch.mul(FT_Y, M2_t.to(dtype=torch.complex64))

        # Inverse Fourier transform
        IFT_result = self._ift3d(FT_X_t_masked + FT_Y_masked).real.float()

        return IFT_result

    def _ft3d(self, x):
        # Permute the dimensions to bring channels to the last dimension for FFT
        x_perm = x.permute(0, 2, 3, 4, 1)  # [N, D, H, W, C]
        output = torch.fft.fftn(x_perm, dim=[1, 2, 3])
        output = torch.fft.fftshift(output, dim=[1, 2, 3])
        output = output.permute(0, 4, 1, 2, 3)
        return output

    def _ift3d(self, x):
        x_perm = x.permute(0, 2, 3, 4, 1)  # [N, D, H, W, C]
        x_perm = torch.fft.ifftshift(x_perm, dim=[1, 2, 3])
        output = torch.fft.ifftn(x_perm, dim=[1, 2, 3])
        output = output.permute(0, 4, 1, 2, 3).real
        return output

    def _fftshift(self, x, dim=None):
        if dim is None:
            dim = list(range(x.ndim))
        shift = [dim_size // 2 for dim_size in x.shape]
        return torch.roll(x, shift, dim)

    def _ifftshift(self, x, dim=None):
        if dim is None:
            dim = list(range(x.ndim))
        shift = [-(dim_size // 2) for dim_size in x.shape]
        return torch.roll(x, shift, dim)

    def _rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        loc_r = params[3:6] * 2 - 1

        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = torch.matmul(torch.matmul(a3, a2), a1).t()

        c = torch.matmul(-rm, loc_r.unsqueeze(1))

        rm = rm.flatten()
        theta = torch.cat([rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2]])

        return theta

    def _mask_rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        loc_r = params[3:6] * 0

        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = torch.matmul(torch.matmul(a3, a2), a1).t()

        c = torch.matmul(-rm, loc_r.unsqueeze(1))

        rm = rm.flatten()
        theta = torch.cat([rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2]])

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
            raise ValueError("Invalid dimension for rotation axis.")
        return rm

    def _interpolate3d(self, imgs, x, y, z):
        n_batch, n_channel, xlen, ylen, zlen = imgs.shape

        x = x.float()
        y = y.float()
        z = z.float()

        x = (x + 1) * (xlen - 1) * 0.5
        y = (y + 1) * (ylen - 1) * 0.5
        z = (z + 1) * (zlen - 1) * 0.5

        x0 = x.floor().long()
        x1 = x0 + 1
        y0 = y.floor().long()
        y1 = y0 + 1
        z0 = z.floor().long()
        z1 = z0 + 1

        x0 = x0.clamp(0, xlen - 1)
        x1 = x1.clamp(0, xlen - 1)
        y0 = y0.clamp(0, ylen - 1)
        y1 = y1.clamp(0, ylen - 1)
        z0 = z0.clamp(0, zlen - 1)
        z1 = z1.clamp(0, zlen - 1)

        base = torch.arange(n_batch, dtype=torch.long).view(n_batch, 1) * xlen * ylen * zlen

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

        imgs_flat = imgs.permute(0, 2, 3, 4, 1).reshape(-1, n_channel).float()

        I000 = imgs_flat[index000.view(-1)].view(n_batch, x.shape[1], -1)
        I001 = imgs_flat[index001.view(-1)].view(n_batch, x.shape[1], -1)
        I010 = imgs_flat[index010.view(-1)].view(n_batch, x.shape[1], -1)
        I011 = imgs_flat[index011.view(-1)].view(n_batch, x.shape[1], -1)
        I100 = imgs_flat[index100.view(-1)].view(n_batch, x.shape[1], -1)
        I101 = imgs_flat[index101.view(-1)].view(n_batch, x.shape[1], -1)
        I110 = imgs_flat[index110.view(-1)].view(n_batch, x.shape[1], -1)
        I111 = imgs_flat[index111.view(-1)].view(n_batch, x.shape[1], -1)

        dx = (x - x0.float()).unsqueeze(-1)
        dy = (y - y0.float()).unsqueeze(-1)
        dz = (z - z0.float()).unsqueeze(-1)

        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w010 = (1 - dx) * dy * (1 - dz)
        w011 = (1 - dx) * dy * dz
        w100 = dx * (1 - dy) * (1 - dz)
        w101 = dx * (1 - dy) * dz
        w110 = dx * dy * (1 - dz)
        w111 = dx * dy * dz

        output = (w000 * I000 + w001 * I001 + w010 * I010 + w011 * I011 + 
                  w100 * I100 + w101 * I101 + w110 * I110 + w111 * I111)

        return output.view(n_batch, n_channel, xlen, ylen, zlen)

    def _batch_warp3d(self, imgs, mappings):
        n_batch, n_channel, xlen, ylen, zlen = imgs.shape
        coords = mappings.view(n_batch, 3, -1)
        x_coords = coords[:, 0, :]
        y_coords = coords[:, 1, :]
        z_coords = coords[:, 2, :]
        output = self._interpolate3d(imgs, x_coords, y_coords, z_coords)
        return output

    def _repeat(self, base_indices, n_repeats):
        base_indices = base_indices.view(-1, 1).matmul(torch.ones(1, n_repeats, dtype=torch.long))
        return base_indices.view(-1)

    def _mgrid(self, *args, **kwargs):
        low = kwargs.pop("low", -1)
        high = kwargs.pop("high", 1)
        coords = [torch.linspace(low, high, steps=arg) for arg in args]
        grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=0)
        return grid

    def _batch_mgrid(self, n_batch, *args, **kwargs):
        grid = self._mgrid(*args, **kwargs)
        grid = grid.unsqueeze(0)
        grids = grid.repeat(n_batch, *([1] * (len(args) + 1)))
        return grids

    def _batch_affine_warp3d(self, imgs, theta):
        n_batch, n_channel, xlen, ylen, zlen = imgs.shape
        c = torch.stack([self._rotation_matrix_zyz(theta[i]) for i in range(len(theta))])
        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]
        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen).view(n_batch, 3, -1)
        T_g = torch.matmul(matrix, grids) + t
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)
        output = self._batch_warp3d(imgs, T_g)
        return output

    def _mask_batch_affine_warp3d(self, masks, theta):
        n_batch, n_channel, xlen, ylen, zlen = masks.shape
        c = torch.stack([self._mask_rotation_matrix_zyz(theta[i]) for i in range(len(theta))])
        theta = c.view(-1, 3, 4)
        matrix = theta[:, :, :3]
        t = theta[:, :, 3:]
        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen).view(n_batch, 3, -1)
        T_g = torch.matmul(matrix, grids) + t
        T_g = T_g.view(n_batch, 3, xlen, ylen, zlen)
        output = self._batch_warp3d(masks, T_g)
        return output



###################### TESTING ############################


def test_identity_transformation(model, X):
    theta = torch.zeros(1, 6)  # No rotation or translation
    Y = X.clone()
    m1 = torch.ones_like(X)
    m2 = torch.ones_like(X)
    output = model(X, Y, m1, m2, theta)
    print("Input and output difference:", torch.abs(output - X).max().item())
    assert torch.allclose(output, X, atol=1e-5), "Identity transformation test failed."

def test_known_rotation(model, X, angle, axis):
    # Create a known rotation matrix for the given angle and axis
    theta = torch.zeros(1, 6)
    theta[0, axis] = angle / (2 * np.pi) + 0.5  # Normalize the angle for the rotation
    Y = torch.zeros_like(X)
    m1 = torch.ones_like(X)
    m2 = torch.ones_like(X)
    output = model(X, Y, m1, m2, theta)
    # Apply the same rotation using scipy and compare
    from scipy.ndimage import rotate
    expected_output = rotate(X.numpy(), angle, axes=(axis+1, (axis+2)%3+1), reshape=False)
    assert torch.allclose(output, torch.tensor(expected_output, dtype=torch.float32), atol=1e-5), f"Rotation test failed for angle {angle} and axis {axis}."

def test_known_translation(model, X, translation):
    theta = torch.zeros(1, 6)
    theta[0, 3:6] = torch.tensor(translation) / 2 + 0.5  # Normalize the translation
    Y = torch.zeros_like(X)
    m1 = torch.ones_like(X)
    m2 = torch.ones_like(X)
    output = model(X, Y, m1, m2, theta)
    # Apply the same translation and compare
    expected_output = torch.roll(X, shifts=translation, dims=[2, 3, 4])
    assert torch.allclose(output, expected_output, atol=1e-5), f"Translation test failed for translation {translation}."

# Create the model
output_size = (32, 32, 32)
model = RigidTransformation3DImputation(output_size, padding_method="fill")

# Create dummy input
X = torch.rand(100, 1, 32, 32, 32)

def test_direct_affine_identity_transformation(model, X):
    # Create an identity theta matrix (3x4) for each batch
    n_batch = X.size(0)
    identity_theta = torch.zeros((n_batch, 6))
    identity_theta[:, :3] = 0.5  # Rotation angles corresponding to zero rotation
    identity_theta[:, 3:] = 0.5  # Translations corresponding to zero translation

    # Apply affine transformation
    X_t = model._batch_affine_warp3d(X, identity_theta)
    
    print("Direct Affine Identity transformation - max difference:", torch.abs(X - X_t).max().item())
    assert torch.allclose(X, X_t, atol=1e-5), "Direct affine identity transformation test failed."

# Run the direct affine identity transformation test
test_direct_affine_identity_transformation(model, X)

def test_rescaled_affine_identity_transformation(model, X):
    # Create an identity theta matrix (3x4) for each batch
    n_batch = X.size(0)
    identity_theta = torch.zeros((n_batch, 6))
    identity_theta[:, :3] = 0.5  # Rotation angles corresponding to zero rotation
    identity_theta[:, 3:] = 0.5  # Translations corresponding to zero translation

    # Rescale theta
    rescaled_theta = torch.stack([model._rescale_theta((identity_theta[i], X[i])) for i in range(n_batch)])
    
    # Apply affine transformation with rescaled theta
    X_t = model._batch_affine_warp3d(X, rescaled_theta)
    
    print("Rescaled Affine Identity transformation - max difference:", torch.abs(X - X_t).max().item())
    assert torch.allclose(X, X_t, atol=1e-5), "Rescaled affine identity transformation test failed."

# Run the rescaled affine identity transformation test
test_rescaled_affine_identity_transformation(model, X)

def test_mask_application(model, X, Y, m1, m2, theta):
    # Apply forward transformation
    output = model(X, Y, m1, m2, theta)
    
    # Intermediate Checks
    M1_t = model._mask_batch_affine_warp3d(m1, theta)
    M2_t = model._mask_batch_affine_warp3d(m2, theta)
    
    # Apply masks separately to see if they are causing the issue
    masked_X = X * M1_t
    masked_Y = Y * M2_t
    
    print("Masked X max difference:", torch.abs(X - masked_X).max().item())
    print("Masked Y max difference:", torch.abs(Y - masked_Y).max().item())
    
    # Ensure that masks are correctly applied without altering the data
    assert torch.allclose(X, masked_X, atol=1e-5), "Mask application test failed for X."
    assert torch.allclose(Y, masked_Y, atol=1e-5), "Mask application test failed for Y."
    
    print("Mask application - max difference:", torch.abs(output - X).max().item())
    assert torch.allclose(output, X, atol=1e-5), "Mask application test failed."

# Create dummy masks and theta for the test
m1 = torch.ones_like(X)
m2 = torch.ones_like(X)
theta = torch.zeros(1, 6)  # Identity transformation

# Run the mask application test
test_mask_application(model, X, X, m1, m2, theta)

# Run the tests
test_identity_transformation(model, X)
test_known_rotation(model, X, angle=45, axis=0)  # Rotate 45 degrees around the x-axis
test_known_rotation(model, X, angle=90, axis=1)  # Rotate 90 degrees around the y-axis
test_known_translation(model, X, translation=[5, 0, 0])  # Translate along x-axis
test_known_translation(model, X, translation=[0, 5, 0])  # Translate along y-axis
test_known_translation(model, X, translation=[0, 0, 5])  # Translate along z-axis

print("All tests passed!")
