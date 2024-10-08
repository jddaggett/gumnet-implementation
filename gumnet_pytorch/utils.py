import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from scipy.ndimage import rotate, shift
from torch.distributions import Normal

def rotate_tensor(tensor, angle, axes, device):
    """
    Rotates a 5D tensor along the specified axes.
    """
    # Convert angle to radians
    angle = torch.tensor(angle, device=device).float() * torch.pi / 180.0

    # Create an identity affine transformation matrix
    affine_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(tensor.size(0), 1, 1)

    # Rotation around x-axis
    if axes == (1, 2):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        affine_matrix[:, 1, 1] = cos_angle
        affine_matrix[:, 1, 2] = -sin_angle
        affine_matrix[:, 2, 1] = sin_angle
        affine_matrix[:, 2, 2] = cos_angle
    # Rotation around y-axis
    elif axes == (2, 3):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        affine_matrix[:, 0, 0] = cos_angle
        affine_matrix[:, 0, 2] = sin_angle
        affine_matrix[:, 2, 0] = -sin_angle
        affine_matrix[:, 2, 2] = cos_angle
    # Rotation around z-axis
    elif axes == (1, 3):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        affine_matrix[:, 0, 0] = cos_angle
        affine_matrix[:, 0, 1] = -sin_angle
        affine_matrix[:, 1, 0] = sin_angle
        affine_matrix[:, 1, 1] = cos_angle

    # Apply affine transformation
    affine_matrix = affine_matrix[:, :3, :]
    grid = F.affine_grid(affine_matrix, tensor.size(), align_corners=False)
    tensor = F.grid_sample(tensor, grid, align_corners=False)
    return tensor

def shift_tensor(tensor, shift, device):
    """
    Shifts a 5D tensor along the specified axes.
    """
    affine_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(tensor.size(0), 1, 1)
    affine_matrix[:, :3, 3] = torch.tensor(shift, device=device).float().unsqueeze(0)

    grid = F.affine_grid(affine_matrix[:, :3, :], tensor.size(), align_corners=False)
    tensor = F.grid_sample(tensor, grid, align_corners=False)
    return tensor

def augment_tensors(x, y, device):
    """
    Applies random rotations and translations to the data to prevent overfitting.

    Parameters:
    x (torch.Tensor): Input data for x.
    y (torch.Tensor): Input data for y.

    Returns:
    tuple: Augmented x and y.
    """
    # Randomly rotate
    angle_x = torch.FloatTensor(1).uniform_(-10, 10).item()
    angle_y = torch.FloatTensor(1).uniform_(-10, 10).item()
    angle_z = torch.FloatTensor(1).uniform_(-10, 10).item()
    
    x = rotate_tensor(x, angle_x, axes=(1, 2), device=device)
    x = rotate_tensor(x, angle_y, axes=(2, 3), device=device)
    x = rotate_tensor(x, angle_z, axes=(1, 3), device=device)
    y = rotate_tensor(y, angle_x, axes=(1, 2), device=device)
    y = rotate_tensor(y, angle_y, axes=(2, 3), device=device)
    y = rotate_tensor(y, angle_z, axes=(1, 3), device=device)

    # Randomly translate
    translate_x = torch.FloatTensor(1).uniform_(-5, 5).item()
    translate_y = torch.FloatTensor(1).uniform_(-5, 5).item()
    translate_z = torch.FloatTensor(1).uniform_(-5, 5).item()
    x = shift_tensor(x, (translate_x, translate_y, translate_z), device)
    y = shift_tensor(y, (translate_x, translate_y, translate_z), device)

    # Add random noise
    noise_x = Normal(0, 0.01).sample(x.size()).to(device)
    noise_y = Normal(0, 0.01).sample(y.size()).to(device)
    x += noise_x
    y += noise_y

    return x, y

def generate_masks(x, tilt_angle=30):
    """
    Generate a missing Fourier mask for a given subtomogram tensor, x, with a specified tilt angle.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape (batches, channels, depth, height, width)
        tilt_angle (float): Maximum tilt angle in degrees (default is 30 degrees)
        
    Returns:
        torch.Tensor: Missing Fourier mask of the same shape as the input tensor
        torch.Tensor: Inverse of mask
    """
    # Get the shape of the tensor
    batches, channels, depth, height, width = x.shape

    # Calculate the missing wedge angles in radians
    tilt_radians = np.radians(tilt_angle)

    # Create a meshgrid for the Fourier space coordinates
    kz = np.fft.fftfreq(depth)
    ky = np.fft.fftfreq(height)
    kx = np.fft.fftfreq(width)

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

    # Calculate the angles for the Fourier space coordinates
    angles = np.abs(np.arcsin(KZ))

    # Create the missing wedge mask
    mask = np.ones((depth, height, width), dtype=np.float32)
    mask[angles > tilt_radians] = 0

    # Expand the mask to match the input tensor shape
    mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batches, channels, depth, height, width)
    
    return mask, 1 - mask

# Data augmentation to prevent overfitting
def augment_data(x, y):
    """
    Applies random rotations and translations to the data to prevent overfitting.

    Parameters:
    x (numpy array): Input data for x.
    y (numpy array): Input data for y.

    Returns:
    tuple: Augmented x and y.
    """
    # Randomly rotate
    angle_x = np.random.uniform(-10, 10)
    angle_y = np.random.uniform(-10, 10)
    angle_z = np.random.uniform(-10, 10)
    x = rotate(x, angle_x, axes=(1, 2), reshape=False)
    x = rotate(x, angle_y, axes=(2, 3), reshape=False)
    x = rotate(x, angle_z, axes=(1, 3), reshape=False)
    y = rotate(y, angle_x, axes=(1, 2), reshape=False)
    y = rotate(y, angle_y, axes=(2, 3), reshape=False)
    y = rotate(y, angle_z, axes=(1, 3), reshape=False)

    # Randomly translate
    translate_x = np.random.uniform(-5, 5)
    translate_y = np.random.uniform(-5, 5)
    translate_z = np.random.uniform(-5, 5)
    x = shift(x, (0, translate_x, translate_y, translate_z, 0), order=1)
    y = shift(y, (0, translate_x, translate_y, translate_z, 0), order=1)

    return x, y


# Added He weight initialization to prevent gradient vanishing
def initialize_weights(module, name='He'):
    """
    Initializes weights for convolutional and linear layers.

    Parameters:
    module (torch.nn.Module): The module to initialize.
    name (str): The initialization method ('He' or 'Xavier').

    """
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def correlation_coefficient_loss(y_true, y_pred):
    """
    Computes the correlation coefficient loss between y_true and y_pred.

    Parameters:
    y_true (torch.Tensor): Ground truth tensor.
    y_pred (torch.Tensor): Predicted tensor.

    Returns:
    torch.Tensor: The computed loss.
    """
    y_true_mean = torch.mean(y_true, dim=[2, 3, 4], keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=[2, 3, 4], keepdim=True)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    covariance = torch.sum(y_true_centered * y_pred_centered, dim=[2, 3, 4])
    y_true_std = torch.sqrt(torch.sum(y_true_centered ** 2, dim=[2, 3, 4]))
    y_pred_std = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=[2, 3, 4]))

    correlation = covariance / (y_true_std * y_pred_std + 1e-6)
    return 1 - correlation.mean()

# For debugging purposes only with [batches, 6] shaped input
def correlation_coefficient_loss_params(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=1, keepdim=True)
    my = torch.mean(y, dim=1, keepdim=True)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym, dim=1)
    r_den = torch.sqrt(torch.sum(xm**2, dim=1) * torch.sum(ym**2, dim=1))
    r = r_num / (r_den + 1e-8)
    r = torch.clamp(r, -1.0, 1.0)
    return torch.mean(1 - r**2)


def compute_cross_correlation(x, y):
    """
    Compute the cross-correlation between two subtomograms x and y.
    """
    # Mean-center the subtomograms
    x = x - x.mean(dim=[2, 3, 4], keepdim=True)
    y = y - y.mean(dim=[2, 3, 4], keepdim=True)
    
    # Compute cross-correlation
    cross_correlation = torch.sum(x * y, dim=[2, 3, 4])
    
    # Normalize by the norms of x and y
    norm_x = torch.sqrt(torch.sum(x ** 2, dim=[2, 3, 4]))
    norm_y = torch.sqrt(torch.sum(y ** 2, dim=[2, 3, 4]))
    
    cross_correlation /= (norm_x * norm_y + 1e-8)  # Add epsilon to avoid division by zero
    
    return cross_correlation.mean()


def alignment_eval(y_true, y_pred, image_size, scale=True):
    """
    Evaluates the alignment of y_pred against y_true.

    Parameters:
    y_true (numpy array): Ground truth transformation parameters.
    y_pred (numpy array): Predicted transformation parameters.
    image_size (int): The size of the image.
    scale (bool): True if the input must be rescaled from [0,1] to [-1,1]

    """
    ang_d = []
    loc_d = []

    for i in range(len(y_true)):
        if scale:
            a = angle_zyz_difference(ang1=y_true[i][:3],
                                     ang2=y_pred[i][:3] * 2 * np.pi - np.pi)
            b = np.linalg.norm(
                np.round(y_true[i][3:6]) -
                np.round((y_pred[i][3:6] * 2 - 1) * (image_size / 2)))
        else:
            a = angle_zyz_difference(ang1=y_true[i][:3],
                                     ang2=y_pred[i][:3])
            b = np.linalg.norm(
                np.round(y_true[i][3:6]) -
                np.round((y_pred[i][3:6])))
        ang_d.append(a)
        loc_d.append(b)

    print('Rotation error: ', np.mean(ang_d), '+/-', np.std(ang_d),
          'Translation error: ', np.mean(loc_d), '+/-', np.std(loc_d), '----------')


def angle_zyz_difference(ang1=np.zeros(3), ang2=np.zeros(3)):
    """
    Computes the angular difference between two sets of ZYZ Euler angles.

    Parameters:
    ang1 (numpy array): First set of angles.
    ang2 (numpy array): Second set of angles.

    Returns:
    float: The angular difference.
    """
    loc1_r = np.zeros(ang1.shape)
    loc2_r = np.zeros(ang2.shape)

    rm1 = rotation_matrix_zyz(ang1)
    rm2 = rotation_matrix_zyz(ang2)
    loc1_r_t = np.array([loc1_r, loc1_r, loc1_r])
    loc2_r_t = np.array([loc2_r, loc2_r, loc2_r])

    dif_m = (rm1.dot(np.eye(3) - loc1_r_t)).transpose() -\
            (rm2.dot(np.eye(3) - loc2_r_t)).transpose()
    dif_d = math.sqrt(np.square(dif_m).sum())

    return dif_d


def rotation_matrix_zyz(ang):
    """
    Generates a rotation matrix from ZYZ Euler angles.

    Parameters:
    ang (numpy array): The Euler angles.

    Returns:
    numpy array: The rotation matrix.
    """
    phi = ang[0]
    theta = ang[1]
    psi_t = ang[2]

    # first rotate about z axis for angle psi_t
    a1 = rotation_matrix_axis(2, psi_t)
    a2 = rotation_matrix_axis(1, theta)
    a3 = rotation_matrix_axis(2, phi)

    # for matrix left multiplication
    rm = a3.dot(a2).dot(a1)
    # note: transform because tformarray use right matrix multiplication
    rm = rm.transpose()

    return rm


def rotation_matrix_axis(dim, theta):
    """
    Generates a rotation matrix for a given axis and angle.

    Parameters:
    dim (int): The axis (0 for x, 1 for y, 2 for z).
    theta (float): The rotation angle.

    Returns:
    numpy array: The rotation matrix.
    """
    # following are left handed system (clockwise rotation)
    # x-axis
    if dim == 0:
        rm = np.array([[1.0, 0.0, 0.0],
                       [0.0, math.cos(theta), -math.sin(theta)],
                       [0.0, math.sin(theta), math.cos(theta)]])
    # y-axis
    elif dim == 1:
        rm = np.array([[math.cos(theta), 0.0, math.sin(theta)],
                       [0.0, 1.0, 0.0],
                       [-math.sin(theta), 0.0, math.cos(theta)]])
    # z-axis
    elif dim == 2:
        rm = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                       [math.sin(theta), math.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")

    return rm


def visualize_2d_slice(y_true, y_pred):
    """
    Visualizes a 2D slice from the reference subtomogram data (y_true) and the corresponding aligned subtomogram data (y_pred).

    Parameters:
    y_true (torch.Tensor): Reference subtomogram data tensor with shape [100, 1, 32, 32, 32]
    y_pred (torch.Tensor): Corresponding aligned subtomogram data tensor obtained from GumNet results with shape [100, 1, 32, 32, 32]
    """
    # Arbitrarily select the first sample in the batch
    y_true_sample = y_true[0].squeeze().cpu().detach().numpy()
    y_pred_sample = y_pred[0].squeeze().cpu().detach().numpy()

    # Arbitrarily select middle 2D slice along depth dimension
    slice_index = y_true_sample.shape[0] // 2
    y_true_slice = y_true_sample[slice_index, :, :]
    y_pred_slice = y_pred_sample[slice_index, :, :]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true_slice, cmap='gray')
    plt.title('Ground Truth Slice')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred_slice, cmap='gray')
    plt.title('Predicted Slice')
    plt.axis('off')

    plt.show()


# the below code loads subtomogram data as mrc files for 3D visualization with ChimeraX
import mrcfile
def save_as_mrc(tensor, filename):
    """
    Saves a tensor as an MRC file for 3D visualization.

    Parameters:
    tensor (numpy array): The tensor data.
    filename (str): The filename to save the MRC file.

    """
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(tensor.astype(np.float32))

def get_mrc_files(y_true, y_pred):
    """
    Generates MRC files for the ground truth and predicted subtomograms for 3D visualization.

    Parameters:
    y_true (torch.Tensor): Reference subtomogram data tensor.
    y_pred (torch.Tensor): Corresponding aligned subtomogram data tensor.

    """
    try:
        y_true_sample = y_true[0].squeeze().cpu().detach().numpy()
        y_pred_sample = y_pred[0].squeeze().cpu().detach().numpy()
        save_as_mrc(y_true_sample, 'y_true_sample.mrc')
        save_as_mrc(y_pred_sample, 'y_pred_sample.mrc')
        print('Generated 2 mrc files for visualization')
    except Exception as e:
        print('Error saving mrc files for visualization:', e)
