import numpy as np
import torch
import math
import matplotlib.pyplot as plt


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=(1, 2, 3, 4), keepdim=True)
    my = torch.mean(y, dim=(1, 2, 3, 4), keepdim=True)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym, dim=(1, 2, 3, 4))
    r_den = torch.sqrt(torch.sum(xm**2, dim=(1, 2, 3, 4)) * torch.sum(ym**2, dim=(1, 2, 3, 4)))
    r = r_num / (r_den + 1e-8)
    r = torch.clamp(r, -1.0, 1.0)
    return torch.mean(1 - r**2)


def alignment_eval(y_true, y_pred, image_size):
    """
    y_true is defined in Radian [-pi, pi] (ZYZ convention) for rotation, and voxels for translation
    y_pred is in [0, 1] from sigmoid activation, need to scale y_pred for comparison
    """

    ang_d = []
    loc_d = []

    for i in range(len(y_true)):
        a = angle_zyz_difference(ang1=y_true[i][:3],
                                 ang2=y_pred[i][:3] * 2 * np.pi - np.pi)
        b = np.linalg.norm(
            np.round(y_true[i][3:6]) -
            np.round((y_pred[i][3:6] * 2 - 1) * (image_size / 2)))
        ang_d.append(a)
        loc_d.append(b)

    print('Rotation error: ', np.mean(ang_d), '+/-', np.std(ang_d),
          'Translation error: ', np.mean(loc_d), '+/-', np.std(loc_d), '----------')


def angle_zyz_difference(ang1=np.zeros(3), ang2=np.zeros(3)):
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
        raise

    return rm


def visualize_2d_slice(y_true, y_pred):
    """
    Visualizes a 2D slice from the reference subtomogram data (y_true) and the corresponding aligned subtomogram data (y_pred).

    Parameters:
    - y_true: Reference subtomogram data tensor with shape [100, 1, 32, 32, 32]
    - y_pred: Corresponding aligned subtomogram data tensor obtained from GumNet results with shape [100, 1, 32, 32, 32]
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
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(tensor.astype(np.float32))

def get_mrc_files(y_true, y_pred):
    try:
        y_true_sample = y_true[0].squeeze().cpu().detach().numpy()
        y_pred_sample = y_pred[0].squeeze().cpu().detach().numpy()
        save_as_mrc(y_true_sample, 'y_true_sample.mrc')
        save_as_mrc(y_pred_sample, 'y_pred_sample.mrc')
        print('Generated 2 mrc files for visualization')
    except:
        print('Error saving mrc files for visualization. Is the mrcfile package installed?')