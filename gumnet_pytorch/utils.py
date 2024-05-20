import numpy as np
import torch
import math
from scipy.ndimage import rotate
from skimage.io import imsave
from skimage import img_as_ubyte


def get_initial_weights(output_size):
    b = np.random.random((6, )) - 0.5
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [torch.tensor(W), torch.tensor(b.flatten(), dtype=torch.float32)]

    return weights


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = torch.tensor(y_pred).permute(0, 2, 3, 4, 1).contiguous()
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x - mx, y - my
    print("shape of xm", xm.shape)
    print("shape of ym", ym.shape)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2))
    r = r_num / r_den
    r = torch.max(torch.min(r, torch.tensor(1.0)), torch.tensor(-1.0))

    return 1 - r**2


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
