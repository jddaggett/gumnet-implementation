import pickle
import numpy as np
import torch
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

# Generates usable data by the model from the GroEL-ES dataset
def load_GroEL_ES():
    # Load data
    with open('../dataset/GroEl-ES.pickle', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Shuffle data
    np.random.shuffle(data)

    # Split data as defined in GumNet paper
    num_training = 617
    num_validation = 69
    num_test = 100
    train_data = data[:num_training]
    valid_data = data[num_training:num_training + num_validation]
    test_data = data[num_training + num_validation:num_training + num_validation + num_test]

    # Convert data to torch tensors and reshape to [B, C, D, H, W]
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x = torch.tensor(train_data, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    valid_x = torch.tensor(valid_data, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    test_x = torch.tensor(test_data, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)

    # Helper for generating grid to transform x by theta
    def gen_grid(x, theta):
        B, C, D, H, W = x.shape

        # Build affine transformation matrix
        R = euler_to_rot_matrix(theta)
        T = theta[:, 3:]
        affine_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        affine_matrix[:, :3, :3] = R
        affine_matrix[:, 0, 3] = T[:, 0] / W * 2  # Normalize by width
        affine_matrix[:, 1, 3] = T[:, 1] / H * 2  # Normalize by height
        affine_matrix[:, 2, 3] = T[:, 2] / D * 2  # Normalize by depth
        affine_matrix = affine_matrix[:, :3]
        
        # Generate grid
        grid = F.affine_grid(affine_matrix, [B, C, D, H, W], align_corners=False)
        return grid

    # Generate ground truth 6D transformation parameters for evaluating the model
    ground_truth = torch.rand(num_test, 6)

    # Generate and sample affine grids to get target subtomagrams
    train_grid = gen_grid(train_x, torch.rand(num_training, 6))
    valid_grid = gen_grid(valid_x, torch.rand(num_validation, 6))
    test_grid = gen_grid(test_x, ground_truth) # transform the test by the ground truth params
    train_y = F.grid_sample(train_x, train_grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    valid_y = F.grid_sample(valid_x, valid_grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    test_y = F.grid_sample(test_x, test_grid, align_corners=False, mode='bilinear', padding_mode='zeros')

    # Generate training data masks: @TODO these are placeholders
    observed_mask = torch.ones(32, 32, 32).to(device).unsqueeze(0).unsqueeze(0).repeat(train_x.shape[0], 1, 1, 1, 1)
    missing_mask = torch.zeros(32, 32, 32).to(device).unsqueeze(0).unsqueeze(0).repeat(train_x.shape[0], 1, 1, 1, 1)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, observed_mask, missing_mask, ground_truth