import torch
import torch.nn as nn
import numpy as np
import pickle
from models.layers.RigidTransformation3DImputation import RigidTransformation3DImputation

def correlation_coefficient_loss(y_true, y_pred):
    y_true_mean = torch.mean(y_true, dim=[2, 3, 4], keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=[2, 3, 4], keepdim=True)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    covariance = torch.sum(y_true_centered * y_pred_centered, dim=[2, 3, 4])
    y_true_std = torch.sqrt(torch.sum(y_true_centered ** 2, dim=[2, 3, 4]))
    y_pred_std = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=[2, 3, 4]))

    correlation = covariance / (y_true_std * y_pred_std + 1e-6)
    return 1 - correlation.mean()

def stn_wrapper(x, y, mask1, mask2, ground_truth_params, device):
    x = torch.tensor(x, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    y = torch.tensor(y, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    mask1 = torch.tensor(mask1, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    mask2 = torch.tensor(mask2, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    ground_truth_params = torch.tensor(ground_truth_params, dtype=torch.float32).to(device)

    rigid_transform = RigidTransformation3DImputation(output_size=(32, 32, 32)).to(device)
    y_pred = rigid_transform(x, y, mask1, mask2, ground_truth_params)
    
    return y_pred

def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the test data
    with open('../dataset/gum_demo_data.pickle', 'rb') as f:
        x_test, y_test, observed_mask, missing_mask, ground_truth_params = pickle.load(f, encoding='latin1')
    
    # Normalize data
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)
    
    # Create masks
    mask_1 = np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
    mask_2 = np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1))

    # Generate output from rigid transformation layer given sa, sb, and ground truth parameters
    y_pred = stn_wrapper(x_test, y_test, mask_1, mask_2, ground_truth_params, device)
    
    # Generate random tensor with the same shape as y_test for comparison
    y_random = np.random.randn(*y_test.shape)

    # Convert to tensors for loss calculation
    y_test = torch.tensor(y_test, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    y_random = torch.tensor(y_random, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)

    # Compare the output to the ground truth
    voxel_diff = correlation_coefficient_loss(y_test, y_pred)
    print(f'Truth correlation loss: {voxel_diff.item()}')

    # Compare a random tensor to the ground truth
    random_diff = correlation_coefficient_loss(y_test, y_random)
    print(f'Random correlation loss: {random_diff.item()}')

if __name__ == '__main__':
    main()
