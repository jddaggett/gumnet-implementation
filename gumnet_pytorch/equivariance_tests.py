import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate
from models.gumnet_v2 import GumNet
from utils import *
from process_data import load_GroEL_ES, load_rat_data

def rotate_3d_volume(volume, angle, axes=(1, 2)):
    # use scipy rotate for simplicity
    return rotate(volume, angle, axes=axes, reshape=False, mode='constant', cval=0.0)

def get_model_output(model, x, y, m1, m2, device):
    model.eval()
    with torch.no_grad():
        x, y, m1, m2 = x.to(device), y.to(device), m1.to(device), m2.to(device)
        output, _ = model(x, y, m1, m2)
    return output.cpu().numpy()

# Returns a list of tuples (angle, error) where angle is the angle of rotation being tested
# and error is the experimental equivariance error (expect 0 for an equivariant network)
def test_rotational_equivariance(model, inputs, masks, targets, angles, device):
    results = []
    model = model.to(device)
    for x, y, m1, m2 in zip(inputs, targets, masks[0], masks[1]):

        # Add batch dimension to tensors
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        m1 = m1.unsqueeze(0)
        m2 = m2.unsqueeze(0)

        # Output generated by model from unmodified input tensors
        original_output = get_model_output(model, x, y, m1, m2, device)
        
        for angle in angles:
            rotated_x = torch.tensor(rotate_3d_volume(x.cpu().numpy(), angle)).to(device)
            rotated_y = torch.tensor(rotate_3d_volume(y.cpu().numpy(), angle)).to(device)
            rotated_m1 = torch.tensor(rotate_3d_volume(m1.cpu().numpy(), angle)).to(device)
            rotated_m2 = torch.tensor(rotate_3d_volume(m2.cpu().numpy(), angle)).to(device)

            # Output generated by model from input tensors rotated by angle
            rotated_output = get_model_output(model, rotated_x, rotated_y, rotated_m1, rotated_m2, device)
            
            # Original output rotated by angle
            rotated_original_output = rotate_3d_volume(original_output, angle)
            
            # Error is difference between rotated output and output of rotated input
            equivariance_error = np.linalg.norm(rotated_output - rotated_original_output)
            results.append((angle, equivariance_error))
    
    return results

def main():
    # Load data and preprocess
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, _, _, test_x, test_y, observed_mask, missing_mask, _, _ = load_GroEL_ES(device)
    
    # Normalize the data
    test_x = (test_x - test_x.mean()) / test_x.std()
    test_y = (test_y - test_y.mean()) / test_y.std()
    
    model = GumNet()
    
    # Test rotational equivariance
    angles = [0, 30, 60, 90, 120, 150, 180]  # Angles to test
    inputs = test_x[:5]  # Select a few test inputs for evaluation
    targets = test_y[:5]
    masks = (observed_mask[:5], missing_mask[:5])
    
    results = test_rotational_equivariance(model, inputs, masks, targets, angles, device)
    
    # Print results
    for angle, error in results:
        if error != 0.0:
            print(f"Angle: {angle}, Equivariance Error: {error}")

if __name__ == '__main__':
    main()