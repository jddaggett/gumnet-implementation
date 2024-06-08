import torch
import numpy as np
from models.gumnet_v2 import RigidTransformation3DImputation  

def test_identity_transformation(device):
    B, C, D, H, W = 1, 1, 4, 4, 4
    x = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    y = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    m1 = torch.ones(B, C, D, H, W, device=device, requires_grad=True)
    m2 = torch.zeros(B, C, D, H, W, device=device, requires_grad=True)
    
    identity_params = torch.full((B, 6), 0.5, device=device, requires_grad=True)
    model = RigidTransformation3DImputation(output_size=(D, H, W)).to(device)
    
    output = model(x, y, m1, m2, identity_params)
    
    assert torch.allclose(x, output, atol=1e-6), "Identity test failed!"

def test_rotation_transformation(device):
    B, C, D, H, W = 1, 1, 3, 3, 3
    x = torch.tensor([[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]]], dtype=torch.float32, device=device, requires_grad=True)
    y = torch.zeros_like(x, device=device, requires_grad=True)
    m1 = torch.ones_like(x, device=device, requires_grad=True)
    m2 = torch.zeros_like(x, device=device, requires_grad=True)
    
    rotation_90_x = torch.tensor([0.75, 0.5, 0.5], device=device, requires_grad=True)
    translation_0 = torch.tensor([0.5, 0.5, 0.5], device=device, requires_grad=True)
    rotation_params = torch.cat((rotation_90_x, translation_0)).unsqueeze(0).repeat(B, 1)
    
    model = RigidTransformation3DImputation(output_size=(D, H, W)).to(device)
    
    output = model(x, y, m1, m2, rotation_params)
    
    expected_output = torch.tensor([[[[[7, 4, 1], [8, 5, 2], [9, 6, 3]],
                                      [[16, 13, 10], [17, 14, 11], [18, 15, 12]],
                                      [[25, 22, 19], [26, 23, 20], [27, 24, 21]]]]], dtype=torch.float32, device=device)
    
    assert torch.allclose(output, expected_output, atol=1e-6), "Rotation test failed!"

def test_translation_transformation(device):
    B, C, D, H, W = 1, 1, 3, 3, 3
    x = torch.tensor([[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]]], dtype=torch.float32, device=device, requires_grad=True)
    y = torch.zeros_like(x, device=device, requires_grad=True)
    m1 = torch.ones_like(x, device=device, requires_grad=True)
    m2 = torch.zeros_like(x, device=device, requires_grad=True)
    
    rotation_0 = torch.tensor([0.5, 0.5, 0.5], device=device, requires_grad=True)
    translation_1_x = torch.tensor([1.0, 0.5, 0.5], device=device, requires_grad=True)
    translation_params = torch.cat((rotation_0, translation_1_x)).unsqueeze(0).repeat(B, 1)
    
    model = RigidTransformation3DImputation(output_size=(D, H, W)).to(device)
    
    output = model(x, y, m1, m2, translation_params)
    
    expected_output = torch.tensor([[[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                      [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]]], dtype=torch.float32, device=device)
    print("Expected:", expected_output)
    print("Got:", output)
    assert torch.allclose(output, expected_output, atol=1e-6), "Translation test failed!"

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nTesting identity transform...")
    test_identity_transformation(device)
    print("...passed!")
    print("\nTesting rotation-only transform...")
    test_rotation_transformation(device)
    print("...passed!")
    print("\nTesting translation-only transform...")
    test_translation_transformation(device)
    print("...passed!")