import torch
import numpy as np
from models.gumnet_v2 import RigidTransformation3DImputation  # Adjust import path as necessary

def test_identity_transformation(device):
    batches, channels, depth, height, width = 1, 1, 4, 4, 4
    x = torch.randn(batches, channels, depth, height, width, device=device)
    y = x.clone()
    m1 = torch.ones(batches, channels, depth, height, width, device=device)
    m2 = torch.zeros(batches, channels, depth, height, width, device=device)
    theta = torch.zeros(batches, 6, device=device)

    stn = RigidTransformation3DImputation((4, 4, 4))
    output = stn(x, y, m1, m2, theta)
    
    print("Input (sa):", x)
    print("Output (out):", output)
    assert torch.allclose(x, output, atol=1e-6), "Identity test failed!"

def test_rotation_transformation(device):
    sa = torch.tensor([[[[[0.8694, 0.3753, 0.6985, 0.9774],
                           [0.4094, 0.9117, 0.0106, 0.8664],
                           [0.3845, 0.5522, 0.3040, 0.0282],
                           [0.6711, 0.5991, 0.2864, 0.8608]],

                          [[0.1012, 0.5894, 0.3106, 0.9291],
                           [0.4679, 0.7205, 0.3826, 0.3833],
                           [0.7721, 0.1182, 0.4129, 0.3038],
                           [0.6882, 0.7790, 0.2505, 0.6561]],

                          [[0.7483, 0.9034, 0.6561, 0.0531],
                           [0.4258, 0.3100, 0.1325, 0.5719],
                           [0.8835, 0.7523, 0.8981, 0.8873],
                           [0.4764, 0.1077, 0.6780, 0.5621]],

                          [[0.9625, 0.5915, 0.7727, 0.0531],
                           [0.7735, 0.8792, 0.5279, 0.5015],
                           [0.2830, 0.9211, 0.1128, 0.9877],
                           [0.1514, 0.1432, 0.2779, 0.7406]]]]], device=device)
    sb = sa.clone().detach().to(device)
    mask1 = torch.ones_like(sa).to(device)
    mask2 = torch.zeros_like(sa).to(device)
    rotation_params = torch.tensor([[0.5, 0.5, 0.5, 0., 0., 0.]]).to(device)
    rotation_params[:, :3] = rotation_params[:, :3] * 2 * 3.141592653589793  # Scale to [0, 2*pi]

    model = RigidTransformation3DImputation((4, 4, 4)).to(device)
    out = model(sa, sb, mask1, mask2, rotation_params)

    print("Input (sa):", sa)
    print("Output (out):", out)
    assert not torch.allclose(out, sa, atol=1e-5), "Rotation transformation did not change the input as expected"

def test_translation_transformation(device):
    sa = torch.tensor([[[[[0.8694, 0.3753, 0.6985, 0.9774],
                           [0.4094, 0.9117, 0.0106, 0.8664],
                           [0.3845, 0.5522, 0.3040, 0.0282],
                           [0.6711, 0.5991, 0.2864, 0.8608]],

                          [[0.1012, 0.5894, 0.3106, 0.9291],
                           [0.4679, 0.7205, 0.3826, 0.3833],
                           [0.7721, 0.1182, 0.4129, 0.3038],
                           [0.6882, 0.7790, 0.2505, 0.6561]],

                          [[0.7483, 0.9034, 0.6561, 0.0531],
                           [0.4258, 0.3100, 0.1325, 0.5719],
                           [0.8835, 0.7523, 0.8981, 0.8873],
                           [0.4764, 0.1077, 0.6780, 0.5621]],

                          [[0.9625, 0.5915, 0.7727, 0.0531],
                           [0.7735, 0.8792, 0.5279, 0.5015],
                           [0.2830, 0.9211, 0.1128, 0.9877],
                           [0.1514, 0.1432, 0.2779, 0.7406]]]]], device=device)
    sb = sa.clone().detach().to(device)
    mask1 = torch.ones_like(sa).to(device)
    mask2 = torch.zeros_like(sa).to(device)
    translation_params = torch.tensor([[0., 0., 0., 0.5, 0.5, 0.5]]).to(device)
    translation_params[:, 3:] = (translation_params[:, 3:] - 0.5) * 2  # Scale to [-1, 1]

    model = RigidTransformation3DImputation((4, 4, 4)).to(device)
    out = model(sa, sb, mask1, mask2, translation_params)

    print("Input (sa):", sa)
    print("Output (out):", out)
    assert not torch.allclose(out, sa, atol=1e-5), "Translation transformation did not change the input as expected"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_identity_transformation(device)
    test_rotation_transformation(device)
    test_translation_transformation(device)