import torch
from models.gumnet_v2 import GumNet

def test_gumnet_forward_pass():
    model = GumNet()
    sa = torch.rand(1, 1, 32, 32, 32)
    sb = torch.rand(1, 1, 32, 32, 32)
    mask1 = torch.rand(1, 1, 32, 32, 32)
    mask2 = torch.rand(1, 1, 32, 32, 32)
    sb_hat, c = model(sa, sb, mask1, mask2)
    
    assert sb_hat.shape == sa.shape, "GumNet forward pass output has incorrect shape"
    assert c.shape == (1, 6), "GumNet forward pass parameters have incorrect shape"
