import torch
from models.layers.FeatureCorrelation import FeatureCorrelation
from models.layers.FeatureL2Norm import FeatureL2Norm
from models.layers.RigidTransformation3DImputation import RigidTransformation3DImputation
from models.layers.SpectralPooling import SpectralPooling

def test_feature_correlation():
    fc = FeatureCorrelation()
    x = torch.rand(1, 256, 8, 8, 8)
    y = torch.rand(1, 256, 8, 8, 8)
    out = fc(x, y)
    assert out.shape == (1, 8, 8, 8, 512), "FeatureCorrelation output has incorrect shape"

def test_feature_l2_norm():
    l2_norm = FeatureL2Norm()
    x = torch.rand(1, 256, 8, 8, 8)
    out = l2_norm(x)
    
    norm_values = out.norm(p=2, dim=1)
    print("Norm values:", norm_values)
    
    assert out.shape == x.shape, "FeatureL2Norm output has incorrect shape"
    assert torch.allclose(norm_values, torch.ones_like(norm_values), rtol=1e-3, atol=1e-3), "FeatureL2Norm output is not normalized"

def test_rigid_transformation_3d_imputation():
    rigid_transform = RigidTransformation3DImputation(output_size=(32, 32, 32))
    sa = torch.rand(1, 1, 32, 32, 32)
    sb = torch.rand(1, 1, 32, 32, 32)
    mask1 = torch.rand(1, 1, 32, 32, 32)
    mask2 = torch.rand(1, 1, 32, 32, 32)
    params = torch.rand(1, 6)
    out = rigid_transform(sa, sb, mask1, mask2, params)
    assert out.shape == sa.shape, "RigidTransformation3DImputation output has incorrect shape"

def test_spectral_pooling():
    sp = SpectralPooling((26, 26, 26), (22, 22, 22))
    x = torch.rand(1, 32, 26, 26, 26)
    out = sp(x)
    assert out.shape == (1, 32, 22, 22, 22), "SpectralPooling output has incorrect shape"
