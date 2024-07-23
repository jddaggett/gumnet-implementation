'''
This experimental version of GumNet makes use of the e3nn library for equivariant convolutions 
for targeted improvement of rotational transformation invariance. The Batch normalization has 
been modified for SE(3)-equivariant-tensor data. The matching module and rigid spatial 
transformer network remain unchanged. 
- Created 06/24/2024 Jackson Daggett
'''

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
from e3nn.nn import BatchNorm
from models.layers.FeatureCorrelation import FeatureCorrelation
from models.layers.FeatureL2Norm import FeatureL2Norm
from models.layers.RigidTransformation3DImputation import RigidTransformation3DImputation
from models.layers.SpectralPooling import SpectralPooling

class SE3ConvBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super(SE3ConvBlock, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear = Linear(irreps_in, irreps_out)
        self.batch_norm = BatchNorm(irreps_out)
        print(f"Initialized SE3ConvBlock with irreps_in: {irreps_in} and irreps_out: {irreps_out}")

    def forward(self, x):
        print(f"Input to SE3ConvBlock: {x.shape}")
        
        batch_size, channels, depth, height, width = x.shape
        # Flatten spatial dimensions and keep batch size and channel dimensions
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Shape: [batch_size, depth*height*width, channels]
        print(f"Reshaped input to SE3ConvBlock: {x.shape}")

        x = self.linear(x)
        print(f"After Linear in SE3ConvBlock: {x.shape}")

        x = self.batch_norm(x)
        print(f"After BatchNorm in SE3ConvBlock: {x.shape}")

        # Reshape back to 5D shape, but adjust for changed channel dimension
        new_channels = x.shape[-1]  # Get new number of channels after Linear layer
        x = x.permute(0, 2, 1).view(batch_size, new_channels, depth, height, width)
        print(f"Output of SE3ConvBlock: {x.shape}")

        return x

class GumNet(nn.Module):
    def __init__(self):
        super(GumNet, self).__init__()

        irreps_in = Irreps("1x0e")
        irreps_out1 = Irreps("32x0e + 32x1o")
        irreps_out2 = Irreps("64x0e + 64x1o")
        irreps_out3 = Irreps("128x0e + 128x1o")
        irreps_out4 = Irreps("256x0e + 256x1o")
        irreps_out5 = Irreps("512x0e + 512x1o")

        # Shared convolutional layers using SE3ConvBlock
        self.shared_conv1 = SE3ConvBlock(irreps_in, irreps_out1)
        self.shared_conv2 = SE3ConvBlock(irreps_out1, irreps_out2)
        self.shared_conv3 = SE3ConvBlock(irreps_out2, irreps_out3)
        self.shared_conv4 = SE3ConvBlock(irreps_out3, irreps_out4)
        self.shared_conv5 = SE3ConvBlock(irreps_out4, irreps_out5)

        # Spectral pooling layers
        self.spectral_pool1 = SpectralPooling((32, 32, 32), (26, 26, 26))
        self.spectral_pool2 = SpectralPooling((26, 26, 26), (22, 22, 22))
        self.spectral_pool3 = SpectralPooling((22, 22, 22), (15, 15, 15))
        self.spectral_pool4 = SpectralPooling((15, 15, 15), (10, 10, 10))
        self.spectral_pool5 = SpectralPooling((10, 10, 10), (7, 7, 7))

        self.relu = nn.ReLU()

        self.reduce_channels_ab = nn.Conv3d(343, 125, kernel_size=1)
        self.reduce_channels_ba = nn.Conv3d(343, 125, kernel_size=1)

        self.conv1_ab = nn.Conv3d(125, 256, kernel_size=3, padding=1)
        self.bn6_ab = nn.BatchNorm3d(256)
        self.conv2_ab = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn7_ab = nn.BatchNorm3d(512)

        self.conv1_ba = nn.Conv3d(125, 256, kernel_size=3, padding=1)
        self.bn6_ba = nn.BatchNorm3d(256)
        self.conv2_ba = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn7_ba = nn.BatchNorm3d(512)

        self.global_pool_ab = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_pool_ba = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(512 * 2, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 6)

        self.sigmoid = nn.Sigmoid()

        self.rigid_transform = RigidTransformation3DImputation(output_size=(32, 32, 32))

    def forward(self, sa, sb, mask1, mask2):
        def feature_extractor(x):
            print(f"Input to feature_extractor: {x.shape}")

            x = self.shared_conv1(x)
            x = self.relu(x)
            x = self.spectral_pool1(x)
            print(f"After shared_conv1: {x.shape}")

            x = self.shared_conv2(x)
            x = self.relu(x)
            x = self.spectral_pool2(x)
            print(f"After shared_conv2: {x.shape}")

            x = self.shared_conv3(x)
            x = self.relu(x)
            x = self.spectral_pool3(x)
            print(f"After shared_conv3: {x.shape}")

            x = self.shared_conv4(x)
            x = self.relu(x)
            x = self.spectral_pool4(x)
            print(f"After shared_conv4: {x.shape}")

            x = self.shared_conv5(x)
            x = self.relu(x)
            x = self.spectral_pool5(x)
            x = FeatureL2Norm()(x)
            print(f"After shared_conv5: {x.shape}")

            return x

        va = feature_extractor(sa)
        vb = feature_extractor(sb)

        print(f"After feature_extractor va: {va.shape}")
        print(f"After feature_extractor vb: {vb.shape}")

        c_ab = FeatureCorrelation()(va, vb)
        c_ab = FeatureL2Norm()(c_ab)
        print(f"After FeatureCorrelation c_ab: {c_ab.shape}")

        c_ba = FeatureCorrelation()(vb, va)
        c_ba = FeatureL2Norm()(c_ba)
        print(f"After FeatureCorrelation c_ba: {c_ba.shape}")

        c_ab = c_ab.permute(0, 4, 1, 2, 3)
        c_ab = self.reduce_channels_ab(c_ab)
        c_ab = self.conv1_ab(c_ab)
        c_ab = self.relu(self.bn6_ab(c_ab))
        c_ab = self.conv2_ab(c_ab)
        c_ab = self.relu(self.bn7_ab(c_ab))
        c_ab = self.global_pool_ab(c_ab)
        c_ab = c_ab.view(c_ab.size(0), -1)
        print(f"After processing c_ab: {c_ab.shape}")

        c_ba = c_ba.permute(0, 4, 1, 2, 3)
        c_ba = self.reduce_channels_ba(c_ba)
        c_ba = self.conv1_ba(c_ba)
        c_ba = self.relu(self.bn6_ba(c_ba))
        c_ba = self.conv2_ba(c_ba)
        c_ba = self.relu(self.bn7_ba(c_ba))
        c_ba = self.global_pool_ba(c_ba)
        c_ba = c_ba.view(c_ba.size(0), -1)
        print(f"After processing c_ba: {c_ba.shape}")

        c = torch.cat((c_ab, c_ba), dim=1)
        print(f"Concatenated c: {c.shape}")

        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        c = self.relu(c)
        c = self.sigmoid(self.fc3(c))
        print(f"Final c (6D rigid transform params): {c.shape}")

        sb_hat = self.rigid_transform(sb, sa, mask1, mask2, c)
        print(f"sb_hat: {sb_hat.shape}")

        return sb_hat, c