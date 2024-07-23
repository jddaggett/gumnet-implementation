import torch
import torch.nn as nn
from models.layers.FeatureCorrelation import FeatureCorrelation
from models.layers.FeatureL2Norm import FeatureL2Norm
from models.layers.STN import STN
from models.layers.SpectralPooling import SpectralPooling

class GumNetNoSTN(nn.Module):
    def __init__(self):
        super(GumNetNoSTN, self).__init__()
        self.shared_conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.shared_conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.shared_conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.shared_conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.shared_conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)

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

    def forward(self, sa, sb):
        def feature_extractor(x):
            x = self.shared_conv1(x)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.spectral_pool1(x)

            x = self.shared_conv2(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = self.spectral_pool2(x)

            x = self.shared_conv3(x)
            x = self.relu(x)
            x = self.bn3(x)
            x = self.spectral_pool3(x)

            x = self.shared_conv4(x)
            x = self.relu(x)
            x = self.bn4(x)
            x = self.spectral_pool4(x)

            x = self.shared_conv5(x)
            x = self.relu(x)
            x = self.bn5(x)
            x = self.spectral_pool5(x)
            x = FeatureL2Norm()(x)

            return x

        va = feature_extractor(sa)
        vb = feature_extractor(sb)

        c_ab = FeatureCorrelation()(va, vb)
        c_ab = FeatureL2Norm()(c_ab)

        c_ba = FeatureCorrelation()(vb, va)
        c_ba = FeatureL2Norm()(c_ba)

        c_ab = c_ab.permute(0, 4, 1, 2, 3)
        c_ab = self.reduce_channels_ab(c_ab)
        c_ab = self.conv1_ab(c_ab)
        c_ab = self.relu(self.bn6_ab(c_ab))
        c_ab = self.conv2_ab(c_ab)
        c_ab = self.relu(self.bn7_ab(c_ab))
        c_ab = self.global_pool_ab(c_ab)
        c_ab = c_ab.view(c_ab.size(0), -1)

        c_ba = c_ba.permute(0, 4, 1, 2, 3)
        c_ba = self.reduce_channels_ba(c_ba)
        c_ba = self.conv1_ba(c_ba)
        c_ba = self.relu(self.bn6_ba(c_ba))
        c_ba = self.conv2_ba(c_ba)
        c_ba = self.relu(self.bn7_ba(c_ba))
        c_ba = self.global_pool_ba(c_ba)
        c_ba = c_ba.view(c_ba.size(0), -1)

        c = torch.cat((c_ab, c_ba), dim=1)

        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        c = self.relu(c)
        c = self.sigmoid(self.fc3(c))

        return c
