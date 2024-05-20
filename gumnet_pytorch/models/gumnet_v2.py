import torch
import torch.nn as nn
from models.layers.FeatureCorrelation import FeatureCorrelation
from models.layers.FeatureL2Norm import FeatureL2Norm
from models.layers.RigidTransformation3DImputation import RigidTransformation3DImputation
from models.layers.SpectralPooling import SpectralPooling

"""
Improvement:
1.The convolutional layers are still defined one by one, but with direct assignment of batch normalization right after each convolutional layer definition.
2.uses FeatureL2Norm() after processing the streams, uses FeatureCorrelation() after each stream processing and applies feature normalization to its output
3.Add extra layers such as batch normalization and ReLU activation.
4.Shows the concatenation of flattened outputs from correlation maps, followed by dense layers to produce the final transformation parameters.
"""

class GumNet(nn.Module):
    def __init__(self):
        super(GumNet, self).__init__()
        # Define the shared convolutional layers
        self.shared_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=0)
        self.shared_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0)
        self.shared_conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0)
        self.shared_conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0)
        self.shared_conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0)

        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)

        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(93312, 2000) # size obtained experimentally
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sa, sb):
        # Process input through shared conv layers and apply spectral pooling
        print("Input shape sa:", sa.shape)
        va = self.shared_conv1(sa)
        print("After shared_conv1 va:", va.shape)
        va = self.bn1(va)
        va = self.relu(va)
        va = SpectralPooling((26, 26, 26), (22, 22, 22))(va)
        print("After spectral pooling va:", va.shape)

        va = self.shared_conv2(va)
        print("After shared_conv2 va:", va.shape)
        va = self.bn2(va)
        va = self.relu(va)
        va = SpectralPooling((18, 18, 18), (15, 15, 15))(va)
        print("After spectral pooling va:", va.shape)

        va = self.shared_conv3(va)
        print("After shared_conv3 va:", va.shape)
        va = self.bn3(va)
        va = self.relu(va)
        va = SpectralPooling((12, 12, 12), (10, 10, 10))(va)
        print("After spectral pooling va:", va.shape)

        va = self.shared_conv4(va)
        print("After shared_conv4 va:", va.shape)
        va = self.bn4(va)
        va = self.relu(va)
        va = SpectralPooling((8, 8, 8), (7, 7, 7))(va)
        print("After spectral pooling va:", va.shape)

        va = self.shared_conv5(va)
        print("After shared_conv5 va:", va.shape)
        va = self.bn5(va)
        va = self.relu(va)
        va = FeatureL2Norm()(va)
        print("After FeatureL2Norm va:", va.shape)

        # Repeat for sb
        print("Input shape sb:", sb.shape)
        vb = self.shared_conv1(sb)
        print("After shared_conv1 vb:", vb.shape)
        vb = self.bn1(vb)
        vb = self.relu(vb)
        vb = SpectralPooling((26, 26, 26), (22, 22, 22))(vb)
        print("After spectral pooling vb:", vb.shape)

        vb = self.shared_conv2(vb)
        print("After shared_conv2 vb:", vb.shape)
        vb = self.bn2(vb)
        vb = self.relu(vb)
        vb = SpectralPooling((18, 18, 18), (15, 15, 15))(vb)
        print("After spectral pooling vb:", vb.shape)

        vb = self.shared_conv3(vb)
        print("After shared_conv3 vb:", vb.shape)
        vb = self.bn3(vb)
        vb = self.relu(vb)
        vb = SpectralPooling((12, 12, 12), (10, 10, 10))(vb)
        print("After spectral pooling vb:", vb.shape)

        vb = self.shared_conv4(vb)
        print("After shared_conv4 vb:", vb.shape)
        vb = self.bn4(vb)
        vb = self.relu(vb)
        vb = SpectralPooling((8, 8, 8), (7, 7, 7))(vb)
        print("After spectral pooling vb:", vb.shape)

        vb = self.shared_conv5(vb)
        print("After shared_conv5 vb:", vb.shape)
        vb = self.bn5(vb)
        vb = self.relu(vb)
        vb = FeatureL2Norm()(vb)
        print("After FeatureL2Norm vb:", vb.shape)

        # Apply correlation and flatten for fully connected layers
        c_ab = FeatureCorrelation()(va, vb).view(va.size(0), -1)
        c_ba = FeatureCorrelation()(vb, va).view(vb.size(0), -1)

        # Concatenate and pass through fully connected layers
        c = torch.cat((c_ab, c_ba), dim=1)
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        c = self.relu(c)
        c = self.fc3(c)
        c = self.sigmoid(c)

        # Apply transformation
        transformed, M1_t, M2_t = RigidTransformation3DImputation((32,32,32))(sa, sb, c)

        return transformed, c

# Example instantiation and forward pass
# model = GumNet()
# Define sa and sb according to your actual data shape and pass them through the model
# sa, sb = torch.randn(...), torch.randn(...)
# output = model(sa, sb)
