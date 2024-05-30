import torch
import torch.nn as nn
from models.layers.FeatureCorrelation import FeatureCorrelation
from models.layers.FeatureL2Norm import FeatureL2Norm
from models.layers.RigidTransformation3DImputation import RigidTransformation3DImputation
from models.layers.SpectralPooling import SpectralPooling

class GumNet(nn.Module):
    def __init__(self):
        super(GumNet, self).__init__()
        # Define the shared convolutional layers
        self.shared_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=0)
        self.shared_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0)
        self.shared_conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0)
        self.shared_conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0)
        self.shared_conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0)

        # Define the batch normalization layers
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)

        # Define dropout layers
        self.dropout1 = nn.Dropout3d(0.3)
        self.dropout2 = nn.Dropout3d(0.3)
        self.dropout3 = nn.Dropout3d(0.3)
        self.dropout4 = nn.Dropout3d(0.3)
        self.dropout5 = nn.Dropout3d(0.3)

        # Define the spectral pooling layers
        self.spectral_pool1 = SpectralPooling((26, 26, 26), (22, 22, 22))
        self.spectral_pool2 = SpectralPooling((18, 18, 18), (15, 15, 15))
        self.spectral_pool3 = SpectralPooling((12, 12, 12), (10, 10, 10))
        self.spectral_pool4 = SpectralPooling((8, 8, 8), (7, 7, 7))

        self.relu = nn.ReLU()

        # Additional convolutional layers after correlation
        self.reduce_channels_ab = nn.Conv3d(125, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.reduce_channels_ba = nn.Conv3d(125, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.conv1 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0)  # Adjust input channels
        self.bn6 = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.bn7 = nn.BatchNorm3d(1024)

        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 2, 2000)  # Adjusted input size based on conv layers
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 6)
        self.sigmoid = nn.Sigmoid()

        # Rigid transformation layer
        self.rigid_transform = RigidTransformation3DImputation(output_size=(32, 32, 32))

    def forward(self, sa, sb, mask1, mask2):
        # Process input through shared conv layers and apply spectral pooling
        va = self.shared_conv1(sa)
        va = self.relu(va)
        va = self.bn1(va)
        va = self.dropout1(va)
        va = self.spectral_pool1(va)

        va = self.shared_conv2(va)
        va = self.relu(va)
        va = self.bn2(va)
        va = self.dropout2(va)
        va = self.spectral_pool2(va)

        va = self.shared_conv3(va)
        va = self.relu(va)
        va = self.bn3(va)
        va = self.dropout3(va)
        va = self.spectral_pool3(va)

        va = self.shared_conv4(va)
        va = self.relu(va)
        va = self.bn4(va)
        va = self.dropout4(va)
        va = self.spectral_pool4(va)

        va = self.shared_conv5(va)
        va = self.relu(va)
        va = self.bn5(va)
        va = self.dropout5(va)
        va = FeatureL2Norm()(va)

        # Repeat for sb
        vb = self.shared_conv1(sb)
        vb = self.relu(vb)
        vb = self.bn1(vb)
        vb = self.dropout1(vb)
        vb = self.spectral_pool1(vb)

        vb = self.shared_conv2(vb)
        vb = self.relu(vb)
        vb = self.bn2(vb)
        vb = self.dropout2(vb)
        vb = self.spectral_pool2(vb)

        vb = self.shared_conv3(vb)
        vb = self.relu(vb)
        vb = self.bn3(vb)
        vb = self.dropout3(vb)
        vb = self.spectral_pool3(vb)

        vb = self.shared_conv4(vb)
        vb = self.relu(vb)
        vb = self.bn4(vb)
        vb = self.dropout4(vb)
        vb = self.spectral_pool4(vb)

        vb = self.shared_conv5(vb)
        vb = self.relu(vb)
        vb = self.bn5(vb)
        vb = self.dropout5(vb)
        vb = FeatureL2Norm()(vb)

        # Apply correlation
        c_ab = FeatureCorrelation()(va, vb)
        c_ab = FeatureL2Norm()(c_ab)

        c_ba = FeatureCorrelation()(vb, va)
        c_ba = FeatureL2Norm()(c_ba)

        # Reshape for convolutional layers
        c_ab = c_ab.permute(0, 4, 1, 2, 3)
        c_ba = c_ba.permute(0, 4, 1, 2, 3)
        c_ab = self.reduce_channels_ab(c_ab)
        c_ba = self.reduce_channels_ba(c_ba)

        # Apply additional convolutional layers before flattening
        c_ab = self.conv1(c_ab)
        c_ab = self.bn6(c_ab)
        c_ab = self.relu(c_ab)
        c_ab = self.conv2(c_ab)
        c_ab = self.bn7(c_ab)
        c_ab = self.relu(c_ab)

        c_ba = self.conv1(c_ba)
        c_ba = self.bn6(c_ba)
        c_ba = self.relu(c_ba)
        c_ba = self.conv2(c_ba)
        c_ba = self.bn7(c_ba)
        c_ba = self.relu(c_ba)

        # Flatten
        c_ab = c_ab.view(c_ab.size(0), -1)
        c_ba = c_ba.view(c_ba.size(0), -1)

        # Concatenate and pass through fully connected layers
        c = torch.cat((c_ab, c_ba), dim=1)
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        c = self.relu(c)
        c = self.fc3(c)
        c = self.sigmoid(c)

        # Apply transformation
        sb_hat = self.rigid_transform(sa, sb, mask1, mask2, c)

        return sb_hat, c  # sb_hat and 6D transformation parameters
