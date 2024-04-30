import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from gumnet_pytorch.models.layers.FeatureCorrelation import FeatureCorrelation
from gumnet_pytorch.models.layers.FeatureL2Norm import FeatureL2Norm
from gumnet_pytorch.models.layers.RigidTransformation3DImputation import RigidTransformation3DImputation
from gumnet_pytorch.models.layers.SpectralPooling import SpectralPooling
from gumnet_pytorch.utils import get_initial_weights, correlation_coefficient_loss, alignment_eval

class GumNet(nn.Module):
    def __init__(self, img_shape=(32, 32, 32)):
        super(GumNet, self).__init__()
        # Assume channel first for PyTorch (N, C, D, H, W)
        in_channels = 1  # Adjust based on your input

        # Define shared convolutional layers
        self.shared_conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=0)
        self.shared_conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0)
        self.shared_conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0)
        self.shared_conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=0)
        self.shared_conv5 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=0)
        
        # Define batch normalization per convolutional layer
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)

        # Assuming SpectralPooling and FeatureL2Norm are custom classes defined elsewhere
        self.spectral_pool = SpectralPooling()
        self.feature_l2norm = FeatureL2Norm()

    def forward(self, main_input, auxiliary_input, mask_1, mask_2):
        v_a = self.process_stream(main_input)
        v_b = self.process_stream(auxiliary_input)

        # Assuming FeatureCorrelation is a custom class that needs inputs as a list
        c_ab = FeatureCorrelation()(v_a, v_b)
        c_ab = self.feature_l2norm(c_ab)
        
        c_ba = FeatureCorrelation()(v_b, v_a)
        c_ba = self.feature_l2norm(c_ba)
        
        # Additional layers and flattening
        c_ab = self.shared_conv_extra1(c_ab)
        c_ab = self.bn_extra1(c_ab)
        c_ab = F.relu(c_ab)

        c_ba = self.shared_conv_extra2(c_ba)
        c_ba = self.bn_extra2(c_ba)
        c_ba = F.relu(c_ba)

        c_ab = c_ab.view(c_ab.size(0), -1)  # Flatten
        c_ba = c_ba.view(c_ba.size(0), -1)  # Flatten

        combined = torch.cat((c_ab, c_ba), dim=1)
        out = self.final_layers(combined)

        # Handling transformation and output
        x, mask1, mask2 = RigidTransformation3DImputation()(main_input, auxiliary_input, mask_1, mask_2, combined)
        return x

    def process_stream(self, x):
        x = F.relu(self.bn1(self.shared_conv1(x)))
        x = self.spectral_pool(x)
        x = F.relu(self.bn2(self.shared_conv2(x)))
        x = self.spectral_pool(x)
        # Repeat for other layers
        x = self.feature_l2norm(x)
        return x
        # Additional convolutional layers defined in __init__
        self.shared_conv_extra1 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=0)
        self.shared_conv_extra2 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=0)
        self.bn_extra1 = nn.BatchNorm3d(1024)
        self.bn_extra2 = nn.BatchNorm3d(1024)

        # Dense layers for final output
        self.final_layers = nn.Sequential(
            nn.Linear(2048, 2000),  # Adjust the input size based on the output of Flatten
            nn.ReLU(),
            nn.Linear(2000, 6),
            nn.Sigmoid()
        )
