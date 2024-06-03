import pytest
import torch
from models.gumnet_v2 import GumNet
from utils import initialize_weights

class TestGumNetComponents:
    
    @staticmethod
    def get_sample_input():
        batch_size = 4
        channels = 1
        depth = 32
        height = 32
        width = 32
        x = torch.randn(batch_size, channels, depth, height, width)
        y = torch.randn(batch_size, channels, depth, height, width)
        observed_mask = torch.ones(batch_size, channels, depth, height, width)
        missing_mask = torch.zeros(batch_size, channels, depth, height, width)
        return x, y, observed_mask, missing_mask
    
    def test_gumnet_initialization(self):
        model = GumNet()
        assert model is not None, "Model initialization failed"
        assert isinstance(model, GumNet), "Model is not an instance of GumNet"
    
    def test_gumnet_forward_pass(self):
        model = GumNet()
        x, y, observed_mask, missing_mask = self.get_sample_input()

        model.eval()
        with torch.no_grad():
            output, params = model(x, y, observed_mask, missing_mask)
        
        assert output.shape == x.shape, "Output shape mismatch"
        assert params is not None, "Params should not be None"

    def test_initialize_weights(self):
        model = GumNet()
        model.apply(initialize_weights)

        for name, param in model.named_parameters():
            if 'weight' in name:
                assert torch.abs(param.mean()).item() < 1e-2, f"{name} weight initialization issue"
            if 'bias' in name:
                assert torch.abs(param.mean()).item() < 1e-2, f"{name} bias initialization issue"

    def test_conv_layer(self):
        model = GumNet()
        sample_input = torch.randn(4, 1, 32, 32, 32)
        conv_layer = model.shared_conv1
        
        output = conv_layer(sample_input)
        expected_output_shape = (4, 1, 32, 32, 32)
        
        assert output.shape == expected_output_shape, f"Conv layer output shape mismatch: expected {expected_output_shape}, got {output.shape}"

    def test_batch_norm_layer(self):
        model = GumNet()
        sample_input = torch.randn(4, 1, 32, 32, 32)
        batch_norm_layer = model.bn1 
        
        output = batch_norm_layer(sample_input)
        assert output.shape == sample_input.shape, "Batch norm layer output shape mismatch"

    def test_activation_layer(self):
        model = GumNet()
        sample_input = torch.randn(4, 1, 32, 32, 32) 
        activation_layer = model.relu
        
        output = activation_layer(sample_input)
        assert output.shape == sample_input.shape, "Activation layer output shape mismatch"

if __name__ == "__main__":
    pytest.main()
