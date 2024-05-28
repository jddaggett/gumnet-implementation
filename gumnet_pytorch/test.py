import torch
import numpy as np
from models.layers.RigidTransformation3DImputation import *

def generate_synthetic_data(batch_size, channels, depth, height, width):
    X = torch.rand(batch_size, channels, depth, height, width, dtype=torch.float32)
    Y = torch.rand(batch_size, channels, depth, height, width, dtype=torch.float32)
    m1 = torch.randint(0, 2, (batch_size, depth, height, width), dtype=torch.float32)
    m2 = torch.randint(0, 2, (batch_size, depth, height, width), dtype=torch.float32)
    theta = torch.rand(batch_size, 6, dtype=torch.float32)
    return X, Y, m1, m2, theta

def main():
    batch_size = 2
    channels = 1
    depth = 8
    height = 8
    width = 8
    output_size = (depth, height, width)
    
    # Generate synthetic data
    X, Y, m1, m2, theta = generate_synthetic_data(batch_size, channels, depth, height, width)
    
    # Instantiate the model
    model = RigidTransformation3DImputation(output_size, padding_method="fill")
    model = model.to(model.device)
    
    # Run the forward pass
    result = model(X, Y, m1, m2, theta)
    
    # Check the output
    print("Output shape:", result.shape)
    print("Output tensor:", result)

if __name__ == "__main__":
    main()
