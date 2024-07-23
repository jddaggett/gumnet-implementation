import torch
from models.gumnet_v2 import GumNet
from utils import correlation_coefficient_loss_params, generate_masks

def numerical_gradient(f, param, epsilon=1e-5):
    grad = torch.zeros_like(param)
    flat_param = param.view(-1)
    for ix in range(flat_param.size(0)):
        old_value = flat_param[ix].item()
        
        flat_param[ix] = old_value + epsilon
        fxh1 = f().item()  # evaluate f(param + epsilon)
        
        flat_param[ix] = old_value - epsilon
        fxh2 = f().item()  # evaluate f(param - epsilon)
        
        grad.view(-1)[ix] = (fxh1 - fxh2) / (2 * epsilon)
        flat_param[ix] = old_value  # restore original value
    
    return grad

def gradient_check(model, x, y, mask1, mask2, device, epsilon=1e-5):
    model.train()  # Ensure the model is in training mode for debugging
    
    x.requires_grad = True
    y.requires_grad = True
    mask1.requires_grad = True
    mask2.requires_grad = True

    y_pred, params = model(x, y, mask1, mask2)
    loss = correlation_coefficient_loss_params(ground_truth, params)
    loss.backward()

    print("Gradient for y_pred:", y_pred.grad)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"No gradients computed for {name}")
                continue
            
            def f():
                model.zero_grad()
                y_pred, params = model(x, y, mask1, mask2)
                loss = correlation_coefficient_loss_params(ground_truth, params)
                return loss
            
            # Compute numerical gradients
            numerical_grad = numerical_gradient(f, param.data, epsilon)

            # Ensure analytical_grad is not None
            analytical_grad = param.grad
            if analytical_grad is None:
                print(f"Analytical gradient is None for {name}")
                continue

            # Compare gradients
            diff = torch.norm(analytical_grad - numerical_grad) / torch.norm(analytical_grad + numerical_grad)
            print(f'Gradient check for {name}: {diff:.6f}')
            
            if diff > epsilon:
                print(f'WARNING: Gradients for {name} might be incorrect!')
        else:
            print(f'Parameter {name} does not require gradients')  # Debugging print

# Run gradient check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GumNet().to(device)
x_test = torch.rand(16, 1, 32, 32, 32).to(device)
ground_truth = torch.rand(16, 6).to(device)
y_test = torch.rand(16, 1, 32, 32, 32).to(device)
observed_mask, missing_mask = generate_masks(x_test, tilt_angle=30)
gradient_check(model, x_test, y_test, observed_mask, missing_mask, device)
