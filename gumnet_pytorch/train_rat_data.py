import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from models.gumnet_v2 import GumNet
from utils import *
from process_data import load_rat_data

torch.autograd.set_detect_anomaly(True)

def get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask, device, batch_size=32):
    model.eval()
    with torch.no_grad():
        y_preds = []
        params_list = []

        # Run 1 batch at a time through the model
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            mask1_batch = observed_mask[i:i+batch_size]
            mask2_batch = missing_mask[i:i+batch_size]
            y_pred, params = model(x_batch, y_batch, mask1_batch, mask2_batch)
            y_preds.append(y_pred)
            params_list.append(params)
            
    y_pred = torch.cat(y_preds, dim=0)
    params = torch.cat(params_list, dim=0)
    # y_pred is the transormed subtomagram and params are the predicted transformation params
    return y_pred, params.detach().cpu().numpy()

def create_dataloaders(x, y, m1, m2, batch_size):
    # Creates a shuffled dataloader for input, target, and masks
    dataloader = DataLoader(TensorDataset(x, y, m1, m2), batch_size=batch_size, shuffle=True)
    return dataloader

def main(DEBUG=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data is loaded and preprocessed in load_rat_data function
    train_x, train_y, valid_x, valid_y, test_x, test_y, observed_mask, missing_mask, ground_truth = load_rat_data()
    
    # Noramlizes the data for improved training and testing
    train_x = (train_x - train_x.mean()) / train_x.std()
    train_y = (train_y - train_y.mean()) / train_y.std() 
    test_x = (test_x - test_x.mean()) / test_x.std()
    test_y = (test_y - test_y.mean()) / test_y.std() 

    # We use the dataloader in the training loop so use train dataset
    dataloader = create_dataloaders(train_x, train_y, observed_mask, missing_mask, batch_size=16)
    print('Data successfully loaded!')

    # Initialize the model on the GPU
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    model = GumNet().to(device)
    print('Gum-Net model initialized!')

    # Hyperparameters
    initial_lr = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
    for param in model.parameters():
        param.requires_grad = True

    # Generate masks with a tilt angle of 30 degrees
    m1, m2 = generate_masks(test_x, tilt_angle=60)

    # Test the model with the 'test' dataset
    y_pred, params = get_transformation_output_from_model(model, test_x, test_y, m1, m2, device)
    print('Evaluation (no fine-tuning):')
    alignment_eval(ground_truth, params, test_x.shape[2])

    # Train model with the train dataset for 20 iterations
    # Uses the same scheduler and optimizer as the GumNet demo data
    model.train()
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        epoch_loss = 0
        for x_batch, y_batch, mask_1_batch, mask_2_batch in dataloader:

            optimizer.zero_grad()
            y_pred, params = model(x_batch, y_batch, mask_1_batch, mask_2_batch)
            loss = correlation_coefficient_loss(y_batch, y_pred)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if DEBUG:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradients for {name}: {param.grad.norm().item()}")
                    else:
                        print(f"No gradients for {name}")
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch {i + 1} complete. Average Loss: {epoch_loss / len(dataloader)}. Current LR: {current_lr}')

    # After training, test model again with the test dataset
    y_pred, params = get_transformation_output_from_model(model, test_x, test_y, m1, m2, device)
    print('After finetuning:')
    alignment_eval(ground_truth, params, test_x.shape[2])

if __name__ == '__main__':
    main(DEBUG=False) # Set DEBUG=True to see gradients printed to terminal during training
