import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from models.gumnet import GumNet
from utils import *
from process_data import load_GroEL_ES, load_rat_data

torch.autograd.set_detect_anomaly(True)

def get_transformation_output_from_model(model, x_test, y_test, batch_size=32, device='cuda:0'):
    model.eval()
    with torch.no_grad():
        params_list = []

        # Run 1 batch at a time through the model
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)

            # Ensure that x_batch and y_batch have the same batch size
            if x_batch.shape[0] != y_batch.shape[0]:
                min_batch_size = min(x_batch.shape[0], y_batch.shape[0])
                x_batch = x_batch[:min_batch_size]
                y_batch = y_batch[:min_batch_size]

            transformed, params = model(x_batch, y_batch)
            params_list.append(params.cpu())
            
    params = torch.cat(params_list, dim=0)
    return params.detach().numpy()

def create_dataloaders(x, y, gt, batch_size):
    # Creates a shuffled dataloader for input, target, and ground truth parameters
    dataloader = DataLoader(TensorDataset(x, y, gt), batch_size=batch_size, shuffle=True)
    return dataloader

def normalize_tensor(x):
    return (x - x.mean()) / x.std()

def main(batch_size=2, initial_lr=1e-4, dataset="Gro"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data is loaded and preprocessed in process_data.py
    if dataset == "Gro":
        train_x, train_y, _, _, test_x, test_y, ground_truth, gt_train = load_GroEL_ES()
    elif dataset == "Qiang":
        train_x, train_y, _, _, test_x, test_y, ground_truth, gt_train = load_rat_data()
    else:
        RuntimeError("Not implemented")
    
    # Normalize the data for improved training and testing
    test_x = normalize_tensor(test_x)
    test_y = normalize_tensor(test_y) 
    train_x = normalize_tensor(train_x)
    train_y = normalize_tensor(train_y)

    train_x, train_y = augment_tensors(train_x, train_y, train_x.device)
    test_x, test_y = augment_tensors(test_x, test_y, test_x.device)
    dataloader = create_dataloaders(train_x, train_y, gt_train, batch_size)
    print('Data successfully loaded!')

    # Initialize the model on the GPU
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    model = GumNet().to(device)
    print('Gum-Net model initialized!')

    # Hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
    for param in model.parameters():
        param.requires_grad = True

    # Test the model with the 'test' dataset
    params = get_transformation_output_from_model(model, test_x, test_y, batch_size=batch_size, device=device)
    print('Evaluation (no fine-tuning):')
    alignment_eval_no_scale(ground_truth, params, test_x.shape[2])

    # Train model with the train dataset for 20 iterations
    model.train()
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        epoch_loss = 0
        for x_batch, y_batch, gt_batch in dataloader:
            x_batch, y_batch, gt_batch = x_batch.to(device), y_batch.to(device), gt_batch.to(device)
            optimizer.zero_grad()
            transformed, params = model(x_batch, y_batch)
            loss = correlation_coefficient_loss(transformed, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch {i + 1} complete. Average Loss: {epoch_loss / len(dataloader)}. Current LR: {current_lr}')

    # After training, test model again with the test dataset
    torch.cuda.empty_cache()
    params = get_transformation_output_from_model(model, test_x, test_y, batch_size=batch_size, device=device)
    print('After finetuning:')
    alignment_eval_no_scale(ground_truth, params, test_x.shape[2])

if __name__ == '__main__':
    datasets = ["Gro", "Qiang"]
    main(batch_size=2, initial_lr=5e-5, dataset=datasets[0])
