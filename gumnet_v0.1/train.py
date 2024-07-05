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

def get_transformation_output_from_model(model, x_test, y_test, batch_size=32):
    model.eval()
    with torch.no_grad():
        y_preds = []
        params_list = []

        # Run 1 batch at a time through the model
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            params = model(x_batch, y_batch)
            params_list.append(params)
            
    params = torch.cat(params_list, dim=0)
    return params.detach().cpu().numpy()

def create_dataloaders(x, y, gt, device, batch_size):
    # Creates a shuffled dataloader for input, target, and ground truth parameters
    dataloader = DataLoader(TensorDataset(x.to(device), y.to(device), gt.to(device)), batch_size=batch_size, shuffle=True)
    return dataloader

def main(batch_size=32, initial_lr=1e-7, dataset="Gro"):
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data is loaded and preprocessed in process_data.py
    if dataset == "Gro":
        train_x, train_y, valid_x, valid_y, test_x, test_y, ground_truth = load_GroEL_ES()
    elif dataset == "Qiang":
        train_x, train_y, valid_x, valid_y, test_x, test_y, ground_truth = load_rat_data()
    else:
        RuntimeError("Not implemented")
    
    # Noramlizes the data for improved training and testing
    test_x = (test_x - test_x.mean()) / test_x.std()
    test_y = (test_y - test_y.mean()) / test_y.std() 

    # @TODO generate ground truth for the train dataset. For now, using test dataset.
    dataloader = create_dataloaders(test_x, test_y, ground_truth, device, batch_size)
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
    params = get_transformation_output_from_model(model, test_x, test_y)
    print('Evaluation (no fine-tuning):')
    alignment_eval_no_scale(ground_truth, params, test_x.shape[2])

    # Train model with the test dataset for 20 iterations
    # @TODO switch to train data when the ground truth is generated
    # Uses the same scheduler and optimizer as the GumNet demo data
    model.train()
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        epoch_loss = 0
        for x_batch, y_batch, gt_batch in dataloader:
            optimizer.zero_grad()
            params = model(x_batch, y_batch)

            # training with the ground truth and the parameters directly
            loss = correlation_coefficient_loss_params(gt_batch, params)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch {i + 1} complete. Average Loss: {epoch_loss / len(dataloader)}. Current LR: {current_lr}')

    # After training, test model again with the test dataset
    torch.cuda.empty_cache()
    params = get_transformation_output_from_model(model, test_x, test_y)
    print('After finetuning:')
    alignment_eval_no_scale(ground_truth, params, test_x.shape[2])

if __name__ == '__main__':
    datasets = ["Gro", "Qiang"]
    main(batch_size=32, initial_lr=1e-4, dataset=datasets[1])
