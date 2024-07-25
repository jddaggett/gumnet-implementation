import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.gumnet_v2 import GumNet
from models.gumnet_v0_1 import GumNetNoSTN
from models.gumnet_v0_2 import GumNetSimpleSTN
from utils import *
from process_data import load_GroEL_ES, load_rat_data

torch.autograd.set_detect_anomaly(True)
AVAILABLE_DEVICES = 8 # 8 GPUs available on the current machine

def get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask, stn, batch_size=32):
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
            if stn == "None":
                params = model(x_batch, y_batch)
                y_pred = None
            elif stn == "Simple":
                y_pred, params = model(x_batch, y_batch)
            elif stn == "Full":
                y_pred, params = model(x_batch, y_batch, mask1_batch, mask2_batch)
            y_preds.append(y_pred)
            params_list.append(params)
    
    if y_pred is not None:
        y_pred = torch.cat(y_preds, dim=0)
    params = torch.cat(params_list, dim=0)

    # y_pred is the transormed subtomagram and params are the predicted transformation params
    return y_pred, params.detach().cpu().numpy()

def create_dataloaders(x, y, m1, m2, gt, batch_size):
    # Creates a shuffled dataloader for input, target, and masks
    dataloader = DataLoader(TensorDataset(x, y, m1, m2, gt), batch_size=batch_size, shuffle=True)
    return dataloader

def train(device, DEBUG=False, batch_size=32, initial_lr=1e-4, dataset="Gro", stn="None"):
    print(f"Using device: {device}")

    # Data is loaded and preprocessed in process_data.py
    if dataset == "Gro":
        train_x, train_y, _, _, test_x, test_y, observed_mask, missing_mask, ground_truth, gt_train = load_GroEL_ES(device)
    elif dataset == "Qiang":
        train_x, train_y, _, _, test_x, test_y, observed_mask, missing_mask, ground_truth, gt_train = load_rat_data(device)
    else:
        RuntimeError("Not implemented")
    
    # Noramlizes the data for improved training and testing
    train_x = (train_x - train_x.mean()) / train_x.std()
    train_y = (train_y - train_y.mean()) / train_y.std() 
    test_x = (test_x - test_x.mean()) / test_x.std()
    test_y = (test_y - test_y.mean()) / test_y.std() 

    # We use the dataloader in the training loop so use train dataset
    dataloader = create_dataloaders(train_x, train_y, observed_mask, missing_mask, gt_train, batch_size)
    print('Data successfully loaded!')

    # Initialize the specified model on the GPU
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    if stn == "None":
        model = GumNetNoSTN().to(device)
    elif stn == "Simple":
        model = GumNetSimpleSTN().to(device)
    elif stn == "Full":
        model = GumNet().to(device)
    else:
        print("Unrecognized STN name. No models found.")
    print('Gum-Net model initialized!')

    # Set optimizer, sceduler and set up parameter gradient requirements
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
    for param in model.parameters():
        param.requires_grad = True
    
    # Generate masks for imputation, @NOTE change tilt_angle according to dataset specification
    m1, m2 = generate_masks(test_x, tilt_angle=25)

    # Test the model with the 'test' dataset
    y_pred, params = get_transformation_output_from_model(model, test_x, test_y, m1, m2, stn)
    print('Evaluation (no fine-tuning):')
    alignment_eval(ground_truth, params, test_x.shape[2])

    # Train model with the train dataset for 20 iterations
    # Uses the same scheduler and optimizer as the GumNet demo data
    model.train()
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        epoch_loss = 0
        for x_batch, y_batch, mask_1_batch, mask_2_batch, gt_batch in dataloader:

            # Run the correct model based on which transformer network we want to test
            optimizer.zero_grad()
            if stn == "None":
                params = model(x_batch, y_batch)
                loss = correlation_coefficient_loss_params(params, gt_batch)
            elif stn == "Simple":
                y_pred, params = model(x_batch, y_batch)
                loss = correlation_coefficient_loss(y_batch, y_pred)
            elif stn == "Full":
                y_pred, params = model(x_batch, y_batch, mask_1_batch, mask_2_batch)
                loss = correlation_coefficient_loss(y_batch, y_pred)

            # Perform backward gradient descent and gradient clipping to prevent exploding gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Prints weights and their gradients during computation if DEBUG set to True
            if DEBUG:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradients for {name}: {param.grad.norm().item()}")
                    else:
                        print(f"No gradients for {name}")
            
            # Step the optimizer and add to the loss for the current epoch
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch {i + 1} complete. Average Loss: {epoch_loss / len(dataloader)}. Current LR: {current_lr}')

    # After training, test model again with the test dataset
    y_pred, params = get_transformation_output_from_model(model, test_x, test_y, m1, m2, stn)
    print('After finetuning:')
    alignment_eval(ground_truth, params, test_x.shape[2])

def main():
    datasets = ["Gro", "Qiang"] # Add more datasets if necessary 
    transformers = ["None", "Simple", "Full"] # gum_v0.1, gum_v0.2, gum_v2

    # Use CPU exclusively if there are no available GPUs
    if AVAILABLE_DEVICES < 1: 
        train(torch.device('cpu'), False, batch_size=32, initial_lr=1e-4, dataset=datasets[0], stn=transformers[0])
    
    # Handle memory errors gracefully
    for i in range(AVAILABLE_DEVICES):
        try:
            device = torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu')
            train(device, False, batch_size=2, initial_lr=1e-4, dataset=datasets[0], stn=transformers[1])
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory. Clearing cache. Devices left to try: {7-i}")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            if 'cuFFT error' in str(e):
                print(f"cuFFT error: {e}. Clearing cache. Devices left to try: {7-i}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

if __name__ == '__main__':
    main()