import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from models.gumnet_v2 import GumNet
from utils import *

def get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask, device, batch_size=32):
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
        mask_1 = np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
        mask_2 = np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
        m1 = torch.tensor(mask_1, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
        m2 = torch.tensor(mask_2, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
        y_preds = []
        params_list = []
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            mask1_batch = m1[i:i+batch_size]
            mask2_batch = m2[i:i+batch_size]
            y_pred, params = model(x_batch, y_batch, mask1_batch, mask2_batch)
            y_preds.append(y_pred)
            params_list.append(params)
            
    y_pred = torch.cat(y_preds, dim=0)
    params = torch.cat(params_list, dim=0)
    return y_pred, params.detach().cpu().numpy()

def create_dataloaders(x_test, y_test, observed_mask, missing_mask, ground_truth, batch_size):
    mask_1 = np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
    mask_2 = np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1))

    dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32),
                            torch.tensor(mask_1, dtype=torch.float32), 
                            torch.tensor(mask_2, dtype=torch.float32),
                            torch.tensor(ground_truth, dtype=torch.float32))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def main(DEBUG=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the train data
    with open('../dataset/gum_demo_data.pickle', 'rb') as f:
        x_test, y_test, observed_mask, missing_mask, ground_truth = pickle.load(f, encoding='latin1')
    
    # 2. Augment and normalize data then create dataloader for batches
    x_test, y_test = augment_data(x_test, y_test)
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)
    dataloader = create_dataloaders(x_test, y_test, observed_mask, missing_mask, ground_truth, batch_size=32)
    print('Data successfully loaded!')

    # 3. Initialize model and weights
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    model = GumNet().to(device)
    # model.apply(initialize_weights)
    print('Gum-Net model initialized!')

    # 4. Initialize hyperparameters and optimizer
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
    for param in model.parameters():
        param.requires_grad = True

    # 5. Evaluate the model before fine-tuning
    y_pred, params = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask, device)
    print('Evaluation (no fine-tuning):')
    alignment_eval(ground_truth, params, x_test.shape[2])

    # 6. Fine-tune the model for 20 iterations
    model.train()
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        epoch_loss = 0
        for x_batch, y_batch, mask_1_batch, mask_2_batch, gt_batch in dataloader:
            x_batch = x_batch.permute(0, 4, 1, 2, 3).to(device)
            y_batch = y_batch.permute(0, 4, 1, 2, 3).to(device)
            mask_1_batch = mask_1_batch.permute(0, 4, 1, 2, 3).to(device)
            mask_2_batch = mask_2_batch.permute(0, 4, 1, 2, 3).to(device)
            gt_batch = gt_batch.to(device)

            optimizer.zero_grad()
            y_pred, params = model(x_batch, y_batch, mask_1_batch, mask_2_batch)
            loss = correlation_coefficient_loss_params(gt_batch, params)

            # Check for NaNs in loss
            if torch.isnan(loss).any():
                print("NaNs in loss detected!")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Avoid exploding gradients

            # Print gradients for debugging
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

    # 7. Evaluate the model after fine-tuning
    torch.cuda.empty_cache()
    y_pred, params = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask, device)
    print('After finetuning:')
    alignment_eval(ground_truth, params, x_test.shape[2])

    # 8. Visualize results
    # x_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
    # visualize_2d_slice(x_tensor.cpu(), y_pred)
    # get_mrc_files(x_tensor, y_pred)

if __name__ == '__main__':
    main(DEBUG=False)  # Set DEBUG=True to print weight gradient values to the terminal at runtime
