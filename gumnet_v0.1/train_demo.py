import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from models.gumnet import GumNet
from utils import *

def get_transformation_output_from_model(model, x_test, y_test, device, batch_size=32):
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
        params_list = []
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            params = model(x_batch, y_batch)
            params_list.append(params)
            
    params = torch.cat(params_list, dim=0)
    return params.detach().cpu().numpy()

def create_dataloaders(x_test, y_test, ground_truth, batch_size):

    dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32),
                            torch.tensor(ground_truth, dtype=torch.float32))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def main(DEBUG=False, batch_size=32, initial_lr=1e-4, AUGMENT=False, random_initialization=True):
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the train data
    with open('../dataset/gum_demo_data.pickle', 'rb') as f:
        x_test, y_test, _, _, ground_truth = pickle.load(f, encoding='latin1')
    
    # 2. Preprocess data (augmentation, normalization, batches) for training
    if augment_data:
        x_test, y_test = augment_data(x_test, y_test)
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)
    dataloader = create_dataloaders(x_test, y_test, ground_truth, batch_size)
    print('Data successfully loaded!')

    # 3. Initialize model and weights
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    model = GumNet().to(device)
    # intialize weights with a custom function in utils if random_initialization=False 
    if not random_initialization:
        model.apply(initialize_weights)
    print('Gum-Net model initialized!')

    # 4. Initialize optimizer, scheduler, and gradients
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
    for param in model.parameters():
        param.requires_grad = True

    # 5. Evaluate the model before fine-tuning
    params = get_transformation_output_from_model(model, x_test, y_test, device, batch_size)
    print('Evaluation (no fine-tuning):')
    alignment_eval(ground_truth, params, x_test.shape[2])
    print(compute_transformation_accuracy(torch.tensor(ground_truth, dtype=torch.float32).to(device), 
                                          torch.tensor(params, dtype=torch.float32).to(device)))

    # 6. Fine-tune the model for 20 iterations
    model.train()
    loss_fn = CombinedLoss(alpha=0)
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        epoch_loss = 0
        for x_batch, y_batch, gt_batch in dataloader:
            x_batch = x_batch.permute(0, 4, 1, 2, 3).to(device)
            y_batch = y_batch.permute(0, 4, 1, 2, 3).to(device)
            gt_batch = gt_batch.to(device)

            optimizer.zero_grad()
            params = model(x_batch, y_batch)

            # calculate loss directly with 6D transformation parameters
            loss = loss_fn(params, gt_batch)

            # Check for NaNs in loss
            if torch.isnan(loss).any():
                print("NaNs in loss detected!")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    params = get_transformation_output_from_model(model, x_test, y_test, device, batch_size)
    print('After finetuning:')
    alignment_eval(ground_truth, params, x_test.shape[2])
    print(compute_transformation_accuracy(torch.tensor(ground_truth, dtype=torch.float32).to(device), 
                                          torch.tensor(params, dtype=torch.float32).to(device)))

if __name__ == '__main__':
    main(DEBUG=False, batch_size=32, initial_lr=1e-4, AUGMENT=False, random_initialization=True)
