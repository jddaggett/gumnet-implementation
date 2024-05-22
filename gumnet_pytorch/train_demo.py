# train demo code for compatibility with pytorch gumnet_v2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from models.gumnet_v2 import GumNet
from utils import *

def get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask):
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        y_test = torch.tensor(y_test, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        mask_1 = np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
        mask_2 = np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
        m1 = torch.tensor(mask_1, dtype=torch.float32)
        m2 = torch.tensor(mask_2, dtype=torch.float32)
        output, params = model(x_test, y_test, m1, m2)
    return output, params.detach().numpy()


def create_dataloaders(x_test, y_test, observed_mask, missing_mask, batch_size):
    mask_1 = np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
    mask_2 = np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1))

    dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), 
                            torch.tensor(y_test, dtype=torch.float32), 
                            torch.tensor(mask_1, dtype=torch.float32), 
                            torch.tensor(mask_2, dtype=torch.float32))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    # 1. Load and preprocess train data
    with open('../dataset/gum_demo_data.pickle', 'rb') as f:
        x_test, y_test, observed_mask, missing_mask, ground_truth = pickle.load(f, encoding='latin1')
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)
    dataloader = create_dataloaders(x_test, y_test, observed_mask, missing_mask, batch_size=32)
    print('Data successfully loaded!')

    # 2. Initialize model
    model = GumNet()
    print('Gum-Net model initialized!')

    # Define optimizer
    initial_lr = float(1e-7)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    for param in model.parameters():
        param.requires_grad = True

    # 3. Evaluate the model before fine-tuning
    transformation_output, y_pred = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask)
    print('Evaluation (no fine-tuning):')
    alignment_eval(ground_truth, y_pred, x_test.shape[2])

    # 4. Fine-tune the model for 20 iterations
    model.train()
    for i in range(20):
        print('Training Iteration ' + str(i+1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * 0.9 ** i

        # Shuffle the data
        indices = np.random.permutation(x_test.shape[0])
        x_test = x_test[indices]
        y_test = y_test[indices]

        epoch_loss = 0
        for x_batch, y_batch, mask_1_batch, mask_2_batch in dataloader:
            x_batch = x_batch.permute(0, 4, 1, 2, 3).requires_grad_()
            y_batch = y_batch.permute(0, 4, 1, 2, 3).requires_grad_()
            mask_1_batch = mask_1_batch.permute(0, 4, 1, 2, 3).requires_grad_()
            mask_2_batch = mask_2_batch.permute(0, 4, 1, 2, 3).requires_grad_()
            
            optimizer.zero_grad()
            output, _ = model(x_batch, y_batch, mask_1_batch, mask_2_batch)
            
            loss = correlation_coefficient_loss(y_batch, output)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {i + 1} complete. Average Loss: {epoch_loss / len(dataloader)}')

    # 5. Evaluate the model after fine-tuning
    transformation_output, y_pred = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask)
    print('After finetuning:')
    alignment_eval(ground_truth, y_pred, x_test.shape[2])

    # 6. Visualize results
    x_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3)
    visualize_2d_slice(x_tensor, transformation_output)
    get_mrc_files(x_tensor, transformation_output)

if __name__ == '__main__':
    main()