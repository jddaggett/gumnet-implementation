# train demo code for compatibility with pytorch gumnet_v2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from models.gumnet_v2 import GumNet
from utils import *

def load_data(data_path):
    with open(data_path, 'rb') as f:
        x_test, y_test, observed_mask, missing_mask, ground_truth = pickle.load(f, encoding='latin1')
    return x_test, y_test, observed_mask, missing_mask, ground_truth

def normalize_data(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return x, y

def create_dataloaders(x_test, y_test, observed_mask, missing_mask, batch_size):
    mask_1 = np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1))
    mask_2 = np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1))

    dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), 
                            torch.tensor(y_test, dtype=torch.float32), 
                            torch.tensor(mask_1, dtype=torch.float32), 
                            torch.tensor(mask_2, dtype=torch.float32))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask):
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        y_test = torch.tensor(y_test, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        observed_mask = torch.tensor(observed_mask, dtype=torch.float32)
        missing_mask = torch.tensor(missing_mask, dtype=torch.float32)
        output, params = model(x_test, y_test)
    return output, params.detach().numpy()


def main(opt):
    # 1. Load and preprocess train data
    x_test, y_test, observed_mask, missing_mask, ground_truth = load_data(opt.data_path)
    x_test, y_test = normalize_data(x_test, y_test)
    dataloader = create_dataloaders(x_test, y_test, observed_mask, missing_mask, batch_size=32)
    print('Data successfully loaded!')

    # 2. Initialize model and optimizer
    model = GumNet()
    print('GumNet model initialized!')
    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr)
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
    # 3. Evaluate before fine-tuning
    transformation_output, y_pred = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask)
    print('Output shape:', transformation_output.shape)
    print('Before finetuning:')
    alignment_eval(ground_truth, y_pred , x_test[0].shape[0])

    for epoch in range(20):
        print(f'Training Iteration {epoch + 1}')
        optimizer.param_groups[0]['lr'] = opt.initial_lr * 0.9 ** epoch

        for x_batch, y_batch, mask_1_batch, mask_2_batch in dataloader:
            optimizer.zero_grad()
            x_batch = torch.tensor(x_batch, dtype=torch.float32).permute(0, 4, 1, 2, 3)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).permute(0, 4, 1, 2, 3)
            mask_1_batch = torch.tensor(mask_1_batch, dtype=torch.float32)
            mask_2_batch = torch.tensor(mask_2_batch, dtype=torch.float32)

            x_batch.requires_grad = True
            y_batch.requires_grad = True

            output, _ = model(x_batch, y_batch)
            
            if isinstance(output, list):
                output = output[0]

            loss = correlation_coefficient_loss(y_batch, output)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} complete. Loss: {loss.item()}')

    # 4. Evaluate after fine-tuning
    transformation_output, y_pred = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask)
    print('After finetuning:')
    alignment_eval(ground_truth, y_pred, x_test[0].shape[0])

if __name__ == '__main__':
    class Opt:
        def __init__(self):
            self.data_path = '../dataset/gum_demo_data.pickle'
            self.initial_lr = 0.001

    opt = Opt()
    main(opt)