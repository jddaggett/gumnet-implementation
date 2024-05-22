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


def main():
    # 1. Load and preprocess train data
    with open('../dataset/gum_demo_data.pickle', 'rb') as f:
        x_test, y_test, observed_mask, missing_mask, ground_truth = pickle.load(f, encoding='latin1')
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)
    print('Data successfully loaded!')

    # 2. Initialize model
    model = GumNet()
    print('Gum-Net model initialized!')

    # 3. Evaluate the model
    transformation_output, y_pred = get_transformation_output_from_model(model, x_test, y_test, observed_mask, missing_mask)
    print('Evaluation (no fine-tuning):')
    alignment_eval(ground_truth, y_pred, x_test.shape[2])
    
    # 4. Visualize results
    x_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3)
    visualize_2d_slice(x_tensor, transformation_output)
    get_mrc_files(x_tensor, transformation_output)

if __name__ == '__main__':
    main()
