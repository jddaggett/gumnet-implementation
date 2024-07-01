import numpy as np
from utils import augment_data

def test_augment_data():
    x_test = np.random.rand(10, 32, 32, 32, 1)
    y_test = np.random.rand(10, 32, 32, 32, 1)
    
    x_aug, y_aug = augment_data(x_test, y_test)
    
    assert x_aug.shape == x_test.shape, "Augmented x_test has incorrect shape"
    assert y_aug.shape == y_test.shape, "Augmented y_test has incorrect shape"

def test_normalization():
    x_test = np.random.rand(10, 32, 32, 32, 1)
    y_test = np.random.rand(10, 32, 32, 32, 1)
    
    x_norm = (x_test - np.mean(x_test)) / np.std(x_test)
    y_norm = (y_test - np.mean(y_test)) / np.std(y_test)
    
    assert np.isclose(np.mean(x_norm), 0, atol=1e-7), "x_test is not properly normalized"
    assert np.isclose(np.std(x_norm), 1, atol=1e-7), "x_test is not properly normalized"
    assert np.isclose(np.mean(y_norm), 0, atol=1e-7), "y_test is not properly normalized"
    assert np.isclose(np.std(y_norm), 1, atol=1e-7), "y_test is not properly normalized"
