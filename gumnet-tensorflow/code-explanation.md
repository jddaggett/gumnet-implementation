# Code Explanation

## Code Tree

```
|-- models/
|   |-- layers/
|   |   |-- FeatureCorrelation.py
|   |   |-- FeatureL2Norm.py
|   |   |-- RigidTransformation3DImputation.py
|   |   |-- SpectralPooling.py
|   |-- Gum_net.py
|-- opt.py
|-- train_demo.py
|-- utils.py
```


- models
    - Gum_Net.py:
Defines the Gum-Net model architecture using TensorFlow and Keras. The model involves multiple custom layers for feature extraction, normalization, and transformation estimation.
    - layers
        - FeatureCorrelation.py: A layer that computes the correlation between feature maps of two different inputs.
        - FeatureL2Norm.py: Applies L2 normalization to the feature maps.
        - RigidTransformation3DImputation.py: Estimates and applies a rigid 3D transformation based on input parameters.
        - SpectralPooling.py: Implements spectral pooling by using Discrete Cosine Transform (DCT) to reduce the spatial resolution of feature maps while retaining important information.

- train_demo.py:
This script demonstrates how to train the Gum-Net model. It includes loading data, initializing or loading a model, normalizing data, and running a training loop to fine-tune the model with evaluation before and after training.
- opt.py:
Contains the script for parsing command-line options for the demo, including paths to data and model, and settings like whether to build a new model or use a pretrained one, and the initial learning rate.

- utils.py:
Provides utility functions like calculating initial weights for the network, defining a custom loss function based on correlation coefficients, and evaluating alignment errors.
