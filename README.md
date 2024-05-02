# Facial Landmark Detection using Deep Learning

## Overview
This project focuses on the task of predicting biometry points (landmarks) on facial images using deep learning techniques. Three different deep learning models were explored and evaluated on a dataset of facial images with ground truth landmark coordinates.

## Dataset
The dataset contained facial images and corresponding ground truth landmark coordinates stored in a CSV file. The data was preprocessed by resizing the images to a consistent size, converting grayscale images to tensors, and normalizing the data. The dataset was split into train, validation, and test sets with a 60-20-20 ratio.

## Models

### Hypothesis 1: Simple CNN
- Architecture: Two convolutional layers (32 and 64 filters), followed by max-pooling layers and two fully connected layers (100 and 8 units).
- Optimizer: Adam with a learning rate of 0.001
- Loss function: Mean Squared Error (MSE)

### Hypothesis 2: Deeper CNN with Batch Normalization and Dropout
- Architecture: Four convolutional layers (64, 128, 256, and 512 filters), batch normalization after each convolutional layer, max-pooling layers, and two fully connected layers (1000 and 8 units) with a dropout layer.
- Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.0001 and momentum of 0.9
- Loss function: Mean Squared Error (MSE)

### Hypothesis 3: ResNet-18 with Transfer Learning
- Architecture: ResNet-18 pretrained on ImageNet, with the final fully connected layer replaced with custom layers for landmark detection (100 and 8 units).
- Optimizer: Adam with a learning rate of 0.001
- Loss function: Mean Squared Error (MSE)

## Results
Among the three hypotheses, the second hypothesis (deeper CNN with batch normalization and dropout) achieved the best performance on the test set, with a test loss of 73602.2242. The third hypothesis (ResNet-18 with transfer learning) also performed reasonably well, with a test loss of 104498.4609.

## Future Work
Potential improvements could include data augmentation techniques, hyperparameter tuning, ensemble methods, attention mechanisms, and exploring more advanced architectures like EfficientNets or Vision Transformers.

## Requirements
- Python
- PyTorch
- NumPy
- Matplotlib

## Setup
1. Clone the repository
2. Install the required libraries
3. Prepare the dataset
4. Run the training and evaluation scripts
