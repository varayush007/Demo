# Facial Cranium Segmentation using U-Net

## Overview
This project focuses on the task of segmenting the cranium region from facial images using deep learning techniques. Three different hypotheses were implemented and evaluated using the U-Net architecture, a popular convolutional neural network for image segmentation tasks.

## Dataset
The dataset consisted of facial images and corresponding binary masks indicating the cranium region. The images were resized to 256x256 pixels, and the masks were converted to grayscale. For the second and third hypotheses, data augmentation techniques such as random horizontal/vertical flips, random rotations, and color jittering were applied to increase the diversity of the training data.

## Models

### Hypothesis 1: Vanilla U-Net
- Architecture: A standard U-Net architecture with two convolutional layers in the encoder and two convolutional layers in the decoder, along with max-pooling and upsampling layers.
- Optimizer: Adam with a learning rate of 0.001
- Loss function: Binary Cross-Entropy Loss

### Hypothesis 2: U-Net with Data Augmentation and Early Stopping
- Architecture: Similar to the first hypothesis, but with data augmentation (random horizontal/vertical flips, random rotations, and color jittering) and early stopping (patience of 3 epochs) techniques implemented.
- Optimizer: Adam with a learning rate of 0.0001
- Loss function: Binary Cross-Entropy Loss

### Hypothesis 3: U-Net with ResNet-50 Encoder
- Architecture: A U-Net architecture with a pretrained ResNet-50 model as the encoder. The first convolutional layer of ResNet-50 was modified to accept 3-channel inputs. Custom decoder layers were defined for upsampling and reducing the number of channels.
- Optimizer: Adam with a learning rate of 0.0001 (ResNet-50 encoder layers were frozen)
- Loss function: Binary Cross-Entropy Loss

## Results
Among the three hypotheses, the second hypothesis (U-Net with data augmentation and early stopping) achieved the best performance on the validation set, with a validation loss of 0.0420. The third hypothesis (U-Net with ResNet-50 encoder) had a comparable performance, with a validation loss of 0.0422.

## Future Work
Potential improvements could include hyperparameter tuning, model ensemble techniques, advanced data augmentation techniques, attention mechanisms, and post-processing techniques like conditional random fields (CRFs) or active contour models.

## Requirements
- Python
- PyTorch
- NumPy
- OpenCV

## Setup
1. Clone the repository
2. Install the required libraries
3. Prepare the dataset
4. Run the training and evaluation scripts
