# MNIST Handwritten Digit Classifier (CNN)

A convolutional neural network built with PyTorch to classify handwritten digits (0-9) from the MNIST dataset.

## Overview

This project implements a CNN from scratch to classify 28x28 grayscale images of handwritten digits, covering model design, training, and evaluation with accuracy and a confusion matrix.

## Dataset

- **MNIST Digit Classification dataset**
- Grayscale images, 28x28 pixels, 10 classes (digits 0-9)

## Model Architecture

- 3 convolutional blocks (Conv2d → ReLU → MaxPool2d), channel depth 1 → 32 → 64 → 128
- Flatten layer
- 2 fully connected layers (1152 → 128 → 10)

## Training

- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Adam
- **Epochs:** 10
- **Batch size:** 64

## Results

- Test accuracy: **99.36%**
- Confusion matrix shows highly consistent performance across all digits, with the model's few remaining mistakes concentrated around visually similar digits (e.g. 9 vs 4, 9 vs 5)

## Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
```

## Usage

```bash
python train.py
```

Trains the model and saves the best checkpoint to `best_model.pt`.

To load the saved model for inference:

```python
model = CNN()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
```

## Author

Samir B K
GitHub: [github.com/Samir-BK](https://github.com/Samir-BK)
