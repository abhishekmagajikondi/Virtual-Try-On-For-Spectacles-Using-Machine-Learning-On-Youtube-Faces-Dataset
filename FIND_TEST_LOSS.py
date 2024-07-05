import os

# Set the environment variable to avoid OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
import pandas as pd
from model import FaceKeypointResNet50
from dataset_test import test_data, test_loader
from tqdm import tqdm
matplotlib.style.use('ggplot')


# model 
model = FaceKeypointResNet50(pretrained=True, requires_grad=True).to(config.DEVICE)
# optimizer
checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.SmoothL1Loss()



def evaluate(model, dataloader, data):
    model.eval()
    running_loss = 0.0
    counter = 0
    predictions = []
    true_values = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            
            # Flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            running_loss += loss.item()

            # Convert outputs and keypoints to numpy arrays
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(keypoints.cpu().numpy())

    loss = running_loss / counter
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Compute metrics (e.g., MSE, MAE)
    mse = np.mean((predictions - true_values) ** 2)
    mae = np.mean(np.abs(predictions - true_values))
    
    # Add more metrics as needed

    return loss, mse, mae

# Evaluate the model
TESTING_loss, TESTING_mse, TESTING_mae = evaluate(model, test_loader, test_data)

print(f"Test Loss: {TESTING_loss:.4f}")
print(f"Test MSE Loss: {TESTING_mse:.4f}")
print(f"Test MAE Loss: {TESTING_mae:.4f}")
