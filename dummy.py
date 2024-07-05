import os
import cv2
import torch
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader

class FaceKeypointTestDataset(Dataset):
    def __init__(self, folder_path, resize_dim):
        self.folder_path = folder_path
        self.resize_dim = resize_dim
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        preprocessed_image = self.preprocess_image(image_path, self.resize_dim)
        return {
            'image': preprocessed_image,
            'image_name': os.path.basename(image_path),  # Store image filename for reference
        }

    def preprocess_image(self, image_path, resize_dim):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image {image_path} not found")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (resize_dim, resize_dim))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float)

# Folder path containing test images
test_images_folder = "C:\\ML\\FINAL_ML\\testing_folder"

resize_dimension = 224  # Desired resize dimension

# Create FaceKeypointTestDataset instance
test_data = FaceKeypointTestDataset(test_images_folder, resize_dimension)

# Creating the test loader
test_loader = DataLoader(test_data,
                         batch_size=config.BATCH_SIZE,
                         shuffle=False)

print(len(test_data))
