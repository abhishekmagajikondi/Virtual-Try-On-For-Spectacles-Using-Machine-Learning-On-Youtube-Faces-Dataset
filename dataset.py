import torch
import cv2
import pandas as pd
import numpy as np
import config
import utils
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


def train_test_split(csv_path, split):
    df_data = pd.read_csv(csv_path)
    len_data = len(df_data)
    # calculate the validation data sample length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = df_data.iloc[:train_split][:]
    valid_samples = df_data.iloc[-valid_split:][:]
    return training_samples, valid_samples



class FaceKeypointDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 224
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv2.imread(f"{self.path}/{self.data.iloc[index][0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))
        # get the keypoints
        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }
        



# Load your dataset
data = pd.read_csv("training_frames_keypoints.csv")

# Split the dataset into training, testing, and validation sets
training_samples, temp_data = train_test_split("training_frames_keypoints.csv", config.TEST_SPLIT1)
# Save the split datasets to CSV files if needed
training_samples.to_csv("train_data.csv", index=False)
temp_data.to_csv("temp_data.csv", index=False)

valid_samples, test_samples = train_test_split("temp_data.csv", config.TEST_SPLIT2)


valid_samples.to_csv("valid_data.csv", index=False)
test_samples.to_csv("test_data.csv", index=False)

# Continue with your DataLoader setup and training/validation loops using these split datasets


# # get the training and validation data samples
# training_samples, valid_samples = train_test_split("training_frames_keypoints.csv",
#                                                    config.TEST_SPLIT)
# # get the training and validation data samples
# training_samples, valid_samples = train_test_split("training_frames_keypoints.csv",
#                                                    config.TEST_SPLIT)


# initialize the dataset - `FaceKeypointDataset()`
train_data = FaceKeypointDataset(training_samples, "training")

valid_data = FaceKeypointDataset(valid_samples, "training")

test_data = FaceKeypointDataset(test_samples, "training")
                                
# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)


valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

test_loader = DataLoader(test_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

# print(len(training_samples))
# print(len(valid_samples))
# print(len(test_samples))

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")
print(f"Testing sample instances: {len(test_data)}")

# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(test_data)