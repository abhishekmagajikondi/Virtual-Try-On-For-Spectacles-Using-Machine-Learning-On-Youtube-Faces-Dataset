import torch
# constant paths
ROOT_PATH = ''
OUTPUT_PATH = 'OUTPUT'
# learning parameters
BATCH_SIZE = 10
LR = 0.001
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.1
# show dataset keypoint plot
SHOW_DATASET_PLOT = False