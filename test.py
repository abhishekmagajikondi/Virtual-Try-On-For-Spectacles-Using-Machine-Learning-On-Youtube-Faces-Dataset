import torch
import numpy as np
import cv2
import config
from model import FaceKeypointResNet50

model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Read the input image
image = cv2.imread('1 (1).jpg')  # Replace 'path_to_your_image.jpg' with the actual path to your image

with torch.no_grad():
    image = cv2.resize(image, (224, 224))
    orig_frame = image.copy()
    orig_h, orig_w, c = orig_frame.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0).to(config.DEVICE)
    outputs = model(image)

outputs = outputs.cpu().detach().numpy()
outputs = outputs.reshape(-1, 2)
keypoints = outputs

for p in range(keypoints.shape[0]):
    cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                1, (0, 0, 255), -1, cv2.LINE_AA)

orig_frame = cv2.resize(orig_frame, (orig_w, orig_h))

cv2.imshow('Facial Keypoint Image', orig_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
