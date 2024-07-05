import cv2
import torch
import numpy as np
from model import FaceKeypointResNet50
import config

model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Load the sunglasses image
sunglasses = cv2.imread('test3.png', cv2.IMREAD_UNCHANGED)

# Create a VideoCapture object for the webcam (usually 0 or 1)
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Perform face keypoint detection
    with torch.no_grad():
        resized_frame = cv2.resize(frame, (224, 224))
        orig_h, orig_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) / 255.0
        rgb_frame = np.transpose(rgb_frame, (2, 0, 1))
        rgb_frame = torch.tensor(rgb_frame, dtype=torch.float).unsqueeze(0).to(config.DEVICE)
        outputs = model(rgb_frame)

    # Process the output to get keypoints
    outputs = outputs.cpu().detach().numpy()
    keypoints = outputs.reshape(-1, 2)
    resized_frame = cv2.resize(resized_frame, (orig_w, orig_h))

    # Choose the location for sunglasses
    x = int(keypoints[17, 0])
    y = int(keypoints[17, 1])
    h = int(abs(keypoints[27, 1] - keypoints[33, 1]))
    w = int(abs(keypoints[17, 0] - keypoints[26, 0]))

    # Resize sunglasses
    new_sunglasses = cv2.resize(sunglasses, (w, h), interpolation=cv2.INTER_CUBIC)

    # Get region of interest on the face to change
    roi_color = frame[y:y + h, x:x + w]

    # Find all non-transparent points
    ind = np.argwhere(new_sunglasses[:, :, 3] > 0)

    # Replace the original image pixel with that of the new_sunglasses for each non-transparent point
    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1], i] = new_sunglasses[ind[:, 0], ind[:, 1], i]

    # Set the area of the image to the changed region with sunglasses
    frame[y:y + h, x:x + w] = roi_color

    # Display the result
    cv2.imshow('Facial Keypoint Webcam', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
video_capture.release()
cv2.destroyAllWindows()
