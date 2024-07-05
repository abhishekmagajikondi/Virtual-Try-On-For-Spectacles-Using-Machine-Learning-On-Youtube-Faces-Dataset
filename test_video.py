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

# capture the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print('Error while trying to open webcam. Please check again...')

while cap.isOpened():
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        with torch.no_grad():
            # preprocess the frame
            image = cv2.resize(frame, (224, 224))
            orig_frame = image.copy()
            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            # forward pass through the model
            outputs = model(image)
        
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1, 2)
        keypoints = outputs

        # # visualize keypoints on the frame
        # for p in range(keypoints.shape[0]):
        #     #cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
        #                 #1, (0, 0, 255), -1, cv2.LINE_AA)

        orig_frame = cv2.resize(orig_frame, (orig_w, orig_h))
        
        # top-left location for sunglasses to go
        # 17 = edge of left eyebrow
        x = int(keypoints[17, 0])
        y = int(keypoints[17, 1])

        # height and width of sunglasses
        # h = length of nose
        h = int(abs(keypoints[27,1] - keypoints[33,1]))
        # w = left to right eyebrow edges
        w = int(abs(keypoints[17,0] - keypoints[26,0]))
        
        # read in sunglasses
        sunglasses = cv2.imread('test3.png', cv2.IMREAD_UNCHANGED)

        # resize sunglassesC:\ML\FINAL_ML\spectacle2.png
        new_sunglasses =  cv2.resize(sunglasses, (w, h), interpolation = cv2.INTER_CUBIC)

        # get region of interest on the face to change
        roi_color = orig_frame[y:y+h,x:x+w]

        # find all non-transparent pts
        ind = np.argwhere(new_sunglasses[:,:,3] > 0)
        
        # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
        for i in range(3):
            roi_color[ind[:,0],ind[:,1],i] = new_sunglasses[ind[:,0],ind[:,1],i]    
        # set the area of the image to the changed region with sunglasses
        orig_frame[y:y+h,x:x+w] = roi_color
        
        orig_frame = cv2.resize(orig_frame, (512, 512))
        cv2.imshow('Facial Keypoint Detection', orig_frame)

        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()
