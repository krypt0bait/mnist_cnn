import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import CNN 


device = torch.device("cpu")
model = CNN()
model.load_state_dict(torch.load('mnist_model.pth', map_location=device)) # Load your 99% accuracy model
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28), interpolation=cv2.INTER_AREA), # Sharp resizing
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


hsv_value = np.load('hsv_value.npy') 
lower_range, upper_range = hsv_value[0], hsv_value[1]
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w), dtype=np.uint8)
x1, y1 = 0, 0
prediction = "None"
noise_thresh = 5000 

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    key = cv2.waitKey(1) & 0xFF
    
    
    if key == 32: 
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > noise_thresh:
                x2, y2, w, h = cv2.boundingRect(c)
                cx, cy = x2 + w//2, y2 + h//2
                if x1 != 0 and y1 != 0:
                    cv2.line(canvas, (x1, y1), (cx, cy), 255, 30) 
                x1, y1 = cx, cy
    else:
        x1, y1 = 0, 0 

    
    if key == ord('p'):
        coords = cv2.findNonZero(canvas)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            roi = canvas[y:y+h, x:x+w]
            
            roi = cv2.copyMakeBorder(roi, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)
            
            
            preview = cv2.resize(roi, (280, 280), interpolation=cv2.INTER_AREA)
            cv2.imshow('Model Input Preview', preview)
            
            img_tensor = transform(roi).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                prediction = str(torch.max(output, 1)[1].item())

    
    frame_display = frame.copy()
    frame_display[canvas > 0] = [0, 255, 0] 
    cv2.putText(frame_display, f"Prediction: {prediction}", (50, 100), 1, 3, (0, 255, 0), 3)
    cv2.imshow('Task 18', frame_display)

    if key == ord('c'): canvas = np.zeros_like(canvas) 
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()