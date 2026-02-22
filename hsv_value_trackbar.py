import cv2
import numpy as np

# 1. Setup Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

def nothing(x):
    pass

# 2. Create Trackbar Window
cv2.namedWindow('Trackbars')
# Default HSV ranges: Hue (0-179), Saturation (0-255), Value (0-255)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

print("--- CALIBRATION STEPS ---")
print("1. Hold your orangish object in front of the webcam.")
print("2. Adjust 'L' (Lower) and 'U' (Upper) sliders until your object is WHITE and background is BLACK.")
print("3. Press 's' to save the hsv_value.npy file and exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    # OpenCV reads camera frames as BGR; we convert to HSV for better color tracking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Get Current Trackbar Positions
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # 4. Create Mask
    # This turns pixels within the range white (255) and everything else black (0)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Show the 'Result' window so you can see if the color is isolated
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Stack windows for easier viewing: Mask | Original | Result
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask_3, frame, res))
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))
    
    key = cv2.waitKey(1) & 0xFF
    
    # Press 's' to save and move to live_writing.py
    if key == ord('s'):
        thearray = np.array([lower_range, upper_range])
        np.save('hsv_value.npy', thearray)
        print(f"Values saved successfully: \n{thearray}")
        break
        
    if key == 27: # Press 'Esc' to quit without saving
        break

cap.release()
cv2.destroyAllWindows()