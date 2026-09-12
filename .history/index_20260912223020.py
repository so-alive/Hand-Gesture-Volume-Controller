import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pycaw.pycaw import AudioUtilities

# 1. System Volume Setup
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# 2. Setup Camera & Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

vol_bar = 400
vol_per = 0

print("Running Gesture Control... Press 'q' to exit")

while True:
    success, img = cap.read()
    if not success:
        break

    # Mirror camera view
    img = cv2.flip(img, 1)

    #Find hands
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        lm_list = hand["lmList"] # List of 21 landmarks points

        # Thumb tip (Landmark 4) and Index tip (Landmark 8)
        x1, y1 = lm_list[4][0], lm_list[4][1]
        x2, y2 = lm_list[8][0], lm_list[8][1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw visuals
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

        # Distance calculation
        length = math.hypot(x2 - x1, y2 - y1)

        # Map distance to volume
        vol = np.interp(length, [30, 200], [min_vol, max_vol])
        vol_bar = np.interp(length, [30, 200], [400, 150])
        vol_pet = np.interp(length, [30, 200], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        if length < 40:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    
    # Draw Volume Indicator Bar
    cv2.rectangle(img, (50,150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Volume Controller", img)

    if cv2.waitKey(1) & 0