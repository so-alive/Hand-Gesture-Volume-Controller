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
    hands, img = detector.find