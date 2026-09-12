import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pycaw.pycaw import AudioUtilities

# 1. System Volume Setup
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
vol_range = volume.