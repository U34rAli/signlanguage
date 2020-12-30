import argparse
import glob
import os
import numpy as np
import cv2
import pathlib
from pathlib import Path
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

dataset_folder = "./data"
resultfolder = "abc"

data_dir = pathlib.Path(f'{dataset_folder}/')                 
file_paths = list(data_dir.glob('*/*.jpg'))

mp_hands = mp.solutions.hands


# Initialize MediaPipe Hands.
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

def getRectangle(X, Y, n): 
    Xmax = max(X) 
    Xmin = min(X) 
    Ymax = max(Y) 
    Ymin = min(Y)
    pad = int((Ymax-Ymin)*0.2)
    return (max(Xmin-pad,0), Ymax+pad), (Xmax+pad, max(Ymin-pad, 0))


def mediap(image):
    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
    image_hight, image_width, _ = image.shape

    print(results.multi_handedness)

    if not results.multi_hand_landmarks:
        return None
    annotated_image = cv2.flip(image.copy(), 1)
    for hand_landmarks in results.multi_hand_landmarks:
        X, Y = [], []
        for i in hand_landmarks.landmark:
            X.append(int(i.x*image_width))
            Y.append(int(i.y*image_hight))
        n = len(X)  
        start_point, end_point = getRectangle(X, Y, n)

        crop_img = annotated_image[ end_point[1]: start_point[1], start_point[0]: end_point[0]]
        crop_img = cv2.flip(crop_img, 1)

        print(image.shape)
        print(start_point, end_point)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        return cv2.resize(gray, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)


# files
for file in file_paths:
    path = Path(file)
    head, tail = os.path.split(path)
    path = os.path.join( os.path.split(head)[-1],  tail)
    path = os.path.join(resultfolder,  path)
    
    print(file)    
    segment = mediap(cv2.imread(str(file)))

    # if not os.path.exists(path):
            
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except: # Guard against race condition
            print("folder error")
    try:
        cv2.imwrite(path, segment)
    except:
        print("Error")


