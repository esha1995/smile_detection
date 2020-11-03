import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_5_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib files/dlib_face_recognition_resnet_model_v1.dat')

def detect(gray, frame):
    imgDetected = detector(gray, 1)
    for (i, imgDetected) in enumerate(imgDetected):
        shape = sp(gray, imgDetected)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(imgDetected)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(gray, 1.4, 15)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xff == ord('p'):
        break
