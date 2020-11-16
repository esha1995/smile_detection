import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')

def detect(gray, frame):
    faces = detector(gray, 1)
    print("faces: " + str(len(faces)))
    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(faces)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        print("smiles: " + str(len(smiles)))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

imagePath = 'images/smile'
writePath = 'dlib_imagewrite'

for i in range(24):
    print("image number: " + str(i) + ": ")
    cap = cv2.imread(imagePath+str(i)+'.jpg')
    frame = imutils.resize(cap, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    canvas = detect(gray, frame)
    filename = 'smileDetected' + str(i) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), canvas)

