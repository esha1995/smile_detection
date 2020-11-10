from scipy.spatial import distance as dist
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    L = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = L / D
    return mar


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
            mouth = shape[mStart:mEnd]
        mar = smile(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        if mar <= .33 or mar > .38:
            cv2.putText(frame, 'no smile', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'smile detected', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame

imagePath = 'images/smile'
writePath = 'dlib(face_mark_smile_detection_)write'

for i in range(24):
    print("image number: " + str(i) + ": ")
    frame = cv2.imread(imagePath+str(i)+'.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    canvas = detect(gray, frame)
    filename = 'smileDetected' + str(i) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), canvas)

