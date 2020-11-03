import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_5_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib files/dlib_face_recognition_resnet_model_v1.dat')
smileCounter = list()
lookAwayCounter = list()
framerate = 24

def detect(gray, frame):
    faces = detector(gray, 1)
    if len(faces) < 1:
        lookAwayCounter.append(1)
    else:
        lookAwayCounter.append(0)

    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(faces)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.4, 15)

        if len(smiles) > 0:
            smileCounter.append(1)
        else:
            smileCounter.append(0)

        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

    return frame

webcam = int(input("enter camera input: "))
print(webcam)
cap = cv2.VideoCapture(webcam)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (700, 512))
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, 'Press p to start', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xff == ord('p'):
        break

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (700, 512))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    canvas = detect(gray, frame)
    cv2.putText(canvas, 'Press q to stop', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt.plot(smileCounter, color='red', label='smiles')
plt.plot(lookAwayCounter, color='blue', label='looking away')
plt.yticks(np.arange(0, 2, 1))
plt.xticks(np.arange(0, 100, framerate))
plt.legend()
plt.show()


smileChanges = np.where(np.roll(smileCounter, 1) != smileCounter)[0]
lookAwayChanges = np.where(np.roll(lookAwayCounter, 1) != lookAwayCounter)[0]

howManySmiles = int(len(smileChanges) / 2)
howManyLookAway = int(len(lookAwayChanges) / 2)

print("person smiled " + str(howManySmiles) + " times, and looked away " + str(howManyLookAway) + " times")

smilebool = False
for index in range(len(smileChanges[1::2])):
    if not smilebool:
        print(str(index + 1) + " smile started at: " + str(
            smileChanges[index] / framerate) + " seconds, and lasted for: " + str(
            (smileChanges[index + 1] - smileChanges[index]) / framerate) + " seconds.")
        smilebool = True
    else:
        print(str(index + 1) + " smile started at: " + str(
            smileChanges[index + 1] / framerate) + " seconds, and lasted for: " + str(
            (smileChanges[index + 2] - smileChanges[index]) / framerate) + " seconds.")

lookbool = False
for index in range(len(lookAwayChanges[1::2])):
    if not lookbool:
        print("person looked away at: " + str(lookAwayChanges[index] / framerate) + " seconds, and lasted for: " + str(
            (lookAwayChanges[index + 1] - smileChanges[index]) / framerate) + " seconds.")
        lookbool = True
    else:
        print("person looked away at: " + str(
            lookAwayChanges[index + 1] / framerate) + " seconds, and lasted for: " + str(
            (lookAwayChanges[index + 2] - lookAwayChanges[index + 1]) / framerate) + " seconds.")
