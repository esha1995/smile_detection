import imutils
from scipy.spatial import distance as dist
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
smileCounter = list()
lookAwayCounter = list()
framerate = 0


def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    L = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = L / D
    return mar


def detect(gray, frame):
    faces = detector(gray, 0)
    if len(faces) < 1:
        lookAwayCounter.append(1)
    else:
        lookAwayCounter.append(0)
    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(faces)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        mouth = shape[mStart:mEnd]
        mar = smile(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mar <= .23 or mar >= .32 and mar < .47:
            cv2.putText(frame, 'smile detected', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
            smileCounter.append(1)
        elif mar > .47:
            cv2.putText(frame, 'suprised', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            smileCounter.append(0)
        cv2.putText(frame, "MAR: {}".format(mar), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


while True:
    try:
        webcam = int(input("enter camera input (default is 0): "))
        framerate = int(input("framerate of camera (default is 24): "))
        break
    except:
        print("Only numbers")

cap = cv2.VideoCapture(webcam)

while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame, (700, 512))
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, 'Press p to start', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xff == ord('p'):
        break

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()

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

plt.plot(smileCounter, color='red', label='smiles')
plt.plot(lookAwayCounter, color='blue', label='looking away')
plt.yticks(np.arange(0, 2, 1))
plt.xticks(np.arange(0, 100, framerate))
plt.legend()
plt.show()

