import imutils
from scipy.spatial import distance as dist
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
sp2 = dlib.shape_predictor('dlib files/shape_predictor_5_face_landmarks.dat')
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


def detect(frame):
    dets = detector(frame, 0)
    if len(dets) > 1:
        cv2.putText(frame, 'face detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp2(frame, detection))
            images = dlib.get_face_chips(frame, faces)
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            shape = sp(frame, faces)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            mar = smile(mouth)
            if mar <= .22 or mar >= .3 and mar < .47:
                cv2.putText(frame, 'smile detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'smile detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "MAR: {}".format(mar), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame

cap = cv2.VideoCapture(2)


while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    canvas = detect(frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()


"""""
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
"""
