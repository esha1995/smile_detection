import copy
from datetime import datetime

import cv2
import dlib
import imutils
import numpy as np
import xlsxwriter
from imutils import face_utils
from scipy.spatial import distance as dist

workbook = xlsxwriter.Workbook('xml files/2nd_iteration_landmarks.xlsx')
worksheet = workbook.add_worksheet()

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
smileCounter = list()
faceCounter = list()
writePath = 'detected_smiles'
counter = 0
smile = False
face = False
imageNumber = 0
neutral = 0

# resizes image with width of 600
def resize(frame):
    frame = imutils.resize(frame, width=600)
    return frame

# returns histogram stretched grayscale image
def preproc(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

# returns image with text on. It needs the frame, the string and where to put it
def textOnImage(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# calculates distance from landmarks
def distance(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    L = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = L / D
    return mar

# returns the distance by using two arguments, where is the faces and the grayscale image
def getMar(faces, gray):
    mar = neutral
    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mar = distance(mouth)
    return mar

# returns the mar value of smile without teeth (20% lower than neutral)
def smileNoTeeth(neutral):
    smileNoTeeth = neutral * 0.80
    return smileNoTeeth

# returns the mar value of smile without teeth (20% higher than neutral)
def smileTeeth(neutral):
    smileTeeth = neutral * 1.14
    return smileTeeth

# function that saves image with timestamp
def saveImage(frame):
    global imageNumber
    imageNumber += 1
    saveImg = copy.copy(frame)
    textOnImage(saveImg, 'timestamp: ' + str(imageNumber), 50, 200)
    filename = 'smileSecond' + str(imageNumber) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), saveImg)

def detect(gray, frame, faces):
    global smile
    global face
    if len(faces) < 1:
        textOnImage(frame, 'no face detected',50,100)
        faceCounter.append(0)
    else:
        textOnImage(frame, 'face detected',50,100)
        faceCounter.append(1)
    if getMar(faces, gray) <= smileNoTeeth(neutral) or getMar(faces, gray) >= smileTeeth(neutral):
        textOnImage(frame, 'smile detected', 50, 50)
        smileCounter.append(0)
    else:
        textOnImage(frame, 'no smile detected', 50,50)
        smileCounter.append(1)
    return frame

neutralIMG = cv2.imread('images/neutral0.jpg')
neutralIMG = preproc(neutralIMG)
neutral = getMar(detector(neutralIMG, 1), neutralIMG)
print("neutral mar value: " + str(neutral) + ", mar value for smile with teeth:"
                                             " " + str(
    smileTeeth(neutral)) + ", mar value for smile without teeth: " + str(smileNoTeeth(neutral)))

cap = cv2.VideoCapture('video/clip0.mov')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps is: " + str(fps))

while True:
    milli = cap.get(cv2.CAP_PROP_POS_MSEC)
    ret, frame = cap.read()
    counter += 1
    if ret:
        frame = resize(frame)
        rgb_frame = frame[:, :, ::-1]
        gray = preproc(frame)
        if counter == fps:
            counter = 0
            frame = detect(gray, frame, detector(rgb_frame, 1))
            saveImage(frame)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()

smileChanges = np.where(np.roll(smileCounter, 1) != smileCounter)[0]
lookAwayChanges = np.where(np.roll(faceCounter, 1) != faceCounter)[0]

howManySmiles = int(len(smileChanges) / 2)
howManyLookAway = int(len(lookAwayChanges) / 2)

worksheet.write('A1', 'Seconds: ')
worksheet.write('B1', 'Smile:')
worksheet.write('C1', 'Face:')
worksheet.write('D1', 'smilecount: ')
worksheet.write('D2', str(howManySmiles))
worksheet.write('E1', 'lookawaycount: ')
worksheet.write('E2', str(howManyLookAway))

for i in range(len(smileCounter)):
    worksheet.write('A'+str(i+2), (i+1))
    worksheet.write('B'+str(i+2), str(smileCounter[i]))
    worksheet.write('C'+str(i+2), str(faceCounter[i]))

workbook.close()