import copy
import threading
import pyglet
import cv2
import dlib
import imutils
import numpy as np
import xlsxwriter
from imutils import face_utils
from pyglet.app import event_loop
from scipy.spatial import distance as dist
import time
import PySimpleGUI as sg
import vlc
import matplotlib.pyplot as plt

# loading the dlib face detector
detector = dlib.get_frontal_face_detector()
# loading file for predicting the 68 landmarks
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
# getting the landmarks of the mouth
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

neutral = 0.28
counter = 0
smile = False
face = False
frame = cv2.imread('images/neutral0.jpg')
window = pyglet.window.Window()
player = pyglet.media.Player()
source = pyglet.media.StreamingSource()
MediaLoad = pyglet.media.load('klovn.mp4')

eyeAL22 = 0

workbook = xlsxwriter.Workbook('xml files/finaliteratio.xlsx')
worksheet = workbook.add_worksheet()

smileCounter = list()
faceCounter = list()
secondCounter = list()


# resizes image with width of 600
def resize(frame):
    frame = imutils.resize(frame, width=600)
    return frame


# returns histogram stretched grayscale image
def preproc(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
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


# calculates distance from mouth to eye right side
def faceDistance(shape):
    faceD = dist.euclidean(shape[0], shape[16])
    return faceD


# calculates distance from mouth to eye right side
def mouthEyeRDistance(shape):
    MToERight = dist.euclidean(shape[36], shape[48])
    return MToERight


# calculates distance from mouth to eye left side
def mouthEyeLDistance(shape):
    MToELeft = dist.euclidean(shape[45], shape[54])
    return MToELeft


# calculates distance from upper eyelid to lower eyelid right eye
def eyeRDistance(shape):
    #   eyeR1 = dist.euclidean(shape[38], shape[42])
    #  eyeR2 = dist.euclidean(shape[39], shape[41])
    # eyeR = (eyeR1+eyeR2)/2
    # return eyeR
    a1 = dist.euclidean(shape[36], shape[37])
    a2 = dist.euclidean(shape[41], shape[37])
    a3 = dist.euclidean(shape[36], shape[41])
    s1 = (a1 + a2 + a3) / 2
    eyeAR1 = np.math.sqrt(s1 * (s1 - a1) * (s1 - a2) * (s1 - a3))

    b1 = dist.euclidean(shape[41], shape[38])
    b2 = dist.euclidean(shape[38], shape[37])
    b3 = dist.euclidean(shape[41], shape[37])
    s2 = (b1 + b2 + b3) / 2
    eyeAR2 = np.math.sqrt(s2 * (s2 - b1) * (s2 - b2) * (s2 - b3))

    c1 = dist.euclidean(shape[41], shape[38])
    c2 = dist.euclidean(shape[38], shape[40])
    c3 = dist.euclidean(shape[40], shape[41])
    s3 = (c1 + c2 + c3) / 2
    eyeAR3 = np.math.sqrt(s3 * (s3 - c1) * (s3 - c2) * (s3 - c3))

    d1 = dist.euclidean(shape[40], shape[39])
    d2 = dist.euclidean(shape[38], shape[39])
    d3 = dist.euclidean(shape[38], shape[40])
    s4 = (d1 + d2 + d3) / 2
    eyeAR4 = np.math.sqrt(s4 * (s4 - d1) * (s4 - d2) * (s4 - d3))

    eyeAreaR = eyeAR1 + eyeAR2 + eyeAR3 + eyeAR4
    return eyeAreaR


# calculates distance from upper eyelid to lower eyelid left eye
def eyeLDistance(shape):
    global eyeAL22

    #  eyeL1 = dist.euclidean(shape[44], shape[48])
    #  eyeL2 = dist.euclidean(shape[45], shape[47])
    #  eyeL = (eyeL1+eyeL2)/2
    a1 = dist.euclidean(shape[42], shape[43])
    a2 = dist.euclidean(shape[43], shape[47])
    a3 = dist.euclidean(shape[42], shape[47])
    s1 = (a1 + a2 + a3) / 2
    eyeAL1 = np.math.sqrt(s1 * (s1 - a1) * (s1 - a2) * (s1 - a3))

    b1 = dist.euclidean(shape[43], shape[44])
    b2 = dist.euclidean(shape[46], shape[47])
    b3 = dist.euclidean(shape[43], shape[47])
    s2 = (b1 + b2 + b3) / 2
    eyeAL2 = np.math.sqrt(s2 * (s2 - b1) * (s2 - b2) * (s2 - b3))
    eyeAL22 = eyeAL2
    c1 = dist.euclidean(shape[44], shape[47])
    c2 = dist.euclidean(shape[46], shape[47])
    c3 = dist.euclidean(shape[46], shape[44])
    s3 = (c1 + c2 + c3) / 2
    eyeAL3 = np.math.sqrt(s3 * (s3 - c1) * (s3 - c2) * (s3 - c3))

    d1 = dist.euclidean(shape[44], shape[45])
    d2 = dist.euclidean(shape[46], shape[45])
    d3 = dist.euclidean(shape[46], shape[44])
    s4 = (d1 + d2 + d3) / 2
    eyeAL4 = np.math.sqrt(s4 * (s4 - d1) * (s4 - d2) * (s4 - d3))

    eyeAreaL = eyeAL1 + eyeAL2 + eyeAL3 + eyeAL4
    return eyeAreaL


# returns the mar-distance by using two arguments, which is the faces and the grayscale image
def getValues(faces, gray):
    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        toRight = mouthEyeRDistance(shape)
        toLeft = mouthEyeLDistance(shape)
        eyeRight = eyeRDistance(shape)
        eyeLeft = eyeLDistance(shape)
        faceD = faceDistance(shape)
    return toRight, toLeft, eyeRight, eyeLeft, faceD


# returns the mar value of smile without teeth calculated
def smileNoTeeth(neutral):
    smileNoTeeth = neutral * 0.8
    return smileNoTeeth


# returns the mar value of smile without teeth
def smileTeeth(neutral):
    smileTeeth = neutral * 1.2
    return smileTeeth


def eyeSmile(neutralL, neutralR):
    eyeSmile = (neutralL + neutralR) / 2
    eyeSmile = eyeSmile * 0.9
    return eyeSmile


def eyeMouth(neutralL, neutralR):
    eyeMouth = (neutralL + neutralR) / 2
    eyeMouth = eyeMouth * 0.9
    return eyeMouth


# function that saves image with timestamp
def saveImage(frame, test):
    global imageNumber
    imageNumber += 1
    saveImg = copy.copy(frame)
    textOnImage(saveImg, 'timestamp: ' + str(imageNumber), 50, 200)
    filename = 'video' + str(test) + 'smileSecond' + str(imageNumber) + '.jpg'
    cv2.imwrite(np.os.path.join('detected_smiles', filename), saveImg)


def timerCheck():
    global smile
    global face
    global eyeAL22
    timeNow = time.time()
    stamp = 0.5
    while player.playing:
        if stamp == 10:
            player.pause()
            window.close()
        timeis = time.time() - timeNow
        if stamp + 0.1 > timeis > stamp - 0.1:
            secondCounter.append(stamp)
            stamp += 0.5
            if face:
                faceCounter.append(1)
                if smile:
                    smileCounter.append(1)
                else:
                    smileCounter.append(0)
            else:
                smileCounter.append(0)
                faceCounter.append(0)

        time.sleep(0.000001)
    print('thread stopped')


def main():
    global counter
    global neutral
    global smile
    global face
    global frame
    global checkerThread
    global player
    cap = cv2.VideoCapture(2)
    while player.playing:
        ret, frame = cap.read()
        if ret:
            frame = resize(frame)
            rgb_frame = frame[:, :, ::-1]
            gray = preproc(frame)
            faces = detector(rgb_frame, 1)
            if len(faces) > 0:
                try:
                    face = True
                    toRight, toLeft, eyeRight, eyeLeft, faceD = getValues(faces, gray)
                    procentChange = faceD / faceDN
                    eyeChangeL = eyeLeftN * procentChange
                    eyeChangeR = eyeRightN * procentChange
                    eyeChange = (eyeRight + eyeLeft) / 2
                    eyeMouthChangeL = toLeftN * procentChange
                    eyeMouthChangeR = toRightN * procentChange
                    eyeMouthChange = (toRight + toLeft) / 2
                    if eyeChange < eyeSmile(eyeChangeL, eyeChangeR) and eyeMouthChange < eyeMouth(eyeMouthChangeL,
                                                                                                  eyeMouthChangeR):
                        smile = True
                    else:
                        smile = False
                except:
                    print('math error')
            else:
                face = False
            cv2.waitKey(1)
            time.sleep(0.3)
        else:
            break
    # realising when video is done and closing all windows
    cap.release()
    print('thread stopped')


cap = cv2.VideoCapture(2)

while True:
    marN = 0
    toRightN = 0
    toLeftN = 0
    eyeRightN = 0
    eyeLeftN = 0
    faceDN = 0
    ret, frame = cap.read()
    counter += 1
    if ret:
        frame = resize(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rgb_frame = frame[:, :, ::-1]
            gray = preproc(frame)
            faces = detector(rgb_frame, 1)
            if len(faces) > 0:
                toRightN, toLeftN, eyeRightN, eyeLeftN, faceDN = getValues(faces, gray)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print(faceDN)

detectorThread = threading.Thread(target=main)
timerThread = threading.Thread(target=timerCheck)

player.queue(MediaLoad)
player.play()

detectorThread.start()
timerThread.start()


@window.event
def on_draw():
    window.clear()
    if player.source and player.source.video_format:
        player.get_texture().blit(50, 50)


pyglet.app.run()

# calculating every time there is a change in the lists
smileChanges = np.where(np.roll(smileCounter, 1) != smileCounter)[0]
lookAwayChanges = np.where(np.roll(faceCounter, 1) != faceCounter)[0]

# calculating number of smiles (which should bed half the times a change has occured)
howManySmiles = int(len(smileChanges) / 2)
howManyLookAway = int(len(lookAwayChanges) / 2)

# writing the values in the created worksheet
worksheet.write('A1', 'Seconds: ')
worksheet.write('B1', 'Smile:')
worksheet.write('C1', 'Face:')
worksheet.write('D1', 'smilecount: ')
worksheet.write('D2', howManySmiles)
worksheet.write('E1', 'lookawaycount: ')
worksheet.write('E2', howManyLookAway)

for j in range(len(smileCounter)):
    worksheet.write('A' + str(j + 2), secondCounter[j])
    worksheet.write('B' + str(j + 2), smileCounter[j])
    worksheet.write('C' + str(j + 2), faceCounter[j])
workbook.close()

timestamp1Start = 7
timestamp1End = 9
count1 = 0
counter = 0

for i in range(7, 9):
    counter += 1
    count1 += smileCounter[i]
count1 = count1 / counter

print(count1)
if count1 == 0:
    print('no smile')
elif count1 > 0.5:
    print('smiled')

plt.plot(secondCounter, smileCounter)
plt.show()
