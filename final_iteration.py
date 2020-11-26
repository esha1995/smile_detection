import copy
import threading
import pyglet
import cv2
import dlib
import imutils
import numpy as np
import xlsxwriter
from imutils import face_utils
from scipy.spatial import distance as dist
import time
import PySimpleGUI as sg
import vlc

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

smileCounter = list()
faceCounter = list()
secondCounter = list()

workbook = xlsxwriter.Workbook('xml files/finaliteration')
worksheet = workbook.add_worksheet()


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


# returns the mar-distance by using two arguments, which is the faces and the grayscale image
def getMar(faces, gray):
    mar = neutral
    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mar = distance(mouth)
    return mar


# returns the mar value of smile without teeth calculated
def smileNoTeeth(neutral):
    smileNoTeeth = neutral * 0.8
    return smileNoTeeth


# returns the mar value of smile without teeth
def smileTeeth(neutral):
    smileTeeth = neutral * 1.2
    return smileTeeth


# function that saves image with timestamp
def saveImage(frame, test):
    global imageNumber
    imageNumber += 1
    saveImg = copy.copy(frame)
    textOnImage(saveImg, 'timestamp: ' + str(imageNumber), 50, 200)
    filename = 'video' + str(test) + 'smileSecond' + str(imageNumber) + '.jpg'
    cv2.imwrite(np.os.path.join('detected_smiles', filename), saveImg)


# function which detects smile based on calculated mar-values and appends answer to lists
def detect(gray, frame, faces):
    global smile
    global face
    if len(faces) < 1:
        face = False
    else:
        face = True
    mar = getMar(faces, gray)
    if mar <= smileNoTeeth(neutral) or mar >= smileTeeth(neutral):
        smile = True
    else:
        smile = False
    return frame


def checker():
    print('checker thread has started')
    global frame
    global smile
    global face
    global player
    global smileCounter
    global faceCounter
    global secondCounter
    global workbook
    global worksheet
    timeNow = time.time()
    stamp = 0.5
    while player.playing:
        timeis = time.time() - timeNow
        if stamp == 10:
            player.pause()
        if stamp + 0.1 > timeis > stamp - 0.05:
            secondCounter.append(stamp)
            stamp += 0.5
            if face:
                print('face')
                faceCounter.append(1)
                if smile:
                    smileCounter.append(1)
                    print('smile')
                else:
                    smileCounter.append(0)
                    print('no smile')
            else:
                faceCounter.append(0)
                print('no face')
                smileCounter.append(0)
                print('no smile')

    print("thread 1 stopped")


checkerThread = threading.Thread(target=checker)

def main():
    global counter
    global neutral
    global smile
    global face
    global frame
    global videoThread
    global checkerThread
    global player

    cap = cv2.VideoCapture(2)

    while player.playing:
        ret, frame = cap.read()
        counter += 1
        if ret:
            frame = resize(frame)
            rgb_frame = frame[:, :, ::-1]
            gray = preproc(frame)
            if counter == 10:
                counter = 0
                faces = detector(rgb_frame, 1)
                frame = detect(gray, frame, faces)
            cv2.waitKey(1)
        else:
            break
    # realising when video is done and closing all windows
    cap.release()

detectorThread = threading.Thread(target=main)

while True:
    try:
        webcam = int(input("enter camera input: "))
        break
    except:
        print("only numbers")


player.queue(MediaLoad)
player.play()
checkerThread.start()
detectorThread.start()

@player.event
def on_eos():

    print('video end')

@window.event
def on_draw():
    window.clear()
    if player.source and player.source.video_format:
        player.get_texture().blit(0, 0)


pyglet.app.run()

