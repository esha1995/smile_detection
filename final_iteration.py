import copy
import threading

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
from sys import platform as PLATFORM


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
videoPlaying = True

def mediaplayer():
    global videoPlaying
    sg.theme('DarkBlue')

    def btn(name):  # a PySimpleGUI "User Defined Element" (see docs)
        return sg.Button(name, size=(6, 1), pad=(1, 1))

    layout = [
        [sg.Input(default_text='Video URL or Local Path:', size=(30, 1), key='-VIDEO_LOCATION-'), sg.Button('load')],
        [sg.Image('', size=(1024, 576), key='-VID_OUT-')],
        [btn('previous'), btn('play'), btn('next'), btn('pause'), btn('stop')],
        [sg.Text('Load media to start', key='-MESSAGE_AREA-')]]

    window = sg.Window('Mini Player', layout, element_justification='center', finalize=True, resizable=True)

    window['-VID_OUT-'].expand(True, True)  # type: sg.Element
    # ------------ Media Player Setup ---------#

    inst = vlc.Instance()
    list_player = inst.media_list_player_new()
    media_list = inst.media_list_new([])
    list_player.set_media_list(media_list)
    player = list_player.get_media_player()
    if PLATFORM.startswith('linux'):
        player.set_xwindow(window['-VID_OUT-'].Widget.winfo_id())
    else:
        player.set_hwnd(window['-VID_OUT-'].Widget.winfo_id())

    test = True
    # ------------ The Event Loop ------------#
    while True:
        event, values = window.read(timeout=1000)  # run with a timeout so that current location can be updated
        if test:
            media_list.add_media("test.mp4")
            list_player.set_media_list(media_list)
            list_player.play()
            test = False
        if videoPlaying == False:
            break

        if event == sg.WIN_CLOSED:
            break
        if event == 'play':
            list_player.play()
        if event == 'pause':
            list_player.pause()
        if event == 'stop':
            list_player.stop()
        if event == 'next':
            list_player.next()
            list_player.play()
        if event == 'previous':
            list_player.previous()  # first call causes current video to start over
            list_player.previous()  # second call moves back 1 video from current
            list_player.play()
        if event == 'load':
            if values['-VIDEO_LOCATION-'] and not 'Video URL' in values['-VIDEO_LOCATION-']:
                media_list.add_media(values['-VIDEO_LOCATION-'])
                list_player.set_media_list(media_list)
                window['-VIDEO_LOCATION-'].update('Video URL or Local Path:')  # only add a legit submit

        # update elapsed time if there is a video loaded and the player is playing
        if player.is_playing():
            checkerThread.start()
            videoPlaying = True
            window['-MESSAGE_AREA-'].update(
                "{:02d}:{:02d} / {:02d}:{:02d}".format(*divmod(player.get_time() // 1000, 60),
                                                       *divmod(player.get_length() // 1000, 60)))
        else:
            videoPlaying = False
            window['-MESSAGE_AREA-'].update('Load media to start' if media_list.count() == 0 else 'Ready to play media')

    window.close()

# resizes image with width of 600
def resize(frame):
    frame = imutils.resize(frame, width=600)
    return frame

# returns histogram stretched grayscale image
def preproc(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
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
    textSmile = ""
    textFace = ""
    timeNow = time.time()
    stamp = 0.5
    while True:
        if videoPlaying == False:
            break;
        timeis = time.time()-timeNow
        textOnImage(frame, textSmile, 50,50)
        textOnImage(frame, textFace, 50,80)
        if stamp+0.2 > timeis > stamp-0.07:
            print(stamp)
            print(timeis)
            stamp += 0.5
            if face:
                textFace = 'face detected'
                if smile:
                    textSmile = 'smile detected'
                else:
                    textSmile = 'no smile detected'
            else:
                textFace = 'no face detected'
                textSmile = 'no smile detected'



mediaThread = threading.Thread(target=mediaplayer)
checkerThread = threading.Thread(target=checker)

while True:
    try:
        webcam = int(input("enter camera input: "))
        break
    except:
        print("only numbers")

cap = cv2.VideoCapture(webcam, cv2.CAP_DSHOW)


mediaThread.start()


while True:
    ret, frame = cap.read()
    counter += 1
    if ret:
        if videoPlaying == False:
            break
        frame = resize(frame)
        rgb_frame = frame[:, :, ::-1]
        gray = preproc(frame)
        if counter == 10:
            counter = 0
            faces = detector(rgb_frame, 1)
            frame = detect(gray, frame, faces)
        #cv2.imshow('Video', frame)
        cv2.waitKey(1)
    else:
        break

    # realising when video is done and closing all windows
cap.release()
cv2.destroyAllWindows()








