import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from imutils import face_utils
import vlc

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
mouth = list(range(48, 61))
model = dlib.face_recognition_model_v1('dlib files/dlib_face_recognition_resnet_model_v1.dat')
smileCounter = list()
lookAwayCounter = list()
framerate = 0

def btn(name):  # a PySimpleGUI "User Defined Element" (see docs)
    return sg.Button(name, size=(6, 1), pad=(1, 1))

def detect(gray, frame,sf,mn):
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
        smiles = smile_cascade.detectMultiScale(roi_gray, sf, mn)
        if len(smiles) > 0:
            smileCounter.append(1)
        else:
            smileCounter.append(0)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

def calibrate(gray, frame,sf,mn):
    faces = detector(gray, 1)
    for (i, faces) in enumerate(faces):
        shape = sp(gray, faces)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(faces)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        smiles = smile_cascade.detectMultiScale(roi_gray, sf, mn)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

def main():
    sg.theme("DarkBlue")

    layout = [
        [sg.Input(default_text='Video URL or Local Path:', size=(30, 1), key='-VIDEO_LOCATION-'), sg.Button('load')],
        [sg.Text("OpenCV test", size=(60, 1), justification="left")],
        [sg.Image(filename="",key="-IMAGE-")],
        [sg.Button("Start cal", size=(10, 1)),sg.Button("Stop cal",size=(10,1))],

        [btn('previous'), btn('play'), btn('next'), btn('pause'), btn('stop')],
        [sg.Text('Load media to start', key='-MESSAGE_AREA-')],
        [sg.Text("Scale Factor", size=(60, 1), justification="left")],
        [

            sg.Slider(
                (110, 200),
                110,
                5,
                orientation="h",
                size=(20, 10),
                key="-SCALE-",
            ),
        ],
        [sg.Text("Neighbor", size=(60, 1), justification="left")],
        [
            sg.Slider(
                (10, 20),
                15,
                1,
                orientation="h",
                size=(20, 10),
                key="-NEIGHB-",
            ),
        ],
        [sg.Button("Exit", size=(10, 1))],
    ]

    while True:
        try:
            webcam = int(input("enter camera input: "))
            framerate = int(input("framerate of camera: "))
            break
        except:
            print("only numbers")

    window = sg.Window("Test", layout, element_justification='center', finalize=True, resizable=True)
    window['-IMAGE-'].expand(True, True)
    inst = vlc.Instance()
    list_player = inst.media_list_player_new()
    media_list = inst.media_list_new([])
    list_player.set_media_list(media_list)
    player = list_player.get_media_player()
    player.set_hwnd(window['-IMAGE-'].Widget.winfo_id())
    cap = cv2.VideoCapture(webcam)
    calibrateBool = False
    videoLoaded = False
    videoPlaying = False
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or sg.WINDOW_CLOSED:
            plt.plot(smileCounter, color='red', label='smiles')
            plt.plot(lookAwayCounter, color='blue', label='looking away')
            plt.yticks(np.arange(0, 2, 1))
            plt.xticks(np.arange(0, 100, framerate))
            plt.legend()
            plt.show()
            break
        elif event == "Start cal":
            calibrateBool = True
        elif event == "Stop cal":
            calibrateBool = False
        elif calibrateBool:
            event, values = window.read(timeout=20)
            sf = values["-SCALE-"]/100
            mn = int(values["-NEIGHB-"])
            _, frame = cap.read()
            frame = cv2.resize(frame, (700, 512))
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            canvas = calibrate(gray, frame, sf, mn)


            cv2.putText(canvas, 'Smile and adjust the scale factor', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'till there is only one smile', (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
            imgbytes = cv2.imencode(".png", canvas)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

        elif videoPlaying:
            event, values = window.read(timeout=20)
            sf = values["-SCALE-"]/100
            mn = int(values["-NEIGHB-"])
            _, frame = cap.read()
            frame = cv2.resize(frame, (700, 512))
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            detect(gray,frame,sf,mn)

        elif not calibrateBool or videoLoaded:
            _, frame = cap.read()
            frame = cv2.resize(frame, (700, 512))
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, 'Press start to calibrate', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
            # cv2.imshow('Video', frame)
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)
        if event == 'play':
            videoPlaying = True
            list_player.play()
        if event == 'pause':
            list_player.pause()
        if event == 'stop':
            videoPlaying = False
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
                videoLoaded = True
                media_list.add_media(values['-VIDEO_LOCATION-'])
                list_player.set_media_list(media_list)
                window['-VIDEO_LOCATION-'].update('Video URL or Local Path:')  # only add a legit submit

        # update elapsed time if there is a video loaded and the player is playing
        if player.is_playing():
            window['-MESSAGE_AREA-'].update(
                "{:02d}:{:02d} / {:02d}:{:02d}".format(*divmod(player.get_time() // 1000, 60),
                                                       *divmod(player.get_length() // 1000, 60)))
        else:
            window['-MESSAGE_AREA-'].update('Load media to start' if media_list.count() == 0 else 'Ready to play media')



    cap.release()
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
            print("person looked away at: " + str(
                lookAwayChanges[index] / framerate) + " seconds, and lasted for: " + str(
                (lookAwayChanges[index + 1] - smileChanges[index]) / framerate) + " seconds.")
            lookbool = True
        else:
            print("person looked away at: " + str(
                lookAwayChanges[index + 1] / framerate) + " seconds, and lasted for: " + str(
                (lookAwayChanges[index + 2] - lookAwayChanges[index + 1]) / framerate) + " seconds.")
    window.close()
main()

