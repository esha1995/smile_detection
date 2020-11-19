import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import xlsxwriter

workbook = xlsxwriter.Workbook('2nd_iteration_face_aligment.xlsx')
worksheet = workbook.add_worksheet()



smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_5_face_landmarks.dat')

imagePath = 'image'
smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')

writePath = 'detected_smiles2'
now = 0
smileCounter = list()
faceCounter = list()

def resize(frame):
    frame = imutils.resize(frame, width=600)
    return frame

def textOnImage(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def saveImage(frame):
    saveImg = copy.copy(frame)
    cv2.putText(saveImg, 'timestamp: ' + str(datetime.now()-now), (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    filename = 'smileTime' + str(datetime.now()-now) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), saveImg)

def grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def detect(frame):
    global counter
    dets = detector(frame, 1)
    if len(dets) > 0:
        textOnImage(frame,'face detected', 30, 80)
        faceCounter.append(1)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(frame, detection))
            images = dlib.get_face_chips(frame, faces)
        for image in images:
            gray = grayscale(image)
            smiles = smile_cascade.detectMultiScale(gray, 1.65, 20)
            if len(smiles) > 0:
                smileCounter.append(1)
                counter += 1
                textOnImage(frame, 'smile detected', 30, 30)
                if counter == 10:
                    saveImage(frame)
                    counter = 0
            else:
                smileCounter.append(0)
                textOnImage(frame, 'no smile detected', 30, 30)
                counter = 0
    else:
        textOnImage(frame, 'no face detected', 30, 80)
        faceCounter.append(0)
    return frame

""""
imagePath = 'images/smile'
writePath = 'dlib_rotated_write'
for i in range(35):
    print("image number: " + str(i) + ": ")
    cap = imread(imagePath+str(i)+'.jpg')
    frame = imutils.resize(cap, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = detect(frame)
    filename = 'smileDetected' + str(i) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), canvas)
"""

while True:
    try:
        webcam = int(input("enter camera input (default is 0): "))
        framerate = int(input("framerate of camera (default is 24): "))
        break
    except:
        print("Only numbers")

cap = cv2.VideoCapture(webcam)
now = datetime.now()
while True:
    _, frame = cap.read()
    frame = resize(frame)
    canvas = detect(frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

smileChanges = np.where(np.roll(smileCounter, 1) != smileCounter)[0]
lookAwayChanges = np.where(np.roll(faceCounter, 1) != faceCounter)[0]

howManySmiles = int(len(smileChanges) / 2)
howManyLookAway = int(len(lookAwayChanges) / 2)

print("person smiled " + str(howManySmiles) + " times, and looked away " + str(howManyLookAway) + " times")

worksheet.write('A1', 'Smiles:')
worksheet.write('B1', 'Faces:')
for i in range(len(smileCounter)):
    worksheet.write('A'+str(i+2), str(smileCounter[i]))
    worksheet.write('B'+str(i+2), str(faceCounter[i]))
workbook.close()

plt.plot(smileCounter, color='red', label='smiles')
plt.plot(faceCounter, color='blue', label='looking away')
plt.yticks(np.arange(0, 2, 1))
plt.xticks(np.arange(0, 100, framerate))
plt.legend()
plt.show()

#"""

