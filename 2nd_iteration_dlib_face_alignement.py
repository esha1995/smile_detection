import cv2
import dlib
import imutils
import numpy as np
import copy
import xlsxwriter

workbook = xlsxwriter.Workbook('xml files/2nd_iteration_face_aligment.xlsx')
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
counter = 0
smile = False
face = False
imageNumber = 0

def resize(frame):
    frame = imutils.resize(frame, width=600)
    return frame

def textOnImage(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def saveImage(frame):
    global imageNumber
    imageNumber += 1
    saveImg = copy.copy(frame)
    textOnImage(saveImg, 'timestamp: ' + str(imageNumber), 50, 200)
    filename = 'smileSecond' + str(imageNumber) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), saveImg)

def detect(frame):
    global face
    global smile
    rgb_frame = frame[:, :, ::-1]
    dets = detector(rgb_frame, 1)
    if len(dets) > 0:
        textOnImage(frame,'face detected', 30, 80)
        faceCounter.append(1)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(frame, detection))
            images = dlib.get_face_chips(frame, faces)
        for image in images:
            gray = grayscale(image)
            smiles = smile_cascade.detectMultiScale(gray, 1.5, 20)
            if len(smiles) > 0:
                smileCounter.append(1)
                textOnImage(frame, 'smile detected', 30, 30)
            else:
                smileCounter.append(0)
                textOnImage(frame, 'no smile detected', 30, 30)
    else:
        textOnImage(frame, 'no face detected', 30, 80)
        faceCounter.append(0)
    return frame

cap = cv2.VideoCapture('video/clip0.mov')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print('fps is: '+str(fps))

while True:
    ret, frame = cap.read()
    counter += 1
    if ret:
        frame = resize(frame)
        if counter == fps:
            counter = 0
            frame = detect(frame)
            saveImage(frame)
        cv2.imshow('video', frame)
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
worksheet.write('D2', howManySmiles)
worksheet.write('E1', 'nofacecunt: ')
worksheet.write('E2', howManyLookAway)

for i in range(len(smileCounter)):
    worksheet.write('A'+str(i+2), i+1)
    worksheet.write('B'+str(i+2), smileCounter[i])
    worksheet.write('C'+str(i+2), faceCounter[i])
workbook.close()
