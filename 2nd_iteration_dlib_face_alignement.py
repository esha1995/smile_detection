import cv2
import dlib
import imutils
import numpy as np
import copy
import xlsxwriter

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_5_face_landmarks.dat')
imagePath = 'image'
smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
writePath = 'detected_smiles2'
counter = 0
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


def saveImage(frame, i):
    global imageNumber
    imageNumber += 1
    saveImg = copy.copy(frame)
    textOnImage(saveImg, 'timestamp: ' + str(imageNumber), 50, 200)
    filename = 'video' + str(i) + 'smileSecond' + str(imageNumber) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), saveImg)


def detect(frame, i, dets):
    if len(dets) > 0:
        textOnImage(frame, 'face detected', 30, 80)
        faceCounter[i].append(1)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(frame, detection))
            images = dlib.get_face_chips(frame, faces)
        for image in images:
            cv2.imshow('window', image)
            gray = grayscale(image)
            smiles = smile_cascade.detectMultiScale(gray, 1.75, 20)
            if len(smiles) > 0:
                smileCounter[i].append(1)
                textOnImage(frame, 'smile detected', 30, 30)
            else:
                smileCounter[i].append(0)
                textOnImage(frame, 'no smile detected', 30, 30)
    else:
        textOnImage(frame, 'no face detected', 30, 80)
        faceCounter[i].append(0)

    return frame


participant = 'participant9'

numberOfTests = 5

smileCounter = list()
faceCounter = list()
secondCounter = list()

for i in range(numberOfTests):

    workbook = xlsxwriter.Workbook('xml files/'+participant+'clip'+str(i+1)+'face_aligment.xlsx')
    worksheet = workbook.add_worksheet()

    smileCounter.append([])
    faceCounter.append([])
    secondCounter.append([])

    cap = cv2.VideoCapture('video/clip' + str(i) + '.mov')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps is: ' + str(fps))
    counter = 0
    while True:
        ret, frame = cap.read()
        counter += 1
        if ret:
            frame = resize(frame)
            if counter == 10:
                counter = 0
                secondCounter[i].append(float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                faces = detector(frame, 1)
                frame = detect(frame, i, faces)
            cv2.imshow('video', frame)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    smileChanges = np.where(np.roll(smileCounter[i], 1) != smileCounter[i])[0]
    lookAwayChanges = np.where(np.roll(faceCounter[i], 1) != faceCounter[i])[0]

    howManySmiles = int(len(smileChanges) / 2)
    howManyLookAway = int(len(lookAwayChanges) / 2)

    worksheet.write('A1', 'Seconds: ')
    worksheet.write('B1', 'Smile:')
    worksheet.write('C1', 'Face:')
    worksheet.write('D1', 'smilecount: ')
    worksheet.write('D2', howManySmiles)
    worksheet.write('E1', 'nofacecunt: ')
    worksheet.write('E2', howManyLookAway)

    for j in range(len(smileCounter[i])):

        worksheet.write('B' + str(j + 2), smileCounter[i][j])


    for j in range(len(secondCounter[i])):
        worksheet.write('A' + str(j + 2), secondCounter[i][j])


    for j in range(len(faceCounter[i])):

        worksheet.write('C' + str(j + 2), faceCounter[i][j])


    workbook.close()

