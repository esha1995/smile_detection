import cv2
import numpy as np

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_frontalface_default.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
    print("faces: " + str(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.2, 15)
        print("smiles: " + str(len(smiles)))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

imagePath = 'images/smile'
writePath = 'openCV_imagewrite'

for i in range(24):
    print("image number: " + str(i) + ": ")
    cap = cv2.imread(imagePath+str(i)+'.jpg')
    frame = cv2.resize(cap, (700, 512))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    canvas = detect(gray, frame)
    filename = 'smileDetected' + str(i) + '.jpg'
    cv2.imwrite(np.os.path.join(writePath, filename), canvas)

