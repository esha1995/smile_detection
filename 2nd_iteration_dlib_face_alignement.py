import cv2
from threading import Thread
import dlib
import imutils
from multiprocessing import Process

smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib files/shape_predictor_5_face_landmarks.dat')

imagePath = 'image'
img = cv2.imread('/Users/emilhansen/Desktop/dlib face_detection 2/images/smile6.jpg')
smile_cascade = cv2.CascadeClassifier('OpenCV files/haarcascade_smile.xml')


def detect(frame):
    dets = detector(frame, 1)
    print("faces: "+str(len(dets)))
    if len(dets) > 0:
        cv2.putText(frame, 'face detected', (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(frame, detection))
            images = dlib.get_face_chips(frame, faces)
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            #gray = cv2.GaussianBlur(gray, (5,5), 0)
            smiles = smile_cascade.detectMultiScale(gray, 1.65, 20)
            print("smiles" + str(len(smiles)))
            if len(smiles) > 0:
                cv2.putText(frame, 'smile detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'no smile detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('window', gray)
    else:
        cv2.putText(frame, 'no face detected', (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
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
def show_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        canvas = detect(frame)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def write_video():
    cap2 = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = float(cap2.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.avi', fourcc, fps, (640, 480))
    while True:
        ret, test = cap2.read()
        test = imutils.resize(test, width=400)
        out.write(test)
        cv2.imshow('Test', test)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    p1= Process(target = show_video)
    p2= Process(target = write_video)
    p1.start()
    p2.start()

    p1.join()
    p2.join()


#"""

