import copy
import cv2
import dlib
import imutils
import numpy as np
import xlsxwriter
from imutils import face_utils
from scipy.spatial import distance as dist

# loading the dlib face detector
detector = dlib.get_frontal_face_detector()
# loading file for predicting the 68 landmarks
sp = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')
# getting the landmarks of the mouth
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# some variables
counter = 0
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
def detect(gray, frame, faces, i):
    global smile
    global face
    if len(faces) < 1:
        textOnImage(frame, 'no face detected', 50, 100)
        faceCounter[i].append(0)
    else:
        textOnImage(frame, 'face detected', 50, 100)
        faceCounter[i].append(1)
    mar = getMar(faces, gray)
    print(mar)
    if mar <= smileNoTeeth(neutral) or mar >= smileTeeth(neutral):
        textOnImage(frame, 'smile detected', 50, 50)
        smileCounter[i].append(1)
    else:
        textOnImage(frame, 'no smile detected', 50, 50)
        smileCounter[i].append(0)
    return frame

# number of tests
numberOfTests = 5

participant = 'participant6'

# lists
smileCounter = list()
faceCounter = list()
secondCounter = list()


# for loop going through code for every test
for i in range(numberOfTests):

    imageNumber = 0

    # creating a list in a list
    smileCounter.append([])
    faceCounter.append([])
    secondCounter.append([])

    # creating a new xml-document
    workbook = xlsxwriter.Workbook('xml files/'+participant+'clip'+str(i+1)+'facial_landmarks_clip.xlsx')
    worksheet = workbook.add_worksheet()

    # loading picture of neutral face and calculating mar values
    neutralIMG = cv2.imread('images/neutral5.png')
    neutralIMG = preproc(neutralIMG)
    neutral = getMar(detector(neutralIMG, 1), neutralIMG)
    print("neutral mar value: " + str(neutral) + ", mar value for smile with teeth:"
                                                 " " + str(
        smileTeeth(neutral)) + ", mar value for smile without teeth: " + str(smileNoTeeth(neutral)))

    # loading video clip and calculating frames pr. second of clip
    cap = cv2.VideoCapture('video/clip' + str(i) + '.mov')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps is: " + str(fps))
    counter = 0
    # going through every frame of video, if one second has passed the detect function is called and an image is saved
    while True:
        ret, frame = cap.read()
        counter += 1
        if ret:
            frame = resize(frame)
            rgb_frame = frame[:, :, ::-1]
            gray = preproc(frame)
            if counter == 10:
                counter = 0
                secondCounter[i].append(float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
                faces = detector(rgb_frame, 1)
                frame = detect(gray, frame, faces, i)
            cv2.imshow('Video', frame)
            cv2.waitKey(1)
        else:
            break

    # realising when video is done and closing all windows
    cap.release()
    cv2.destroyAllWindows()

    # calculating every time there is a change in the lists
    smileChanges = np.where(np.roll(smileCounter[i], 1) != smileCounter[i])[0]
    lookAwayChanges = np.where(np.roll(faceCounter[i], 1) != faceCounter[i])[0]

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

    for j in range(len(smileCounter[i])):
        worksheet.write('A' + str(j + 2), secondCounter[i][j])
        worksheet.write('B' + str(j + 2), smileCounter[i][j])
        worksheet.write('C' + str(j + 2), faceCounter[i][j])
    workbook.close()
