import os
import numpy as np
import cv2
from PIL import Image  # For face recognition we will use the LBPH Face Recognizer
import pickle


def main():
    while True:
        choice = int(input('''
        ###################################################################
        1 --> Register New Face.
        2 --> Face Recognization.
        3 --> Exit.
        Your Choice: 
        '''))

        if choice == 1:
            FaceCapture()
            FaceLearning()
        elif choice == 2:
            FaceRecognition()
        elif choice == 3:
            print('\nExiting Program.')
            break



def FaceLearning():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    path = "C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\facedata"

    def getImagesWithID(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

        # print image_path

        # getImagesWithID(path)

        faces = []
        IDs = []

        for imagePath in imagePaths:
            # Read the image and convert to grayscale

            facesImg = Image.open(imagePath).convert('L')

            faceNP = np.array(facesImg, 'uint8')

            # Get the label of the image

            ID = int(os.path.split(imagePath)[-1].split(".")[1])

            # Detect the face in the image

            faces.append(faceNP)

            IDs.append(ID)

            cv2.imshow("Adding faces for traning", faceNP)

            cv2.waitKey(10)

        return np.array(IDs), faces

    Ids, faces = getImagesWithID(path)

    recognizer.train(faces, Ids)

    recognizer.save("C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\train_data\\trainingdata.yml")

    cv2.destroyAllWindows()


def FaceRecognition():

    face_cascade = cv2.CascadeClassifier('C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\train_data\\trainingdata.yml")
    #id = ""
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # font = cv2.(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 4)
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            id, confidence = rec.predict(gray[y:y + h, x:x + w])
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()


def FaceCapture():
    face_cascade = cv2.CascadeClassifier('C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    with open('C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\names.pkl', 'rb') as f:
        names = pickle.load(f)

    name = input('Enter name for the Face: ')
    names.append(name)
    id = names.index(name)

    sampleN = 0

    while 1:

        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            print(sampleN)
            sampleN = sampleN + 1;

            cv2.imwrite(
                "C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\facedata\\User." + str(id) + "." + str(
                    sampleN) + ".jpg", gray[y:y + h, x:x + w])

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.waitKey(100)

        cv2.imshow('img', img)

        cv2.waitKey(1)

        if sampleN > 100:
            break
    with open('C:\\Users\\Sourabh Burmun\\PycharmProjects\\FaceRecognition\\names.pkl', 'wb') as f:
        pickle.dump(names, f)
    cap.release()

    cv2.destroyAllWindows()

main()