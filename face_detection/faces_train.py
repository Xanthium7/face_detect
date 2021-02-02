import os
from  PIL import Image
import numpy as np
import cv2
import pickle
import time


# ----------- Gets the path of the current python file -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')
face_cascade = cv2.CascadeClassifier('/home/asus/PycharmProjects/face_detection/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() # This thing recognses the face



current_id = 0

label_ids = {}

x_train = []
y_labels = []

for root, dirs, files, in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
            path = os.path.join(root, file)
            # print(path)
            # ----- Getting The Label of the folder ------
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(f"Label: {label},  Path: {path} ")   # ------- Uncomment this line to get the output of te label 
            # x_train.append(path)
            # y_labels.append(label)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # print(label_ids)

            pil_image = Image.open(path).convert("L") # This converts the image to gray scale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            # print(image_array)
            # -------------- We use this arrayed image to train our DATA-------------#

            # finding the region of intrest of the image to train the data #
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            print("training...")
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

time.sleep(0.2)
print(".")
time.sleep(0.2)
print(".")
time.sleep(0.2)
print(".")
time.sleep(0.2)



with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")


print("Training completed Successfully")





