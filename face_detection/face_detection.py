import cv2 
import numpy as np
import pickle
import PIL


# -------------------------------------Providing the location of the aar cascade file for fontal face detection ----------------
face_cascade = cv2.CascadeClassifier('/home/asus/PycharmProjects/face_detection/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() # This thing recognses the face
labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

recognizer.read("trainner.yml")

cap = cv2.VideoCapture(0)
img_counter = 0
icnt = 0
counter = 0 # Counter initialised for limiting the loop

flag = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=5)
    # # print("30", faces)
    # if  len(faces) == 0:
    #     print(faces)
    #     flag = False
    # if  flag == False:
    for (x, y ,w, h) in faces:
        print("35", faces)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]

#-------------------------- How to recognise Faces--------------------------------#

        id_, conf = recognizer.predict(roi_gray)
        if conf>80 and conf<=100: # and conf <=85:
            # print(labels[id_])
            print("known")
            font= cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 123, 255)
            stroke = 2
            color = (0, 255, 0)
            stroke = 5
            x_end = x+w
            y_end = y+h
            cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)
            
            cv2.putText(frame, name, (x, y-20), font, 1, color, 2, cv2.LINE_AA)
        # else:
        #     color = (0, 0, 255)
        #     stroke = 5
        #     x_end = x+w
        #     y_end = y+h
        #     cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)
        #     cv2.putText(frame, 'unknown', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #     # image_name = "/home/asus/PycharmProjects/face_detection/images/unknown{}.png".format(img_counter)
        #     # cv2.imwrite(image_name, frame)
        else:
            print("unknown")
            color = (0, 0, 255)
            stroke = 5
            x_end = x+w
            y_end = y+h
            
            cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)
            cv2.putText(frame, 'unknown', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image_name = "/home/asus/PycharmProjects/face_detection/unknown_faces/unknown{}.png".format(img_counter)
            cv2.imwrite(image_name, frame)
            img_counter = img_counter+1 # Incrementing the unknown face naming counter
        
                
        # flag = True
    def imwrite(icnt,cv2) :
    
        if icnt == 1:
            image_name = "/home/asus/PycharmProjects/face_detection/unknown_faces/unknown{}.png".format(img_counter)
            cv2.imwrite(image_name, frame)
            icnt = 0
    



    cv2.imshow('Face Detection', frame)
    
    # Excicuting the Quit command
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print("Successful 🔥🔥🔥")