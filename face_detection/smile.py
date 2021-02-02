import cv2 

smile_cascade = cv2.CascadeClassifier('/home/asus/PycharmProjects/face_detection/cascades/data/haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('/home/asus/PycharmProjects/face_detection/cascades/data/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('/home/asus/PycharmProjects/face_detection/cascades/data/haarcascade_profileface.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=17)
    # profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)

    for (x, y ,w, h) in smiles:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        color = (0, 255, 0)
        stroke = 5
        x_end = x+w
        y_end = y+h
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)
        


    cv2.imshow('Smile Detection', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













