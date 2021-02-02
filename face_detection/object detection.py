import cv2
import numpy as np


wht = 320
cap = cv2.VideoCapture(0)
nmsThreshold = 0.3

classesFile = 'names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(len(classNames))

modelConfig = 'yolo3-320.cfg'
modelWeights = 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confidenceThreshhold = 0.5

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshhold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidenceThreshhold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, x+h), (0,255, 0), 2)
        cv2.putText(img, f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (wht, wht),[0,0,0], 1, crop=False )
    net.setInput(blob)

    layerNames = net.getLayerNames()
    net.getUnconnectedOutLayers()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    output = net.forward(outputNames)
    # print(output[0].shape)
    # print(output[1].shape)
    # # print(output[2].shape)
    # print(output[0][0])

    findObjects(output, img)

    cv2.imshow('Object Detector', img)
    cv2.waitKey(1)

print("Successful")