import cv2 as cv
import numpy as np
import time

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None
imgWidth=416
imgHeight=416

# Load names of classes and get random colors
classesFile='coco.names'
classes=None
with open(classesFile,'r') as f:
    classes = f.read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
modelConfig='/home/ntquyen/ntquyen/Machine_Learning/ThayHoang/HumanDetection/withDarknet/yolov4-tiny.cfg'
modelWeights='yolov4-tiny_last.weights'
net = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# determine the output layer, getUnconnectedOutLayers return the names of the unconnected output layers
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def load_image(path):
    global img, img0, outputs, ln

    img0 = cv.imread(path)
    img = img0.copy()

    # Create a 4D bolb image
    blob = cv.dnn.blobFromImage(img, 1/255.0, (imgWidth, imgHeight), swapRB=True, crop=False)
    
    # Sets the input to the network
    net.setInput(blob)
    t0 = time.time()
    # Runs the forward pass to get output of the output layers, forward function need the ending layers 
    outputs = net.forward(ln)
    t = time.time() - t0

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)

    # Removes the bouding boxes with low confidence
    post_process(img, outputs, 0.5)
    cv.imshow('window',  img)
    cv.displayOverlay('window', f'forward propagation time={t:.3}')
    cv.waitKey(0)

def post_process(img, outputs, conf):
    # H: frameHeight, W: frameWeigh
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []
    # Scan through all the bounding boxes output form the network and keep onely the ones with confidence scores
    # Assign the box's class label as the class with the heighest score
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)
    #remove noise due to numerous boxes
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
      for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # color = [int(c) for c in colorsOANGPHJFAF[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h),(255,0,0), 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
def trackbar(x):
    global img
    conf = x/100
    img = img0.copy()
    post_process(img, outputs, conf)
    cv.displayOverlay('window', f'confidence level={conf}')
    cv.imshow('window', img)

cv.namedWindow('window')
cv.createTrackbar('confidence', 'window', 50, 100, trackbar)
# load_image('/home/ntquyen/ntquyen/Machine_Learning/images/val2017/000000000139.jpg')
# load_image('/home/ntquyen/ntquyen/Machine_Learning/images/val2017/000000010707.jpg')
load_image('/home/ntquyen/ntquyen/Machine_Learning/ThayHoang/HumanDetection/withDarknet/yolotinyv3_medmask_demo/obj/0634.jpg')
cv.waitKey()
cv.destroyAllWindows()

