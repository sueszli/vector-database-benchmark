import cv2 as cv
import numpy as np
import glob

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    check = net.getUnconnectedOutLayers().tolist()
    # If error switch line
    return [layersNames[i - 1] for i in check]
    # return [layersNames[i[0] - 1] for i in check]

# Draw the predicted bounding box
def drawPred(image, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(image, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(image, outs):
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * imageWidth)
                center_y = int(detection[1] * imageHeight)
                width = int(detection[2] * imageWidth)
                height = int(detection[3] * imageHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    oBoxes = []
    oClasses = []
    for i in indices:
        # If error uncomment below
        # i = i[0]
        box = boxes[i]
        oBoxes.append(box)
        oClasses.append(classIds[i])
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(image, classIds[i], confidences[i], left, top, left + width, top + height)
    return oBoxes, oClasses

confThreshold = 0.75  #Confidence threshold
nmsThreshold = 0.5   #Non-maximum suppression threshold
inpWidth = 832       #Width of network's input image
inpHeight = 832      #Height of network's input image

# Load names of classes
classesFile = "cones.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov4-tiny-cones.cfg"
modelWeights = "yolov4-tiny-cones_best.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if(not cv.cuda.getCudaEnabledDeviceCount()):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
else:
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')


objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = []
boxsizeL = []
boxsizeR = []
images = glob.glob("Camera Calibration GRID IMAGES/*")
images = sorted(images)

for image_name in images:
    image = cv.imread(image_name)
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    boxes, classes = postprocess(image, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    center = []
    for r in range(len(boxes)):
        box = boxes[r]
        x = box[0] + box[2]/2
        y = box[1] + box[3]/2
        center.append([x, y])
        boxsize = [box[2], box[3]]
    center = np.array(center)
    for point, cl in zip(center, classes):
        x, y = point
        if cl == 0:
            color = (255, 0, 0)
        elif cl == 1:
            color = (0, 255, 0)
        elif cl == 2:
            color = (0, 255, 255)
        else:
            color = (0, 0, 0)
        image = cv.circle(image,(int(x),int(y)),5,color,-1)
    
    image_name = image_name.replace("Camera Calibration GRID IMAGES/", "").replace(".png","")
    side, x, y = image_name.split("_")
    x = float(x)
    y = float(y)
    try:
        ind = objpoints.index([x, y, 0])
    except ValueError:
        objpoints.append([x, y, 0])
        ind = objpoints.index([x, y, 0])
    
    if side == 'R':
        while len(imgpointsR) <= ind:
            imgpointsR.append([])
            boxsizeR.append([])
        imgpointsR[ind] = center[0]
        boxsizeR[ind] = boxsize
    elif side == 'L':
        while len(imgpointsL) <= ind:
            imgpointsL.append([])
            boxsizeL.append([])
        imgpointsL[ind] = center[0]
        boxsizeL[ind] = boxsize
    else:
        print("ERROR! Invalid file name")
        exit()
    
    


    # cv.destroyAllWindows()
    # w, h, c = image.shape
    # imageZ = cv.resize(image, (int(h/2), int(w/2)))
    # cv.imshow(image_name, imageZ)
    # cv.waitKey(100)

cv.destroyAllWindows()
        
objpoints = np.array([objpoints], dtype=np.float32)
imgpointsL = np.array([imgpointsL], dtype=np.float32)
imgpointsR = np.array([imgpointsR], dtype=np.float32)
print("RealWorld")
print(objpoints)
print("Image Points L")
print(imgpointsL)
print(np.array(boxsizeL))
print("Image Points R")
print(imgpointsR)
print(np.array(boxsizeR))


w, h, c = image.shape

if not len(imgpointsL) == len(imgpointsR):
    print("Error, not all points found")
    exit()
ret, mtxL, distL, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpointsL, (w, h), None, None)
print("Camera 1 Rotation")
print(rvecs)
print("Camera 1 Translation")
print(tvecs)
if not ret:
    print("Error, Camera Left Calibration")
    exit()
ret, mtxR, distR, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpointsR, (w, h), None, None)
if not ret:
    print("Error, Camera Right Calibration")
    exit()

# Stereo Calibration
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, (w, h))
if not ret:
    print("Error, Stereo Calibration")
    exit()
print("Calibration Matrix L")
print(mtxL)
print("Distortion L")
print(distL)
print("Calibration Matrix R")
print(mtxR)
print("Distortion R")
print(distR)
print("Rotation")
print(R)
print("Translation")
print(T)

print("Essential")
print(E)
print("Fundamental")
print(F)

RTL = np.concatenate([np.eye(3), [[0],[0],[0]]], axis=-1)
PL = mtxL @ RTL
RTR = np.concatenate([R, T], axis=-1)
PR = mtxR @ RTR

print("Projection Matrices")
print(PL)
print(PR)

points4D = cv.triangulatePoints(PL, PR, imgpointsL, imgpointsR)

points3D = points4D[0:3,:]/points4D[3,:]
print("Points:")
print(points3D.transpose())
print("Actual")
print(objpoints)
