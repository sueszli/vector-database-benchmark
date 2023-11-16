
from cmath import inf
from random import randint, random
import cv2 as cv
import numpy as np
from numpy import infty

# Camera Parameters (Left is camera 1)
mtxL = np.array([[1.21293155e+03, 0.00000000e+00, 1.01207716e+03],
 [0.00000000e+00, 3.68235956e+02, 7.75790311e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dstL = np.array([[-0.13071266,  0.49034763, -0.02996941,  0.00285484, -0.64545585]])
mtxR = np.array([[1.31475299e+03, 0.00000000e+00, 9.65652771e+02],
 [0.00000000e+00, 7.18441885e+02, 9.20397684e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dstR = np.array([[-1.02831315e-01,  2.81378194e-01, -2.73569525e-02, -2.53072280e-04, -3.37615720e-01]])

E = np.array([[-0.0235291,  -0.22562236, -0.99232161],
 [ 0.34649048, -0.02820199,  0.2337239 ],
 [ 0.95854388, -0.24072778, -0.05166317]])
F = np.array([[-9.32777768e-07, -2.94621915e-05, -2.39152011e-02],
 [ 2.51371848e-05, -6.73930541e-06,  3.54223161e-04],
 [ 2.77252851e-02, -6.67575866e-03,  1.00000000e+00]])

# Projection Matrices
PL = np.array([[1.21293155e+03, 0.00000000e+00, 1.01207716e+03, 0.00000000e+00],
 [0.00000000e+00, 3.68235956e+02, 7.75790311e+02, 0.00000000e+00],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])

PR = np.array([[ 1.33753465e+03, -1.11813249e+02,  9.27124358e+02,  8.68619259e+00],
 [ 2.31777852e+01,  6.05261262e+02,  9.98204505e+02, -3.75505040e+02],
 [ 2.39852651e-02, -1.17556250e-01,  9.92776528e-01,  3.40744625e-01]])


RESIZE = 2
PAUSEFIRSTX = 10
LINE = True

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    check = net.getUnconnectedOutLayers().tolist()
    # If error switch line
    # return [layersNames[i - 1] for i in check]
    return [layersNames[i - 1] for i in check]

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

for img_id in range(1, 30):
    right_image = cv.imread("right/"  + str(img_id) + ".png")
    left_image = cv.imread("left/" + str(img_id) + ".png")
    right_image_og = right_image
    left_image_og = left_image
    right_image_gray = cv.cvtColor(right_image, cv.COLOR_RGB2GRAY)
    left_image_gray = cv.cvtColor(left_image, cv.COLOR_RGB2GRAY)
    size = left_image_gray.shape

    h,  w = left_image.shape[:2]

    # cv.imshow("Blue only", left_image[:,:,0])
    # cv.waitKey(0)
    # cv.imshow("right", right_image)
    # cv.waitKey(1000)
    # cv.imshow("left", left_image)
    # cv.waitKey(1000)

    confThreshold = 0.75  #Confidence threshold
    nmsThreshold = 0.75   #Non-maximum suppression threshold
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

    # Create a 4D blob from a frame.
    blobR = cv.dnn.blobFromImage(right_image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    print(blobR)
    # Sets the input to the network
    net.setInput(blobR)

    # Runs the forward pass to get output of the output layers
    outsR = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    boxesR, classR = postprocess(right_image, outsR)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(right_image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Create a 4D blob from a frame.
    blobL = cv.dnn.blobFromImage(left_image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blobL)

    # Runs the forward pass to get output of the output layers
    outsL = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    boxesL, classL = postprocess(left_image, outsL)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(left_image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))



    # TODO: Find Matches between found cones
    # PROBLEM: Fundamental matrix is not great - giving bad matches = poor stereo calibration
    centerL = []
    classLnew = []
    for l in range(len(boxesL)):
        box = boxesL[l]
        x = box[0] + box[2]/2
        y = box[1] + box[3]/2
        if y > 600:
            try:
                ind = centerL.index([x, y])
            except ValueError:
                centerL.append([x, y])
                classLnew.append(classL[l])
        else:
            print("Not Added!")
    centerL = np.array(centerL)
    classL = classLnew
    print(centerL)

    centerR = []
    classRnew = []
    for r in range(len(boxesR)):
        box = boxesR[r]
        x = box[0] + box[2]/2
        y = box[1] + box[3]/2
        if y > 600:
            try:
                ind = centerR.index([x, y])
            except ValueError:
                centerR.append([x, y])
                classRnew.append(classR[r])
        else:
            print("Not Added!")
    centerR = np.array(centerR)
    classR = classRnew
    print(centerR)
    # centerL = cv.undistortImagePoints(np.array(centerL), mtxL, dstL)
    # for point, cl in zip(centerL, classL):
    #     x, y = point
    #     if cl == 0:
    #         color = (255, 0, 0)
    #     elif cl == 1:
    #         color = (0, 255, 0)
    #     elif cl == 2:
    #         color = (0, 255, 255)
    #     else:
    #         color = (0, 0, 0)
    #     left_image = cv.circle(left_image,(int(x),int(y)),5,color,-1)




    #  
    # for point, cl in zip(centerR, classR):
    #     x, y = point
    #     if cl == 0:
    #         color = (255, 0, 0)
    #     elif cl == 1:
    #         color = (0, 255, 0)
    #     elif cl == 2:
    #         color = (0, 255, 255)
    #     else:
    #         color = (0, 0, 0)
    #     right_image = cv.circle(right_image,(int(x),int(y)),5,color,-1)




    # Collect all distances
    # all_cost = np.ones((len(centerR), len(linesR)))*inf
    # for r in range(len(centerR)):
    #     for l in range(len(linesR)):
    #         if classR[r] == classL[l]:
    #             point = np.array([centerR[r][0], centerR[r][1], 1])
    #             line = linesR[l][0]
    #             line = np.array([line[0], line[1], line[2]])
    #             boxL = boxesL[l]
    #             boxR = boxesR[r]
    #             cost = (1-a)*(abs(boxL[2] - boxR[2]) + abs(boxL[3] - boxL[3])) + a*abs(np.dot(point, line))
    #             all_cost[r][l] = cost
    def world_from_L(u, v):
        u = (u - 1030)/452.9
        v = (v - 725.9)/76.41
        y = -0.06899 - 1.705*u + 0.09506*v + 0.06137*u**2 + 0.7549*u*v - 0.05601*v**2 + 0.01561*u**3 - 0.05147*v*(u**2) - 0.1594*u*(v**2) + 0.01651*v**3
        x = 4.861 - 0.2278*u - 2.396*v - 0.05831*u**2 + 0.1824*u*v + 0.9738*v**2 + 0.01231*u**3 + 0.02113*v*(u**2) - 0.05309*u*(v**2) - 0.1779*v**3
        return x, y
    
    
    def world_from_R(u, v):
        u = (u - 881.8)/452.7
        v = (v - 764.5)/77.06
        y = -0.04364 -1.71*u -0.006886*v + 0.06176*u**2 + 0.7675*u*v -0.01835*v**2 + 0.0171*u**3 -0.04484*v*(u**2) -0.1662*u*(v**2) + 0.007698*v**3
        x = 4.835 -0.292*u -2.424*v -0.05401*u**2 + 0.1824*u*v + 1.043*v**2 + 0.007452*u**3 + 0.03166*v*(u**2) -0.04845*u*(v**2)  -0.201*v**3
        return x, y

    all_cost = np.ones((len(centerR), len(centerL)))*inf
    for r in range(len(centerR)):
        for l in range(len(centerL)):
            if classR[r] == classL[l]:
                # yR_from_L = 1.008*centerL[l][1] + 33.06
                # AreaR_from_L = 1.033*boxesL[l][2]*boxesL[l][3] - 226.1
                # distL = 376.5*(boxesL[l][3]**-0.9404)
                # distR = 406.8*(boxesR[r][3]**-0.9597)
                # abs(distL - distR)
                # xRfL = 115.7 + 0.9977*centerL[l][0] - 0.3607*centerL[l][1]
                # yRfL = 38.3  - 0.005147*centerL[l][0] + 1.008*centerL[l][1]
                # right_image = cv.circle(right_image,(int(xRfL),int(yRfL)),3,(0,0,0),-1)
                xL, yL = world_from_L(centerL[l][0], centerL[l][1])
                xR, yR = world_from_R(centerR[r][0], centerR[r][1])
                cost = abs(xL - xR) + 2*abs(yL - yR)
                if abs(yL - yR) > 0.7:
                    cost = cost*1000
                all_cost[r][l] =  cost


    print(np.round(all_cost, 3))
    # Pair up cones from left and right image based on distance
    pairs = []
    for r in range(len(centerR)):
        for l in range(len(centerL)):
            if min(all_cost[r, :]) == all_cost[r, l] and min(all_cost[:, l]) == all_cost[r, l]:
                pairs.append((r, l))

    print(pairs)

    # Select only the right points
    sortL = []
    sortR = []
    for r, l in pairs:
        sortR.append(r)
        sortL.append(l)


    centerL = centerL[sortL]
    classL = np.array(classL)
    classL = classL[sortL]
    centerR = centerR[sortR]
    classR = np.array(classR)
    classR = classR[sortR]
    print(classR)

    image = np.concatenate((left_image, right_image), axis=1)
    
    for pointL, pointR, cl, pair in zip(centerL, centerR, classR, pairs):
        xL, yL = pointL
        xR, yR = pointR
        if cl == 0:
            color = (100 + randint(0, 150), 0, 0)
        elif cl == 1:
            color = (0, 50 + randint(0, 200), 0)
        elif cl == 2:
            rn = 50 + randint(0, 200)
            color = (0, rn, rn)
        else:
            color = (0, 0, 0)
        image = cv.circle(image,(w + int(xR),int(yR)),5,color,-1)
        cv.putText(image,str(pair[0]) + "," + str(pair[1]), (w + int(xR),int(yR)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        image = cv.circle(image,(int(xL),int(yL)),5,color,-1)
        cv.putText(image,str(pair[0]) + "," + str(pair[1]), (int(xL),int(yL)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        if LINE:
            image = cv.line(image, (int(xL), int(yL)), (w + int(xR), int(yR)), color, RESIZE)


    h,  w = image.shape[:2]

    image = cv.resize(image, (int(w/RESIZE), int(h/RESIZE)))
    
    cv.imshow("Both", image)
    if img_id <= PAUSEFIRSTX:
        cv.waitKey(0)
    else:
        cv.waitKey(500)
        
    # TODO: Once Matches have been found compute cone locations
    # triangulate points
    # points4D = cv.triangulatePoints(PL, PR, centerL.transpose(), centerR.transpose())

    # points3D = points4D[0:3,:]/points4D[3,:]
    # print("Points:")
    # print(points3D)
    # print("Dist:")
    # print(np.sqrt(points3D[0]*points3D[0] + points3D[1]*points3D[1] + points3D[2]*points3D[2]))

    # Using mapping 
    points = []
    for i in range(len(centerL)):
        xL, yL = world_from_L(centerL[i][0], centerL[i][1])
        xR, yR = world_from_R(centerR[i][0], centerR[i][1])
        x = (xL + xR)/2
        y = (yL + yR)/2
        print(pairs[i])
        print(xL, yL)
        print(xR, yR)
        print(x, y)
        points.append([x, y])
    
    

    # f = open("cones.csv", "a")
    # for x, y, z in points3D.transpose():
    #     f.write(str(x) + "," + str(y) + "," + str(z) + "\n")

    # f.close()

 
