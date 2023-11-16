from cmath import inf
from random import randint, random
import cv2 as cv
import numpy as np
from numpy import infty
mtxL = np.array([[1212.93155, 0.0, 1012.07716], [0.0, 368.235956, 775.790311], [0.0, 0.0, 1.0]])
dstL = np.array([[-0.13071266, 0.49034763, -0.02996941, 0.00285484, -0.64545585]])
mtxR = np.array([[1314.75299, 0.0, 965.652771], [0.0, 718.441885, 920.397684], [0.0, 0.0, 1.0]])
dstR = np.array([[-0.102831315, 0.281378194, -0.0273569525, -0.00025307228, -0.33761572]])
E = np.array([[-0.0235291, -0.22562236, -0.99232161], [0.34649048, -0.02820199, 0.2337239], [0.95854388, -0.24072778, -0.05166317]])
F = np.array([[-9.32777768e-07, -2.94621915e-05, -0.0239152011], [2.51371848e-05, -6.73930541e-06, 0.000354223161], [0.0277252851, -0.00667575866, 1.0]])
PL = np.array([[1212.93155, 0.0, 1012.07716, 0.0], [0.0, 368.235956, 775.790311, 0.0], [0.0, 0.0, 1.0, 0.0]])
PR = np.array([[1337.53465, -111.813249, 927.124358, 8.68619259], [23.1777852, 605.261262, 998.204505, -375.50504], [0.0239852651, -0.11755625, 0.992776528, 0.340744625]])
RESIZE = 2
PAUSEFIRSTX = 10
LINE = True

def getOutputsNames(net):
    if False:
        print('Hello World!')
    layersNames = net.getLayerNames()
    check = net.getUnconnectedOutLayers().tolist()
    return [layersNames[i - 1] for i in check]

def drawPred(image, classId, conf, left, top, right, bottom):
    if False:
        i = 10
        return i + 15
    cv.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert classId < len(classes)
        label = '%s:%s' % (classes[classId], label)
    (labelSize, baseLine) = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(image, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

def postprocess(image, outs):
    if False:
        return 10
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
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
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    oBoxes = []
    oClasses = []
    for i in indices:
        box = boxes[i]
        oBoxes.append(box)
        oClasses.append(classIds[i])
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(image, classIds[i], confidences[i], left, top, left + width, top + height)
    return (oBoxes, oClasses)
for img_id in range(1, 30):
    right_image = cv.imread('right/' + str(img_id) + '.png')
    left_image = cv.imread('left/' + str(img_id) + '.png')
    right_image_og = right_image
    left_image_og = left_image
    right_image_gray = cv.cvtColor(right_image, cv.COLOR_RGB2GRAY)
    left_image_gray = cv.cvtColor(left_image, cv.COLOR_RGB2GRAY)
    size = left_image_gray.shape
    (h, w) = left_image.shape[:2]
    confThreshold = 0.75
    nmsThreshold = 0.75
    inpWidth = 832
    inpHeight = 832
    classesFile = 'cones.names'
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    modelConfiguration = 'yolov4-tiny-cones.cfg'
    modelWeights = 'yolov4-tiny-cones_best.weights'
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    blobR = cv.dnn.blobFromImage(right_image, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    print(blobR)
    net.setInput(blobR)
    outsR = net.forward(getOutputsNames(net))
    (boxesR, classR) = postprocess(right_image, outsR)
    (t, _) = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(right_image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    blobL = cv.dnn.blobFromImage(left_image, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blobL)
    outsL = net.forward(getOutputsNames(net))
    (boxesL, classL) = postprocess(left_image, outsL)
    (t, _) = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(left_image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    centerL = []
    classLnew = []
    for l in range(len(boxesL)):
        box = boxesL[l]
        x = box[0] + box[2] / 2
        y = box[1] + box[3] / 2
        if y > 600:
            try:
                ind = centerL.index([x, y])
            except ValueError:
                centerL.append([x, y])
                classLnew.append(classL[l])
        else:
            print('Not Added!')
    centerL = np.array(centerL)
    classL = classLnew
    print(centerL)
    centerR = []
    classRnew = []
    for r in range(len(boxesR)):
        box = boxesR[r]
        x = box[0] + box[2] / 2
        y = box[1] + box[3] / 2
        if y > 600:
            try:
                ind = centerR.index([x, y])
            except ValueError:
                centerR.append([x, y])
                classRnew.append(classR[r])
        else:
            print('Not Added!')
    centerR = np.array(centerR)
    classR = classRnew
    print(centerR)

    def world_from_L(u, v):
        if False:
            print('Hello World!')
        u = (u - 1030) / 452.9
        v = (v - 725.9) / 76.41
        y = -0.06899 - 1.705 * u + 0.09506 * v + 0.06137 * u ** 2 + 0.7549 * u * v - 0.05601 * v ** 2 + 0.01561 * u ** 3 - 0.05147 * v * u ** 2 - 0.1594 * u * v ** 2 + 0.01651 * v ** 3
        x = 4.861 - 0.2278 * u - 2.396 * v - 0.05831 * u ** 2 + 0.1824 * u * v + 0.9738 * v ** 2 + 0.01231 * u ** 3 + 0.02113 * v * u ** 2 - 0.05309 * u * v ** 2 - 0.1779 * v ** 3
        return (x, y)

    def world_from_R(u, v):
        if False:
            while True:
                i = 10
        u = (u - 881.8) / 452.7
        v = (v - 764.5) / 77.06
        y = -0.04364 - 1.71 * u - 0.006886 * v + 0.06176 * u ** 2 + 0.7675 * u * v - 0.01835 * v ** 2 + 0.0171 * u ** 3 - 0.04484 * v * u ** 2 - 0.1662 * u * v ** 2 + 0.007698 * v ** 3
        x = 4.835 - 0.292 * u - 2.424 * v - 0.05401 * u ** 2 + 0.1824 * u * v + 1.043 * v ** 2 + 0.007452 * u ** 3 + 0.03166 * v * u ** 2 - 0.04845 * u * v ** 2 - 0.201 * v ** 3
        return (x, y)
    all_cost = np.ones((len(centerR), len(centerL))) * inf
    for r in range(len(centerR)):
        for l in range(len(centerL)):
            if classR[r] == classL[l]:
                (xL, yL) = world_from_L(centerL[l][0], centerL[l][1])
                (xR, yR) = world_from_R(centerR[r][0], centerR[r][1])
                cost = abs(xL - xR) + 2 * abs(yL - yR)
                if abs(yL - yR) > 0.7:
                    cost = cost * 1000
                all_cost[r][l] = cost
    print(np.round(all_cost, 3))
    pairs = []
    for r in range(len(centerR)):
        for l in range(len(centerL)):
            if min(all_cost[r, :]) == all_cost[r, l] and min(all_cost[:, l]) == all_cost[r, l]:
                pairs.append((r, l))
    print(pairs)
    sortL = []
    sortR = []
    for (r, l) in pairs:
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
    for (pointL, pointR, cl, pair) in zip(centerL, centerR, classR, pairs):
        (xL, yL) = pointL
        (xR, yR) = pointR
        if cl == 0:
            color = (100 + randint(0, 150), 0, 0)
        elif cl == 1:
            color = (0, 50 + randint(0, 200), 0)
        elif cl == 2:
            rn = 50 + randint(0, 200)
            color = (0, rn, rn)
        else:
            color = (0, 0, 0)
        image = cv.circle(image, (w + int(xR), int(yR)), 5, color, -1)
        cv.putText(image, str(pair[0]) + ',' + str(pair[1]), (w + int(xR), int(yR)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        image = cv.circle(image, (int(xL), int(yL)), 5, color, -1)
        cv.putText(image, str(pair[0]) + ',' + str(pair[1]), (int(xL), int(yL)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        if LINE:
            image = cv.line(image, (int(xL), int(yL)), (w + int(xR), int(yR)), color, RESIZE)
    (h, w) = image.shape[:2]
    image = cv.resize(image, (int(w / RESIZE), int(h / RESIZE)))
    cv.imshow('Both', image)
    if img_id <= PAUSEFIRSTX:
        cv.waitKey(0)
    else:
        cv.waitKey(500)
    points = []
    for i in range(len(centerL)):
        (xL, yL) = world_from_L(centerL[i][0], centerL[i][1])
        (xR, yR) = world_from_R(centerR[i][0], centerR[i][1])
        x = (xL + xR) / 2
        y = (yL + yR) / 2
        print(pairs[i])
        print(xL, yL)
        print(xR, yR)
        print(x, y)
        points.append([x, y])