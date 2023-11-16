from __future__ import print_function
from builtins import str
import sys, os, time
import numpy as np
import selectivesearch
from easydict import EasyDict
from fastRCNN.nms import nms as nmsPython
from builtins import range
import cv2, copy, textwrap
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
available_font = 'arial.ttf'
try:
    dummy = ImageFont.truetype(available_font, 16)
except:
    available_font = 'FreeMono.ttf'

def getSelectiveSearchRois(img, ssScale, ssSigma, ssMinSize, maxDim):
    if False:
        for i in range(10):
            print('nop')
    (img, scale) = imresizeMaxDim(img, maxDim, boUpscale=True, interpolation=cv2.INTER_AREA)
    (_, ssRois) = selectivesearch.selective_search(img, scale=ssScale, sigma=ssSigma, min_size=ssMinSize)
    rects = []
    for ssRoi in ssRois:
        (x, y, w, h) = ssRoi['rect']
        rects.append([x, y, x + w, y + h])
    return (rects, img, scale)

def getGridRois(imgWidth, imgHeight, nrGridScales, aspectRatios=[1.0]):
    if False:
        print('Hello World!')
    rects = []
    for iter in range(nrGridScales):
        cellWidth = 1.0 * min(imgHeight, imgWidth) / 2 ** iter
        step = cellWidth / 2.0
        for aspectRatio in aspectRatios:
            wStart = 0
            while wStart < imgWidth:
                hStart = 0
                while hStart < imgHeight:
                    if aspectRatio < 1:
                        wEnd = wStart + cellWidth
                        hEnd = hStart + cellWidth / aspectRatio
                    else:
                        wEnd = wStart + cellWidth * aspectRatio
                        hEnd = hStart + cellWidth
                    if wEnd < imgWidth - 1 and hEnd < imgHeight - 1:
                        rects.append([wStart, hStart, wEnd, hEnd])
                    hStart += step
                wStart += step
    return rects

def filterRois(rects, maxWidth, maxHeight, roi_minNrPixels, roi_maxNrPixels, roi_minDim, roi_maxDim, roi_maxAspectRatio):
    if False:
        return 10
    filteredRects = []
    filteredRectsSet = set()
    for rect in rects:
        if tuple(rect) in filteredRectsSet:
            continue
        (x, y, x2, y2) = rect
        w = x2 - x
        h = y2 - y
        assert w >= 0 and h >= 0
        if h == 0 or w == 0 or x2 > maxWidth or (y2 > maxHeight) or (w < roi_minDim) or (h < roi_minDim) or (w > roi_maxDim) or (h > roi_maxDim) or (w * h < roi_minNrPixels) or (w * h > roi_maxNrPixels) or (w / h > roi_maxAspectRatio) or (h / w > roi_maxAspectRatio):
            continue
        filteredRects.append(rect)
        filteredRectsSet.add(tuple(rect))
    assert len(filteredRects) > 0
    return filteredRects

def readRois(roiDir, subdir, imgFilename):
    if False:
        i = 10
        return i + 15
    roiPath = os.path.join(roiDir, subdir, imgFilename[:-4] + '.roi.txt')
    rois = np.loadtxt(roiPath, np.int)
    if len(rois) == 4 and type(rois[0]) == np.int32:
        rois = [rois]
    return rois

def readGtAnnotation(imgPath):
    if False:
        return 10
    bboxesPath = imgPath[:-4] + '.bboxes.tsv'
    labelsPath = imgPath[:-4] + '.bboxes.labels.tsv'
    bboxes = np.array(readTable(bboxesPath), np.int32)
    labels = readFile(labelsPath)
    assert len(bboxes) == len(labels)
    return (bboxes, labels)

def getCntkInputPaths(cntkFilesDir, image_set):
    if False:
        i = 10
        return i + 15
    cntkImgsListPath = os.path.join(cntkFilesDir, image_set + '.txt')
    cntkRoiCoordsPath = os.path.join(cntkFilesDir, image_set + '.rois.txt')
    cntkRoiLabelsPath = os.path.join(cntkFilesDir, image_set + '.roilabels.txt')
    cntkNrRoisPath = os.path.join(cntkFilesDir, image_set + '.nrRois.txt')
    return (cntkImgsListPath, cntkRoiCoordsPath, cntkRoiLabelsPath, cntkNrRoisPath)

def roiTransformPadScaleParams(imgWidth, imgHeight, padWidth, padHeight, boResizeImg=True):
    if False:
        print('Hello World!')
    scale = 1.0
    if boResizeImg:
        assert padWidth == padHeight, 'currently only supported equal width/height'
        scale = 1.0 * padWidth / max(imgWidth, imgHeight)
        imgWidth = round(imgWidth * scale)
        imgHeight = round(imgHeight * scale)
    targetw = padWidth
    targeth = padHeight
    w_offset = (targetw - imgWidth) / 2.0
    h_offset = (targeth - imgHeight) / 2.0
    if boResizeImg and w_offset > 0 and (h_offset > 0):
        print('ERROR: both offsets are > 0:', imgCounter, imgWidth, imgHeight, w_offset, h_offset)
        error
    if w_offset < 0 or h_offset < 0:
        print('ERROR: at least one offset is < 0:', imgWidth, imgHeight, w_offset, h_offset, scale)
    return (targetw, targeth, w_offset, h_offset, scale)

def roiTransformPadScale(rect, w_offset, h_offset, scale=1.0):
    if False:
        print('Hello World!')
    rect = [int(round(scale * d)) for d in rect]
    rect[0] += w_offset
    rect[1] += h_offset
    rect[2] += w_offset
    rect[3] += h_offset
    return rect

def getCntkRoiCoordsLine(rect, targetw, targeth):
    if False:
        i = 10
        return i + 15
    (x1, y1, x2, y2) = rect
    return ' {} {} {} {}'.format(x1, y1, x2, y2)

def getCntkRoiLabelsLine(overlaps, thres, nrClasses):
    if False:
        return 10
    maxgt = np.argmax(overlaps)
    if overlaps[maxgt] < thres:
        maxgt = 0
    oneHot = np.zeros(nrClasses, dtype=int)
    oneHot[maxgt] = 1
    oneHotString = ' {}'.format(' '.join((str(x) for x in oneHot)))
    return oneHotString

def cntkPadInputs(currentNrRois, targetNrRois, nrClasses, boxesStr, labelsStr):
    if False:
        return 10
    assert currentNrRois <= targetNrRois, 'Current number of rois ({}) should be <= target number of rois ({})'.format(currentNrRois, targetNrRois)
    while currentNrRois < targetNrRois:
        boxesStr += ' 0 0 0 0'
        labelsStr += ' 1' + ' 0' * (nrClasses - 1)
        currentNrRois += 1
    return (boxesStr, labelsStr)

def checkCntkOutputFile(cntkImgsListPath, cntkOutputPath, cntkNrRois, outputDim):
    if False:
        for i in range(10):
            print('nop')
    imgPaths = getColumn(readTable(cntkImgsListPath), 1)
    with open(cntkOutputPath) as fp:
        for imgIndex in range(len(imgPaths)):
            if imgIndex % 100 == 1:
                print('Checking cntk output file, image %d of %d...' % (imgIndex, len(imgPaths)))
            for roiIndex in range(cntkNrRois):
                assert fp.readline() != ''
        assert fp.readline() == ''

def parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, cntkNrRois, outputDim, saveCompressed=False, skipCheck=False, skip5Mod=None):
    if False:
        print('Hello World!')
    if not skipCheck and skip5Mod == None:
        checkCntkOutputFile(cntkImgsListPath, cntkOutputPath, cntkNrRois, outputDim)
    imgPaths = getColumn(readTable(cntkImgsListPath), 1)
    with open(cntkOutputPath) as fp:
        for imgIndex in range(len(imgPaths)):
            line = fp.readline()
            if skip5Mod != None and imgIndex % 5 != skip5Mod:
                print('Skipping image {} (skip5Mod = {})'.format(imgIndex, skip5Mod))
                continue
            print('Parsing cntk output file, image %d of %d' % (imgIndex, len(imgPaths)))
            data = []
            values = np.fromstring(line, dtype=float, sep=' ')
            assert len(values) == cntkNrRois * outputDim, 'ERROR: expected dimension of {} but found {}'.format(cntkNrRois * outputDim, len(values))
            for i in range(cntkNrRois):
                posStart = i * outputDim
                posEnd = posStart + outputDim
                currValues = values[posStart:posEnd]
                data.append(currValues)
            data = np.array(data, np.float32)
            outPath = os.path.join(outParsedDir, str(imgIndex) + '.dat')
            if saveCompressed:
                np.savez_compressed(outPath, data)
            else:
                np.savez(outPath, data)
        assert fp.readline() == ''

def readCntkRoiLabels(roiLabelsPath, nrRois, roiDim, stopAtImgIndex=None):
    if False:
        print('Hello World!')
    roiLabels = []
    for (imgIndex, line) in enumerate(readFile(roiLabelsPath)):
        if stopAtImgIndex and imgIndex == stopAtImgIndex:
            break
        roiLabels.append([])
        pos = line.find(b'|roiLabels ')
        valuesString = line[pos + 10:].strip().split(b' ')
        assert len(valuesString) == nrRois * roiDim
        for boxIndex in range(nrRois):
            oneHotLabels = [int(s) for s in valuesString[boxIndex * roiDim:(boxIndex + 1) * roiDim]]
            assert sum(oneHotLabels) == 1
            roiLabels[imgIndex].append(np.argmax(oneHotLabels))
    return roiLabels

def readCntkRoiCoordinates(imgPaths, cntkRoiCoordsPath, nrRois, padWidth, padHeight, stopAtImgIndex=None):
    if False:
        for i in range(10):
            print('nop')
    roiCoords = []
    for (imgIndex, line) in enumerate(readFile(cntkRoiCoordsPath)):
        if stopAtImgIndex and imgIndex == stopAtImgIndex:
            break
        roiCoords.append([])
        pos = line.find(b'|rois ')
        valuesString = line[pos + 5:].strip().split(b' ')
        assert len(valuesString) == nrRois * 4
        (imgWidth, imgHeight) = imWidthHeight(imgPaths[imgIndex])
        for boxIndex in range(nrRois):
            rect = [float(s) for s in valuesString[boxIndex * 4:(boxIndex + 1) * 4]]
            (x1, y1, x2, y2) = rect
            rect = getAbsoluteROICoordinates([x1, y1, x2, y2], imgWidth, imgHeight, padWidth, padHeight)
            roiCoords[imgIndex].append(rect)
    return roiCoords

def getAbsoluteROICoordinates(roi, imgWidth, imgHeight, padWidth, padHeight, resizeMethod='padScale'):
    if False:
        for i in range(10):
            print('nop')
    ' \n        The input image are usually padded to a fixed size, this method compute back the original \n        ROI absolute coordinate before the padding.\n    '
    if roi == [0, 0, 0, 0]:
        return [0, 0, 0, 0]
    if resizeMethod == 'pad' or resizeMethod == 'padScale':
        if resizeMethod == 'padScale':
            scale = float(padWidth) / max(imgWidth, imgHeight)
            imgWidthScaled = int(round(imgWidth * scale))
            imgHeightScaled = int(round(imgHeight * scale))
        else:
            scale = 1.0
            imgWidthScaled = imgWidth
            imgHeightScaled = imgHeight
        w_offset = float(padWidth - imgWidthScaled) / 2.0
        h_offset = float(padHeight - imgHeightScaled) / 2.0
        if resizeMethod == 'padScale':
            assert w_offset == 0 or h_offset == 0
        rect = [roi[0] - w_offset, roi[1] - h_offset, roi[2] - w_offset, roi[3] - h_offset]
        rect = [int(round(r / scale)) for r in rect]
    else:
        print("ERROR: Unknown resize method '%s'" % resizeMethod)
        error
    assert min(rect) >= 0 and max(rect[0], rect[2]) <= imgWidth and (max(rect[1], rect[3]) <= imgHeight)
    return rect

def getSvmModelPaths(svmDir, experimentName):
    if False:
        for i in range(10):
            print('nop')
    svmWeightsPath = '{}svmweights_{}.txt'.format(svmDir, experimentName)
    svmBiasPath = '{}svmbias_{}.txt'.format(svmDir, experimentName)
    svmFeatScalePath = '{}svmfeature_scale_{}.txt'.format(svmDir, experimentName)
    return (svmWeightsPath, svmBiasPath, svmFeatScalePath)

def loadSvm(svmDir, experimentName):
    if False:
        for i in range(10):
            print('nop')
    (svmWeightsPath, svmBiasPath, svmFeatScalePath) = getSvmModelPaths(svmDir, experimentName)
    svmWeights = np.loadtxt(svmWeightsPath, np.float32)
    svmBias = np.loadtxt(svmBiasPath, np.float32)
    svmFeatScale = np.loadtxt(svmFeatScalePath, np.float32)
    return (svmWeights, svmBias, svmFeatScale)

def saveSvm(svmDir, experimentName, svmWeights, svmBias, featureScale):
    if False:
        while True:
            i = 10
    (svmWeightsPath, svmBiasPath, svmFeatScalePath) = getSvmModelPaths(svmDir, experimentName)
    np.savetxt(svmWeightsPath, svmWeights)
    np.savetxt(svmBiasPath, svmBias)
    np.savetxt(svmFeatScalePath, featureScale)

def svmPredict(imgIndex, cntkOutputIndividualFilesDir, svmWeights, svmBias, svmFeatScale, roiSize, roiDim, decisionThreshold=0):
    if False:
        i = 10
        return i + 15
    cntkOutputPath = os.path.join(cntkOutputIndividualFilesDir, str(imgIndex) + '.dat.npz')
    data = np.load(cntkOutputPath)['arr_0']
    assert len(data) == roiSize
    labels = []
    maxScores = []
    for roiIndex in range(roiSize):
        feat = data[roiIndex]
        scores = np.dot(svmWeights, feat * 1.0 / svmFeatScale) + svmBias.ravel()
        assert len(scores) == roiDim
        maxArg = np.argmax(scores[1:]) + 1
        maxScore = scores[maxArg]
        if maxScore < decisionThreshold:
            maxArg = 0
        labels.append(maxArg)
        maxScores.append(maxScore)
    return (labels, maxScores)

def nnPredict(imgIndex, cntkParsedOutputDir, roiSize, roiDim, decisionThreshold=None):
    if False:
        return 10
    cntkOutputPath = os.path.join(cntkParsedOutputDir, str(imgIndex) + '.dat.npz')
    data = np.load(cntkOutputPath)['arr_0']
    assert len(data) == roiSize
    labels = []
    maxScores = []
    for roiIndex in range(roiSize):
        scores = data[roiIndex]
        scores = softmax(scores)
        assert len(scores) == roiDim
        maxArg = np.argmax(scores)
        maxScore = scores[maxArg]
        if decisionThreshold and maxScore < decisionThreshold:
            maxArg = 0
        labels.append(maxArg)
        maxScores.append(maxScore)
    return (labels, maxScores)

def imdbUpdateRoisWithHighGtOverlap(imdb, positivesGtOverlapThreshold):
    if False:
        return 10
    addedPosCounter = 0
    existingPosCounter = 0
    for imgIndex in range(imdb.num_images):
        for (boxIndex, gtLabel) in enumerate(imdb.roidb[imgIndex]['gt_classes']):
            if gtLabel > 0:
                existingPosCounter += 1
            else:
                overlaps = imdb.roidb[imgIndex]['gt_overlaps'][boxIndex, :].toarray()[0]
                maxInd = np.argmax(overlaps)
                maxOverlap = overlaps[maxInd]
                if maxOverlap >= positivesGtOverlapThreshold and maxInd > 0:
                    addedPosCounter += 1
                    imdb.roidb[imgIndex]['gt_classes'][boxIndex] = maxInd
    return (existingPosCounter, addedPosCounter)

def visualizeResults(imgPath, roiLabels, roiScores, roiRelCoords, padWidth, padHeight, classes, nmsKeepIndices=None, boDrawNegativeRois=True, decisionThreshold=0.0):
    if False:
        for i in range(10):
            print('nop')
    (imgWidth, imgHeight) = imWidthHeight(imgPath)
    scale = 800.0 / max(imgWidth, imgHeight)
    imgDebug = imresize(imread(imgPath), scale)
    assert len(roiLabels) == len(roiRelCoords)
    if roiScores:
        assert len(roiLabels) == len(roiScores)
    for iter in range(0, 3):
        for roiIndex in range(len(roiRelCoords)):
            label = roiLabels[roiIndex]
            if roiScores:
                score = roiScores[roiIndex]
                if decisionThreshold and score < decisionThreshold:
                    label = 0
            thickness = 1
            if label == 0:
                color = (255, 0, 0)
            else:
                color = getColorsPalette()[label]
            rect = [int(scale * i) for i in roiRelCoords[roiIndex]]
            if iter == 0 and boDrawNegativeRois:
                drawRectangles(imgDebug, [rect], color=color, thickness=thickness)
            elif iter == 1 and label > 0:
                if not nmsKeepIndices or roiIndex in nmsKeepIndices:
                    thickness = 4
                drawRectangles(imgDebug, [rect], color=color, thickness=thickness)
            elif iter == 2 and label > 0:
                if not nmsKeepIndices or roiIndex in nmsKeepIndices:
                    try:
                        font = ImageFont.truetype(available_font, 18)
                    except:
                        font = ImageFont.load_default()
                    text = classes[label]
                    if roiScores:
                        text += '(' + str(round(score, 2)) + ')'
                    imgDebug = drawText(imgDebug, (rect[0], rect[1]), text, color=(255, 255, 255), font=font, colorBackground=color)
    return imgDebug

def applyNonMaximaSuppression(nmsThreshold, labels, scores, coords, ignore_background=False):
    if False:
        while True:
            i = 10
    allIndices = []
    nmsRects = [[[]] for _ in range(max(labels) + 1)]
    coordsWithScores = np.hstack((coords, np.array([scores]).T))
    for i in range(max(labels) + 1):
        indices = np.where(np.array(labels) == i)[0]
        nmsRects[i][0] = coordsWithScores[indices, :]
        allIndices.append(indices)
    (_, nmsKeepIndicesList) = apply_nms(nmsRects, nmsThreshold, ignore_background=ignore_background)
    nmsKeepIndices = []
    for i in range(max(labels) + 1):
        for keepIndex in nmsKeepIndicesList[i][0]:
            nmsKeepIndices.append(allIndices[i][keepIndex])
    assert len(nmsKeepIndices) == len(set(nmsKeepIndices))
    return nmsKeepIndices

def apply_nms(all_boxes, thresh, ignore_background=False, boUsePythonImpl=True):
    if False:
        while True:
            i = 10
    'Apply non-maximum suppression to all predicted boxes output by the test_net method.'
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    nms_keepIndices = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        if ignore_background and cls_ind == 0:
            continue
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            if boUsePythonImpl:
                keep = nmsPython(dets, thresh)
            else:
                keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
            nms_keepIndices[cls_ind][im_ind] = keep
    return (nms_boxes, nms_keepIndices)

class DummyNet(object):

    def __init__(self, dim, num_classes, cntkParsedOutputDir):
        if False:
            i = 10
            return i + 15
        self.name = 'dummyNet'
        self.cntkParsedOutputDir = cntkParsedOutputDir
        self.params = {'cls_score': [EasyDict({'data': np.zeros((num_classes, dim), np.float32)}), EasyDict({'data': np.zeros((num_classes, 1), np.float32)})], 'trainers': None}

def im_detect(net, im, boxes, feature_scale=None, bboxIndices=None, boReturnClassifierScore=True, classifier='svm'):
    if False:
        for i in range(10):
            print('nop')
    cntkOutputPath = os.path.join(net.cntkParsedOutputDir, str(im) + '.dat.npz')
    cntkOutput = np.load(cntkOutputPath)['arr_0']
    if bboxIndices != None:
        cntkOutput = cntkOutput[bboxIndices, :]
    else:
        cntkOutput = cntkOutput[:len(boxes), :]
    scores = None
    if boReturnClassifierScore:
        if classifier == 'nn':
            scores = softmax2D(cntkOutput)
        elif classifier == 'svm':
            svmBias = net.params['cls_score'][1].data.transpose()
            svmWeights = net.params['cls_score'][0].data.transpose()
            scores = np.dot(cntkOutput * 1.0 / feature_scale, svmWeights) + svmBias
            assert np.unique(scores[:, 0]) == 0
        else:
            error
    return (scores, None, cntkOutput)

def makeDirectory(directory):
    if False:
        while True:
            i = 10
    if not os.path.exists(directory):
        os.makedirs(directory)

def getFilesInDirectory(directory, postfix=''):
    if False:
        for i in range(10):
            print('nop')
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(os.path.join(directory, s))]
    if not postfix or postfix == '':
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def readFile(inputFile):
    if False:
        while True:
            i = 10
    with open(inputFile, 'rb') as f:
        lines = f.readlines()
    return [removeLineEndCharacters(s) for s in lines]

def readTable(inputFile, delimiter='\t', columnsToKeep=None):
    if False:
        i = 10
        return i + 15
    lines = readFile(inputFile)
    if columnsToKeep != None:
        header = lines[0].split(delimiter)
        columnsToKeepIndices = listFindItems(header, columnsToKeep)
    else:
        columnsToKeepIndices = None
    return splitStrings(lines, delimiter, columnsToKeepIndices)

def getColumn(table, columnIndex):
    if False:
        i = 10
        return i + 15
    column = []
    for row in table:
        column.append(row[columnIndex])
    return column

def deleteFile(filePath):
    if False:
        print('Hello World!')
    if os.path.exists(filePath):
        os.remove(filePath)

def writeFile(outputFile, lines):
    if False:
        print('Hello World!')
    with open(outputFile, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)

def writeTable(outputFile, table):
    if False:
        print('Hello World!')
    lines = tableToList1D(table)
    writeFile(outputFile, lines)

def deleteFile(filePath):
    if False:
        while True:
            i = 10
    if os.path.exists(filePath):
        os.remove(filePath)

def deleteAllFilesInDirectory(directory, fileEndswithString, boPromptUser=False):
    if False:
        return 10
    if boPromptUser:
        userInput = raw_input('--> INPUT: Press "y" to delete files in directory ' + directory + ': ')
        if not (userInput.lower() == 'y' or userInput.lower() == 'yes'):
            print('User input is %s: exiting now.' % userInput)
            exit()
    for filename in getFilesInDirectory(directory):
        if fileEndswithString == None or filename.lower().endswith(fileEndswithString):
            deleteFile(os.path.join(directory, filename))

def removeLineEndCharacters(line):
    if False:
        for i in range(10):
            print('nop')
    if line.endswith(b'\r\n'):
        return line[:-2]
    elif line.endswith(b'\n'):
        return line[:-1]
    else:
        return line

def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if False:
        i = 10
        return i + 15
    if string == None:
        return None
    items = string.decode('utf-8').split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumns([items], columnsToKeepIndices)
        items = items[0]
    return items

def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    if False:
        print('Hello World!')
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table

def find(list1D, func):
    if False:
        return 10
    return [index for (index, item) in enumerate(list1D) if func(item)]

def tableToList1D(table, delimiter='\t'):
    if False:
        return 10
    return [delimiter.join([str(s) for s in row]) for row in table]

def sortDictionary(dictionary, sortIndex=0, reverseSort=False):
    if False:
        print('Hello World!')
    return sorted(dictionary.items(), key=lambda x: x[sortIndex], reverse=reverseSort)

def imread(imgPath, boThrowErrorIfExifRotationTagSet=True):
    if False:
        while True:
            i = 10
    if not os.path.exists(imgPath):
        print('ERROR: image path does not exist.')
        error
    rotation = rotationFromExifTag(imgPath)
    if boThrowErrorIfExifRotationTagSet and rotation != 0:
        print('Error: exif roation tag set, image needs to be rotated by %d degrees.' % rotation)
    img = cv2.imread(imgPath)
    if img is None:
        print('ERROR: cannot load image ' + imgPath)
        error
    if rotation != 0:
        img = imrotate(img, -90).copy()
    return img

def rotationFromExifTag(imgPath):
    if False:
        print('Hello World!')
    TAGSinverted = {v: k for (k, v) in TAGS.items()}
    orientationExifId = TAGSinverted['Orientation']
    try:
        imageExifTags = Image.open(imgPath)._getexif()
    except:
        imageExifTags = None
    rotation = 0
    if imageExifTags != None and orientationExifId != None and (orientationExifId in imageExifTags):
        orientation = imageExifTags[orientationExifId]
        if orientation == 1 or orientation == 0:
            rotation = 0
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            print('ERROR: orientation = ' + str(orientation) + ' not_supported!')
            error
    return rotation

def imwrite(img, imgPath):
    if False:
        return 10
    cv2.imwrite(imgPath, img)

def imresize(img, scale, interpolation=cv2.INTER_LINEAR):
    if False:
        print('Hello World!')
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)

def imresizeMaxDim(img, maxDim, boUpscale=False, interpolation=cv2.INTER_LINEAR):
    if False:
        while True:
            i = 10
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1 or boUpscale:
        img = imresize(img, scale, interpolation)
    else:
        scale = 1.0
    return (img, scale)

def imWidth(input):
    if False:
        while True:
            i = 10
    return imWidthHeight(input)[0]

def imHeight(input):
    if False:
        while True:
            i = 10
    return imWidthHeight(input)[1]

def imWidthHeight(input):
    if False:
        for i in range(10):
            print('nop')
    (width, height) = Image.open(input).size
    return (width, height)

def imArrayWidth(input):
    if False:
        i = 10
        return i + 15
    return imArrayWidthHeight(input)[0]

def imArrayHeight(input):
    if False:
        return 10
    return imArrayWidthHeight(input)[1]

def imArrayWidthHeight(input):
    if False:
        print('Hello World!')
    width = input.shape[1]
    height = input.shape[0]
    return (width, height)

def imshow(img, waitDuration=0, maxDim=None, windowName='img'):
    if False:
        while True:
            i = 10
    if isinstance(img, str):
        img = cv2.imread(img)
    if maxDim is not None:
        scaleVal = 1.0 * maxDim / max(img.shape[:2])
        if scaleVal < 1:
            img = imresize(img, scaleVal)
    cv2.imshow(windowName, img)
    cv2.waitKey(waitDuration)

def drawRectangles(img, rects, color=(0, 255, 0), thickness=2):
    if False:
        for i in range(10):
            print('nop')
    for rect in rects:
        pt1 = tuple(ToIntegers(rect[0:2]))
        pt2 = tuple(ToIntegers(rect[2:]))
        cv2.rectangle(img, pt1, pt2, color, thickness)

def drawCrossbar(img, pt):
    if False:
        i = 10
        return i + 15
    (x, y) = pt
    cv2.rectangle(img, (0, y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, 0), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (img.shape[1], y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, img.shape[0]), (x, y), (255, 255, 0), 1)

def ptClip(pt, maxWidth, maxHeight):
    if False:
        print('Hello World!')
    pt = list(pt)
    pt[0] = max(pt[0], 0)
    pt[1] = max(pt[1], 0)
    pt[0] = min(pt[0], maxWidth)
    pt[1] = min(pt[1], maxHeight)
    return pt

def drawText(img, pt, text, textWidth=None, color=(255, 255, 255), colorBackground=None, font=None):
    if False:
        for i in range(10):
            print('nop')
    if font == None:
        font = ImageFont.truetype('arial.ttf', 16)
    pilImg = imconvertCv2Pil(img)
    pilImg = pilDrawText(pilImg, pt, text, textWidth, color, colorBackground, font)
    return imconvertPil2Cv(pilImg)

def pilDrawText(pilImg, pt, text, textWidth=None, color=(255, 255, 255), colorBackground=None, font=None):
    if False:
        i = 10
        return i + 15
    if font == None:
        font = ImageFont.truetype('arial.ttf', 16)
    textY = pt[1]
    draw = ImageDraw.Draw(pilImg)
    if textWidth == None:
        lines = [text]
    else:
        lines = textwrap.wrap(text, width=textWidth)
    for line in lines:
        (width, height) = font.getsize(line)
        if colorBackground != None:
            draw.rectangle((pt[0], pt[1], pt[0] + width, pt[1] + height), fill=tuple(colorBackground[::-1]))
        draw.text(pt, line, fill=tuple(color), font=font)
        textY += height
    return pilImg

def getColorsPalette():
    if False:
        for i in range(10):
            print('nop')
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
    for i in range(5):
        for dim in range(0, 3):
            for s in (0.25, 0.5, 0.75):
                if colors[i][dim] != 0:
                    newColor = copy.deepcopy(colors[i])
                    newColor[dim] = int(round(newColor[dim] * s))
                    colors.append(newColor)
    return colors

def imconvertPil2Cv(pilImg):
    if False:
        return 10
    rgb = pilImg.convert('RGB')
    return np.array(rgb).copy()[:, :, ::-1]

def imconvertCv2Pil(img):
    if False:
        for i in range(10):
            print('nop')
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)

def ToIntegers(list1D):
    if False:
        return 10
    return [int(float(x)) for x in list1D]

def softmax(vec):
    if False:
        while True:
            i = 10
    expVec = np.exp(vec)
    if max(expVec) == np.inf:
        outVec = np.zeros(len(expVec))
        outVec[expVec == np.inf] = vec[expVec == np.inf]
        outVec = outVec / np.sum(outVec)
    else:
        outVec = expVec / np.sum(expVec)
    return outVec

def softmax2D(w):
    if False:
        print('Hello World!')
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)[:, np.newaxis]
    return dist

def getDictionary(keys, values, boConvertValueToInt=True):
    if False:
        while True:
            i = 10
    dictionary = {}
    for (key, value) in zip(keys, values):
        if boConvertValueToInt:
            value = int(value)
        dictionary[key] = value
    return dictionary

class Bbox:
    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left, top, right, bottom):
        if False:
            i = 10
            return i + 15
        self.left = int(round(float(left)))
        self.top = int(round(float(top)))
        self.right = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        if False:
            return 10
        return 'Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}'.format(self.left, self.top, self.right, self.bottom)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str(self)

    def rect(self):
        if False:
            return 10
        return [self.left, self.top, self.right, self.bottom]

    def max(self):
        if False:
            return 10
        return max([self.left, self.top, self.right, self.bottom])

    def min(self):
        if False:
            return 10
        return min([self.left, self.top, self.right, self.bottom])

    def width(self):
        if False:
            for i in range(10):
                print('nop')
        width = self.right - self.left + 1
        assert width >= 0
        return width

    def height(self):
        if False:
            for i in range(10):
                print('nop')
        height = self.bottom - self.top + 1
        assert height >= 0
        return height

    def surfaceArea(self):
        if False:
            for i in range(10):
                print('nop')
        return self.width() * self.height()

    def getOverlapBbox(self, bbox):
        if False:
            return 10
        (left1, top1, right1, bottom1) = self.rect()
        (left2, top2, right2, bottom2) = bbox.rect()
        overlapLeft = max(left1, left2)
        overlapTop = max(top1, top2)
        overlapRight = min(right1, right2)
        overlapBottom = min(bottom1, bottom2)
        if overlapLeft > overlapRight or overlapTop > overlapBottom:
            return None
        else:
            return Bbox(overlapLeft, overlapTop, overlapRight, overlapBottom)

    def standardize(self):
        if False:
            return 10
        leftNew = min(self.left, self.right)
        topNew = min(self.top, self.bottom)
        rightNew = max(self.left, self.right)
        bottomNew = max(self.top, self.bottom)
        self.left = leftNew
        self.top = topNew
        self.right = rightNew
        self.bottom = bottomNew

    def crop(self, maxWidth, maxHeight):
        if False:
            while True:
                i = 10
        leftNew = min(max(self.left, 0), maxWidth)
        topNew = min(max(self.top, 0), maxHeight)
        rightNew = min(max(self.right, 0), maxWidth)
        bottomNew = min(max(self.bottom, 0), maxHeight)
        return Bbox(leftNew, topNew, rightNew, bottomNew)

    def isValid(self):
        if False:
            i = 10
            return i + 15
        if self.left >= self.right or self.top >= self.bottom:
            return False
        if min(self.rect()) < -self.MAX_VALID_DIM or max(self.rect()) > self.MAX_VALID_DIM:
            return False
        return True

def getEnclosingBbox(pts):
    if False:
        return 10
    left = top = float('inf')
    right = bottom = float('-inf')
    for pt in pts:
        left = min(left, pt[0])
        top = min(top, pt[1])
        right = max(right, pt[0])
        bottom = max(bottom, pt[1])
    return Bbox(left, top, right, bottom)

def bboxComputeOverlapVoc(bbox1, bbox2):
    if False:
        i = 10
        return i + 15
    surfaceRect1 = bbox1.surfaceArea()
    surfaceRect2 = bbox2.surfaceArea()
    overlapBbox = bbox1.getOverlapBbox(bbox2)
    if overlapBbox == None:
        return 0
    else:
        surfaceOverlap = overlapBbox.surfaceArea()
        overlap = max(0, 1.0 * surfaceOverlap / (surfaceRect1 + surfaceRect2 - surfaceOverlap))
        assert overlap >= 0 and overlap <= 1
        return overlap

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    if False:
        return 10
    ' ap = voc_ap(recalls, precisions, [use_07_metric])\n    Compute VOC AP given precision and recall.\n    If use_07_metric is true, uses the\n    VOC 07 11 point method (default:False).\n    '
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.0
    else:
        mrecalls = np.concatenate(([0.0], recalls, [1.0]))
        mprecisions = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap