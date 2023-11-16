from __future__ import print_function
import os, importlib, sys
from cntk_helpers import *
import PARAMETERS
image_set = 'train'
parseNrImages = 10
boUseNonMaximaSurpression = False
nmsThreshold = 0.1

def generate_rois_visualization(testing=False):
    if False:
        print('Hello World!')
    p = PARAMETERS.get_parameters_for_dataset()
    print('Load ROI co-ordinates and labels')
    (cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, nrRoisPath) = getCntkInputPaths(p.cntkFilesDir, image_set)
    imgPaths = getColumn(readTable(cntkImgsPath), 1)
    nrRealRois = [int(s) for s in readFile(nrRoisPath)]
    roiAllLabels = readCntkRoiLabels(cntkRoiLabelsPath, p.cntk_nrRois, len(p.classes), parseNrImages)
    if parseNrImages:
        imgPaths = imgPaths[:parseNrImages]
        nrRealRois = nrRealRois[:parseNrImages]
        roiAllLabels = roiAllLabels[:parseNrImages]
    roiAllCoords = readCntkRoiCoordinates(imgPaths, cntkRoiCoordsPath, p.cntk_nrRois, p.cntk_padWidth, p.cntk_padHeight, parseNrImages)
    assert len(imgPaths) == len(roiAllCoords) == len(roiAllLabels) == len(nrRealRois)
    for (imgIndex, imgPath) in enumerate(imgPaths):
        print('Visualizing image %d at %s...' % (imgIndex, imgPath))
        roiCoords = roiAllCoords[imgIndex][:nrRealRois[imgIndex]]
        roiLabels = roiAllLabels[imgIndex][:nrRealRois[imgIndex]]
        nmsKeepIndices = []
        if boUseNonMaximaSurpression:
            (imgWidth, imgHeight) = imWidthHeight(imgPath)
            nmsKeepIndices = applyNonMaximaSuppression(nmsThreshold, roiLabels, [0] * len(roiLabels), roiCoords)
            print('Non-maxima surpression kept {} of {} rois (nmsThreshold={})'.format(len(nmsKeepIndices), len(roiLabels), nmsThreshold))
        imgDebug = visualizeResults(imgPath, roiLabels, None, roiCoords, p.cntk_padWidth, p.cntk_padHeight, p.classes, nmsKeepIndices, boDrawNegativeRois=True)
        if not testing:
            imshow(imgDebug, waitDuration=0, maxDim=800)
    print('DONE.')
    return True
if __name__ == '__main__':
    generate_rois_visualization()