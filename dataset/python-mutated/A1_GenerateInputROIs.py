from __future__ import print_function
from builtins import input
import os, sys, datetime
import numpy as np
import shutil, time
import PARAMETERS
from cntk_helpers import makeDirectory, getFilesInDirectory, imread, imWidth, imHeight, imWidthHeight, getSelectiveSearchRois, imArrayWidthHeight, getGridRois, filterRois, imArrayWidth, imArrayHeight, getCntkInputPaths, getCntkRoiCoordsLine, getCntkRoiLabelsLine, roiTransformPadScaleParams, roiTransformPadScale, cntkPadInputs
boSaveDebugImg = True
subDirs = ['positive', 'testImages', 'negative']
image_sets = ['train', 'test']
boAddSelectiveSearchROIs = True
boAddRoisOnGrid = True

def generate_input_rois(testing=False):
    if False:
        print('Hello World!')
    p = PARAMETERS.get_parameters_for_dataset()
    if not p.datasetName.startswith('pascalVoc'):
        makeDirectory(p.roiDir)
        roi_minDim = p.roi_minDimRel * p.roi_maxImgDim
        roi_maxDim = p.roi_maxDimRel * p.roi_maxImgDim
        roi_minNrPixels = p.roi_minNrPixelsRel * p.roi_maxImgDim * p.roi_maxImgDim
        roi_maxNrPixels = p.roi_maxNrPixelsRel * p.roi_maxImgDim * p.roi_maxImgDim
        for subdir in subDirs:
            makeDirectory(os.path.join(p.roiDir, subdir))
            imgFilenames = getFilesInDirectory(os.path.join(p.imgDir, subdir), '.jpg')
            for (imgIndex, imgFilename) in enumerate(imgFilenames):
                roiPath = '{}/{}/{}.roi.txt'.format(p.roiDir, subdir, imgFilename[:-4])
                print(imgIndex, len(imgFilenames), subdir, imgFilename)
                tstart = datetime.datetime.now()
                imgPath = os.path.join(p.imgDir, subdir, imgFilename)
                imgOrig = imread(imgPath)
                if imWidth(imgPath) > imHeight(imgPath):
                    print(imWidth(imgPath), imHeight(imgPath))
                if boAddSelectiveSearchROIs:
                    print('Calling selective search..')
                    (rects, img, scale) = getSelectiveSearchRois(imgOrig, p.ss_scale, p.ss_sigma, p.ss_minSize, p.roi_maxImgDim)
                    print('   Number of rois detected using selective search: ' + str(len(rects)))
                else:
                    rects = []
                    (img, scale) = imresizeMaxDim(imgOrig, p.roi_maxImgDim, boUpscale=True, interpolation=cv2.INTER_AREA)
                (imgWidth, imgHeight) = imArrayWidthHeight(img)
                if boAddRoisOnGrid:
                    rectsGrid = getGridRois(imgWidth, imgHeight, p.grid_nrScales, p.grid_aspectRatios)
                    print('   Number of rois on grid added: ' + str(len(rectsGrid)))
                    rects += rectsGrid
                print('   Number of rectangles before filtering  = ' + str(len(rects)))
                rois = filterRois(rects, imgWidth, imgHeight, roi_minNrPixels, roi_maxNrPixels, roi_minDim, roi_maxDim, p.roi_maxAspectRatio)
                if len(rois) == 0:
                    rois = [[5, 5, imgWidth - 5, imgHeight - 5]]
                print('   Number of rectangles after filtering  = ' + str(len(rois)))
                rois = np.int32(np.array(rois) / scale)
                assert np.min(rois) >= 0
                assert np.max(rois[:, [0, 2]]) < imArrayWidth(imgOrig)
                assert np.max(rois[:, [1, 3]]) < imArrayHeight(imgOrig)
                np.savetxt(roiPath, rois, fmt='%d')
                print('   Time [ms]: ' + str((datetime.datetime.now() - tstart).total_seconds() * 1000))
    if os.path.exists(p.cntkFilesDir):
        assert p.cntkFilesDir.endswith('cntkFiles')
        if not testing:
            userInput = input('--> INPUT: Press "y" to delete directory ' + p.cntkFilesDir + ': ')
            if userInput.lower() not in ['y', 'yes']:
                print('User input is %s: exiting now.' % userInput)
                exit(-1)
        shutil.rmtree(p.cntkFilesDir)
        time.sleep(0.1)
    for image_set in image_sets:
        imdb = p.imdbs[image_set]
        print('Number of images in set {} = {}'.format(image_set, imdb.num_images))
        makeDirectory(p.cntkFilesDir)
        (cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, nrRoisPath) = getCntkInputPaths(p.cntkFilesDir, image_set)
        with open(nrRoisPath, 'w') as nrRoisFile, open(cntkImgsPath, 'w') as cntkImgsFile, open(cntkRoiCoordsPath, 'w') as cntkRoiCoordsFile, open(cntkRoiLabelsPath, 'w') as cntkRoiLabelsFile:
            for imgIndex in range(0, imdb.num_images):
                if imgIndex % 50 == 0:
                    print("Processing image set '{}', image {} of {}".format(image_set, imgIndex, imdb.num_images))
                currBoxes = imdb.roidb[imgIndex]['boxes']
                currGtOverlaps = imdb.roidb[imgIndex]['gt_overlaps']
                imgPath = imdb.image_path_at(imgIndex)
                (imgWidth, imgHeight) = imWidthHeight(imgPath)
                (targetw, targeth, w_offset, h_offset, scale) = roiTransformPadScaleParams(imgWidth, imgHeight, p.cntk_padWidth, p.cntk_padHeight)
                boxesStr = ''
                labelsStr = ''
                nrBoxes = len(currBoxes)
                for (boxIndex, box) in enumerate(currBoxes):
                    rect = roiTransformPadScale(box, w_offset, h_offset, scale)
                    boxesStr += getCntkRoiCoordsLine(rect, p.cntk_padWidth, p.cntk_padHeight)
                    labelsStr += getCntkRoiLabelsLine(currGtOverlaps[boxIndex, :].toarray()[0], p.train_posOverlapThres, p.nrClasses)
                (boxesStr, labelsStr) = cntkPadInputs(nrBoxes, p.cntk_nrRois, p.nrClasses, boxesStr, labelsStr)
                nrRoisFile.write('{}\n'.format(nrBoxes))
                cntkImgsFile.write('{}\t{}\t0\n'.format(imgIndex, imgPath))
                cntkRoiCoordsFile.write('{} |rois{}\n'.format(imgIndex, boxesStr))
                cntkRoiLabelsFile.write('{} |roiLabels{}\n'.format(imgIndex, labelsStr))
    print('DONE.')
    return True
if __name__ == '__main__':
    generate_input_rois()