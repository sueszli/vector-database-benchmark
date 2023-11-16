from __future__ import print_function
import os, importlib, sys
from cntk_helpers import imWidthHeight, nnPredict, applyNonMaximaSuppression, makeDirectory, visualizeResults, imshow
import PARAMETERS
image_set = 'test'

def visualize_output_rois(testing=False):
    if False:
        for i in range(10):
            print('nop')
    p = PARAMETERS.get_parameters_for_dataset()
    boUseNonMaximaSurpression = True
    visualizationDir = os.path.join(p.resultsDir, 'visualizations')
    cntkParsedOutputDir = os.path.join(p.cntkFilesDir, image_set + '_parsed')
    makeDirectory(p.resultsDir)
    makeDirectory(visualizationDir)
    imdb = p.imdbs[image_set]
    for imgIndex in range(0, imdb.num_images):
        imgPath = imdb.image_path_at(imgIndex)
        (imgWidth, imgHeight) = imWidthHeight(imgPath)
        (labels, scores) = nnPredict(imgIndex, cntkParsedOutputDir, p.cntk_nrRois, len(p.classes), None)
        scores = scores[:len(imdb.roidb[imgIndex]['boxes'])]
        labels = labels[:len(imdb.roidb[imgIndex]['boxes'])]
        nmsKeepIndices = []
        if boUseNonMaximaSurpression:
            nmsKeepIndices = applyNonMaximaSuppression(p.nmsThreshold, labels, scores, imdb.roidb[imgIndex]['boxes'])
            print('Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})'.format(len(nmsKeepIndices), len(labels), p.nmsThreshold))
        imgDebug = visualizeResults(imgPath, labels, scores, imdb.roidb[imgIndex]['boxes'], p.cntk_padWidth, p.cntk_padHeight, p.classes, nmsKeepIndices, boDrawNegativeRois=True)
        if not testing:
            imshow(imgDebug, waitDuration=0, maxDim=800)
    print('DONE.')
    return True
if __name__ == '__main__':
    visualize_output_rois()