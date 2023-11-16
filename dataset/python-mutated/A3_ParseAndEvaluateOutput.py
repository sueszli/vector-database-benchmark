"""Test a Fast R-CNN network on an image database."""
from __future__ import print_function
import os
from fastRCNN.test import test_net as evaluate_net
from fastRCNN.timer import Timer
from imdb_data import imdb_data
from cntk_helpers import makeDirectory, parseCntkOutput, DummyNet, deleteAllFilesInDirectory
import PARAMETERS
image_set = 'test'

def evaluate_output():
    if False:
        print('Hello World!')
    p = PARAMETERS.get_parameters_for_dataset()
    print('Parsing CNTK output for image set: ' + image_set)
    cntkImgsListPath = os.path.join(p.cntkFilesDir, image_set + '.txt')
    outParsedDir = os.path.join(p.cntkFilesDir, image_set + '_parsed')
    cntkOutputPath = os.path.join(p.cntkFilesDir, image_set + '.z')
    makeDirectory(outParsedDir)
    parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, p.cntk_nrRois, p.cntk_featureDimensions[p.classifier], saveCompressed=True, skipCheck=True)
    imdb = p.imdbs[image_set]
    net = DummyNet(4096, imdb.num_classes, outParsedDir)
    if type(imdb) == imdb_data:
        evalTempDir = None
    else:
        evalTempDir = os.path.join(p.procDir, 'eval_mAP_' + image_set)
        makeDirectory(evalTempDir)
        deleteAllFilesInDirectory(evalTempDir, None)
    evaluate_net(net, imdb, evalTempDir, None, p.classifier, p.nmsThreshold, boUsePythonImpl=True)
    print('DONE.')
    return True
if __name__ == '__main__':
    evaluate_output()