from __future__ import print_function
import os
from imdb_data import imdb_data
import fastRCNN, time, datetime
from fastRCNN.pascal_voc import pascal_voc
print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
dataset = 'Grocery'

class Parameters:

    def __init__(self, datasetName):
        if False:
            while True:
                i = 10
        self.datasetName = datasetName
        self.cntk_nrRois = 100
        self.cntk_padWidth = 1000
        self.cntk_padHeight = 1000
        self.rootDir = os.path.dirname(os.path.abspath(__file__))
        self.imgDir = os.path.join(self.rootDir, '..', '..', '..', 'DataSets', datasetName)
        self.procDir = os.path.join(self.rootDir, 'proc', datasetName + '_{}'.format(self.cntk_nrRois))
        self.resultsDir = os.path.join(self.rootDir, 'results', datasetName + '_{}'.format(self.cntk_nrRois))
        self.roiDir = os.path.join(self.procDir, 'rois')
        self.cntkFilesDir = os.path.join(self.procDir, 'cntkFiles')
        self.cntkTemplateDir = self.rootDir
        self.roi_minDimRel = 0.01
        self.roi_maxDimRel = 1.0
        self.roi_minNrPixelsRel = 0
        self.roi_maxNrPixelsRel = 1.0
        self.roi_maxAspectRatio = 4.0
        self.roi_maxImgDim = 200
        self.ss_scale = 100
        self.ss_sigma = 1.2
        self.ss_minSize = 20
        self.grid_nrScales = 7
        self.grid_aspectRatios = [1.0, 2.0, 0.5]
        self.train_posOverlapThres = 0.5
        self.nmsThreshold = 0.3
        self.cntk_num_train_images = -1
        self.cntk_num_test_images = -1
        self.cntk_mb_size = -1
        self.cntk_max_epochs = -1
        self.cntk_momentum_per_sample = -1
        self.distributed_flg = False
        self.num_quantization_bits = 32
        self.warm_up = 0

class GroceryParameters(Parameters):

    def __init__(self, datasetName):
        if False:
            return 10
        super(GroceryParameters, self).__init__(datasetName)
        self.classes = ('__background__', 'avocado', 'orange', 'butter', 'champagne', 'eggBox', 'gerkin', 'joghurt', 'ketchup', 'orangeJuice', 'onion', 'pepper', 'tomato', 'water', 'milk', 'tabasco', 'mustard')
        self.roi_minDimRel = 0.04
        self.roi_maxDimRel = 0.4
        self.roi_minNrPixelsRel = 2 * self.roi_minDimRel * self.roi_minDimRel
        self.roi_maxNrPixelsRel = 0.33 * self.roi_maxDimRel * self.roi_maxDimRel
        self.classifier = 'nn'
        self.cntk_num_train_images = 25
        self.cntk_num_test_images = 5
        self.cntk_mb_size = 5
        self.cntk_max_epochs = 20
        self.cntk_momentum_per_sample = 0.8187307530779818
        self.nmsThreshold = 0.01
        self.imdbs = dict()
        for image_set in ['train', 'test']:
            self.imdbs[image_set] = imdb_data(image_set, self.classes, self.cntk_nrRois, self.imgDir, self.roiDir, self.cntkFilesDir, boAddGroundTruthRois=image_set != 'test')

class CustomDataset(Parameters):

    def __init__(self, datasetName):
        if False:
            print('Hello World!')
        super(CustomDataset, self).__init__(datasetName)

class PascalParameters(Parameters):

    def __init__(self, datasetName):
        if False:
            i = 10
            return i + 15
        super(PascalParameters, self).__init__(datasetName)
        if datasetName.startswith('pascalVoc_aeroplanesOnly'):
            self.classes = ('__background__', 'aeroplane')
            self.lutImageSet = {'train': 'trainval.aeroplaneOnly', 'test': 'test.aeroplaneOnly'}
        else:
            self.classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
            self.lutImageSet = {'train': 'trainval', 'test': 'test'}
        self.classifier = 'nn'
        self.cntk_num_train_images = 5011
        self.cntk_num_test_images = 4952
        self.cntk_mb_size = 2
        self.cntk_max_epochs = 17
        self.cntk_momentum_per_sample = 0.951229424500714
        self.pascalDataDir = os.path.join(self.rootDir, '..', '..', 'DataSets', 'Pascal')
        self.imgDir = self.pascalDataDir
        self.imdbs = dict()
        for (image_set, year) in zip(['train', 'test'], ['2007', '2007']):
            self.imdbs[image_set] = fastRCNN.pascal_voc(self.lutImageSet[image_set], year, self.classes, self.cntk_nrRois, cacheDir=self.cntkFilesDir, devkit_path=self.pascalDataDir)
            print('Number of {} images: {}'.format(image_set, self.imdbs[image_set].num_images))

def get_parameters_for_dataset(datasetName=dataset):
    if False:
        i = 10
        return i + 15
    if datasetName == 'Grocery':
        parameters = GroceryParameters(datasetName)
    elif datasetName.startswith('pascalVoc'):
        parameters = PascalParameters(datasetName)
    elif datasetName == 'CustomDataset':
        parameters = CustomDataset(datasetName)
    else:
        ERROR
    nrClasses = len(parameters.classes)
    parameters.cntk_featureDimensions = {'nn': nrClasses}
    parameters.nrClasses = nrClasses
    assert parameters.cntk_padWidth == parameters.cntk_padHeight, 'ERROR: different width and height for padding currently not supported.'
    assert parameters.classifier.lower() in ['svm', 'nn'], "ERROR: only 'nn' or 'svm' classifier supported."
    assert not (parameters.datasetName == 'pascalVoc' and parameters.classifier == 'svm'), 'ERROR: while technically possibly, writing 2nd-last layer of CNTK model for all pascalVOC images takes too much disk memory.'
    print('PARAMETERS: datasetName = ' + datasetName)
    print('PARAMETERS: cntk_nrRois = {}'.format(parameters.cntk_nrRois))
    return parameters