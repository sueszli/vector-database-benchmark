"""Small library that points to the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from inception.dataset import Dataset

class FlowersData(Dataset):
    """Flowers data set."""

    def __init__(self, subset):
        if False:
            for i in range(10):
                print('nop')
        super(FlowersData, self).__init__('Flowers', subset)

    def num_classes(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of classes in the data set.'
        return 5

    def num_examples_per_epoch(self):
        if False:
            while True:
                i = 10
        'Returns the number of examples in the data subset.'
        if self.subset == 'train':
            return 3170
        if self.subset == 'validation':
            return 500

    def download_message(self):
        if False:
            while True:
                i = 10
        'Instruction to download and extract the tarball from Flowers website.'
        print('Failed to find any Flowers %s files' % self.subset)
        print('')
        print('If you have already downloaded and processed the data, then make sure to set --data_dir to point to the directory containing the location of the sharded TFRecords.\n')
        print('Please see README.md for instructions on how to build the flowers dataset using download_and_preprocess_flowers.\n')