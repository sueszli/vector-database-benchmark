"""Small library that points to the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from inception.dataset import Dataset

class ImagenetData(Dataset):
    """ImageNet data set."""

    def __init__(self, subset):
        if False:
            for i in range(10):
                print('nop')
        super(ImagenetData, self).__init__('ImageNet', subset)

    def num_classes(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of classes in the data set.'
        return 1000

    def num_examples_per_epoch(self):
        if False:
            return 10
        'Returns the number of examples in the data set.'
        if self.subset == 'train':
            return 1281167
        if self.subset == 'validation':
            return 50000

    def download_message(self):
        if False:
            while True:
                i = 10
        'Instruction to download and extract the tarball from Flowers website.'
        print('Failed to find any ImageNet %s files' % self.subset)
        print('')
        print('If you have already downloaded and processed the data, then make sure to set --data_dir to point to the directory containing the location of the sharded TFRecords.\n')
        print('If you have not downloaded and prepared the ImageNet data in the TFRecord format, you will need to do this at least once. This process could take several hours depending on the speed of your computer and network connection\n')
        print('Please see README.md for instructions on how to build the ImageNet dataset using download_and_preprocess_imagenet.\n')
        print('Note that the raw data size is 300 GB and the processed data size is 150 GB. Please ensure you have at least 500GB disk space.')