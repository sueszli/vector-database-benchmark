"""Downloads pretrained InceptionV3 and ResnetV2-50 checkpoints."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tarfile
import urllib
INCEPTION_URL = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
RESNET_URL = 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz'

def DownloadWeights(model_dir, url):
    if False:
        i = 10
        return i + 15
    os.makedirs(model_dir)
    tar_path = os.path.join(model_dir, 'ckpt.tar.gz')
    urllib.urlretrieve(url, tar_path)
    tar = tarfile.open(os.path.join(model_dir, 'ckpt.tar.gz'))
    tar.extractall(model_dir)
if __name__ == '__main__':
    ckpt_dir = 'pretrained_checkpoints'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print('Downloading inception pretrained weights...')
    inception_dir = os.path.join(ckpt_dir, 'inception')
    DownloadWeights(inception_dir, INCEPTION_URL)
    print('Done downloading inception pretrained weights.')
    print('Downloading resnet pretrained weights...')
    resnet_dir = os.path.join(ckpt_dir, 'resnet')
    DownloadWeights(resnet_dir, RESNET_URL)
    print('Done downloading resnet pretrained weights.')