from __future__ import print_function
import os
import sys
import glob
import tarfile
import numpy as np
from scipy.io import loadmat
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

def write_to_file(filename, base_path, img_paths, img_labels):
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(base_path, filename), 'w+') as f:
        for i in range(0, len(img_paths)):
            f.write('%s\t%s\n' % (os.path.join(base_path, img_paths[i]), img_labels[i]))

def download_flowers_data():
    if False:
        for i in range(10):
            print('nop')
    dataset_folder = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(dataset_folder, 'jpg')):
        print('Downloading data from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ...')
        tar_filename = os.path.join(dataset_folder, '102flowers.tgz')
        label_filename = os.path.join(dataset_folder, 'imagelabels.mat')
        set_filename = os.path.join(dataset_folder, 'setid.mat')
        if not os.path.exists(tar_filename):
            urlretrieve('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', tar_filename)
        if not os.path.exists(label_filename):
            urlretrieve('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', label_filename)
        if not os.path.exists(set_filename):
            urlretrieve('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat', set_filename)
        print('Extracting ' + tar_filename + '...')
        tarfile.open(tar_filename).extractall(path=dataset_folder)
        print('Writing map files ...')
        setid = loadmat(set_filename)
        idx_train = setid['trnid'][0] - 1
        idx_test = setid['tstid'][0] - 1
        idx_val = setid['valid'][0] - 1
        image_paths = np.array(sorted(glob.glob(dataset_folder + '/jpg/*.jpg')))
        image_labels = loadmat(label_filename)['labels'][0]
        image_labels -= 1
        write_to_file('1k_img_map.txt', dataset_folder, image_paths[idx_train], image_labels[idx_train])
        write_to_file('6k_img_map.txt', dataset_folder, image_paths[idx_test], image_labels[idx_test])
        write_to_file('val_map.txt', dataset_folder, image_paths[idx_val], image_labels[idx_val])
        os.remove(tar_filename)
        os.remove(label_filename)
        os.remove(set_filename)
        print('Done.')
    else:
        print('Data already available at ' + dataset_folder + '/Flowers')
if __name__ == '__main__':
    download_flowers_data()