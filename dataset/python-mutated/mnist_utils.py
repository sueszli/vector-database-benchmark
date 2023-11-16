from __future__ import print_function
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import sys
import gzip
import shutil
import os
import struct
import numpy as np

def loadData(src, cimg):
    if False:
        for i in range(10):
            print('nop')
    print('Downloading ' + src)
    (gzfname, h) = urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            if n[0] != 50855936:
                raise Exception('Invalid file: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))

def loadLabels(src, cimg):
    if False:
        for i in range(10):
            print('nop')
    print('Downloading ' + src)
    (gzfname, h) = urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            if n[0] != 17301504:
                raise Exception('Invalid file: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            res = np.fromstring(gz.read(cimg), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))

def load(dataSrc, labelsSrc, cimg):
    if False:
        for i in range(10):
            print('nop')
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

def savetxt(filename, ndarray):
    if False:
        while True:
            i = 10
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))