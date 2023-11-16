import struct
import numpy as np

def loadlocal_mnist(images_path, labels_path):
    if False:
        while True:
            i = 10
    'Read MNIST from ubyte files.\n\n    Parameters\n    ----------\n    images_path : str\n        path to the test or train MNIST ubyte file\n    labels_path : str\n        path to the test or train MNIST class labels file\n\n    Returns\n    --------\n    images : [n_samples, n_pixels] numpy.array\n        Pixel values of the images.\n    labels : [n_samples] numpy array\n        Target class labels\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/\n\n    '
    with open(labels_path, 'rb') as lbpath:
        (magic, n) = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        (magic, num, rows, cols) = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return (images, labels)