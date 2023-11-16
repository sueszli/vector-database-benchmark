"""Base class for deep learning models
"""
from __future__ import division
from __future__ import print_function
import tensorflow

def _get_tensorflow_version():
    if False:
        i = 10
        return i + 15
    ' Utility function to decide the version of tensorflow, which will \n    affect how to import keras models. \n\n    Returns\n    -------\n    tensorflow version : int\n\n    '
    tf_version = str(tensorflow.__version__)
    if int(tf_version.split('.')[0]) != 1 and int(tf_version.split('.')[0]) != 2:
        raise ValueError('tensorflow version error')
    return int(tf_version.split('.')[0]) * 100 + int(tf_version.split('.')[1])