"""Tests VLOG printing in TensorFlow."""
import os
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

class VlogTest(test.TestCase):

    def test_simple_conv(self):
        if False:
            for i in range(10):
                print('nop')
        (height, width) = (7, 9)
        images = random_ops.random_uniform((5, height, width, 3))
        w = random_ops.random_normal([5, 5, 3, 32], mean=0, stddev=1)
        nn_ops.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME')
if __name__ == '__main__':
    test.main()