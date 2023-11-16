"""Tests for feature_io, the python interface of DelfFeatures."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from delf import feature_io

def create_data():
    if False:
        for i in range(10):
            print('nop')
    'Creates data to be used in tests.\n\n  Returns:\n    locations: [N, 2] float array which denotes the selected keypoint\n      locations. N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations.\n  '
    locations = np.arange(8, dtype=np.float32).reshape(4, 2)
    scales = np.arange(4, dtype=np.float32)
    attention = np.arange(4, dtype=np.float32)
    orientations = np.arange(4, dtype=np.float32)
    descriptors = np.zeros([4, 1024])
    descriptors[0,] = np.arange(1024)
    descriptors[1,] = np.zeros([1024])
    descriptors[2,] = np.ones([1024])
    descriptors[3,] = -np.ones([1024])
    return (locations, scales, descriptors, attention, orientations)

class DelfFeaturesIoTest(tf.test.TestCase):

    def testConversionAndBack(self):
        if False:
            i = 10
            return i + 15
        (locations, scales, descriptors, attention, orientations) = create_data()
        serialized = feature_io.SerializeToString(locations, scales, descriptors, attention, orientations)
        parsed_data = feature_io.ParseFromString(serialized)
        self.assertAllEqual(locations, parsed_data[0])
        self.assertAllEqual(scales, parsed_data[1])
        self.assertAllEqual(descriptors, parsed_data[2])
        self.assertAllEqual(attention, parsed_data[3])
        self.assertAllEqual(orientations, parsed_data[4])

    def testConversionAndBackNoOrientations(self):
        if False:
            print('Hello World!')
        (locations, scales, descriptors, attention, _) = create_data()
        serialized = feature_io.SerializeToString(locations, scales, descriptors, attention)
        parsed_data = feature_io.ParseFromString(serialized)
        self.assertAllEqual(locations, parsed_data[0])
        self.assertAllEqual(scales, parsed_data[1])
        self.assertAllEqual(descriptors, parsed_data[2])
        self.assertAllEqual(attention, parsed_data[3])
        self.assertAllEqual(np.zeros([4]), parsed_data[4])

    def testWriteAndReadToFile(self):
        if False:
            i = 10
            return i + 15
        (locations, scales, descriptors, attention, orientations) = create_data()
        tmpdir = tf.test.get_temp_dir()
        filename = os.path.join(tmpdir, 'test.delf')
        feature_io.WriteToFile(filename, locations, scales, descriptors, attention, orientations)
        data_read = feature_io.ReadFromFile(filename)
        self.assertAllEqual(locations, data_read[0])
        self.assertAllEqual(scales, data_read[1])
        self.assertAllEqual(descriptors, data_read[2])
        self.assertAllEqual(attention, data_read[3])
        self.assertAllEqual(orientations, data_read[4])

    def testWriteAndReadToFileEmptyFile(self):
        if False:
            while True:
                i = 10
        tmpdir = tf.test.get_temp_dir()
        filename = os.path.join(tmpdir, 'test.delf')
        feature_io.WriteToFile(filename, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        data_read = feature_io.ReadFromFile(filename)
        self.assertAllEqual(np.array([]), data_read[0])
        self.assertAllEqual(np.array([]), data_read[1])
        self.assertAllEqual(np.array([]), data_read[2])
        self.assertAllEqual(np.array([]), data_read[3])
        self.assertAllEqual(np.array([]), data_read[4])
if __name__ == '__main__':
    tf.test.main()