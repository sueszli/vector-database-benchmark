"""Python interface for DelfFeatures proto.

Support read and write of DelfFeatures from/to numpy arrays and file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from delf import feature_pb2
from delf import datum_io

def ArraysToDelfFeatures(locations, scales, descriptors, attention, orientations=None):
    if False:
        i = 10
        return i + 15
    'Converts DELF features to DelfFeatures proto.\n\n  Args:\n    locations: [N, 2] float array which denotes the selected keypoint locations.\n      N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations. If None, all orientations\n      are set to zero.\n\n  Returns:\n    delf_features: DelfFeatures object.\n  '
    num_features = len(attention)
    assert num_features == locations.shape[0]
    assert num_features == len(scales)
    assert num_features == descriptors.shape[0]
    if orientations is None:
        orientations = np.zeros([num_features], dtype=np.float32)
    else:
        assert num_features == len(orientations)
    delf_features = feature_pb2.DelfFeatures()
    for i in range(num_features):
        delf_feature = delf_features.feature.add()
        delf_feature.y = locations[i, 0]
        delf_feature.x = locations[i, 1]
        delf_feature.scale = scales[i]
        delf_feature.orientation = orientations[i]
        delf_feature.strength = attention[i]
        delf_feature.descriptor.CopyFrom(datum_io.ArrayToDatum(descriptors[i,]))
    return delf_features

def DelfFeaturesToArrays(delf_features):
    if False:
        for i in range(10):
            print('nop')
    'Converts data saved in DelfFeatures to numpy arrays.\n\n  If there are no features, the function returns four empty arrays.\n\n  Args:\n    delf_features: DelfFeatures object.\n\n  Returns:\n    locations: [N, 2] float array which denotes the selected keypoint\n      locations. N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations.\n  '
    num_features = len(delf_features.feature)
    if num_features == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    descriptor_dim = len(datum_io.DatumToArray(delf_features.feature[0].descriptor))
    locations = np.zeros([num_features, 2])
    scales = np.zeros([num_features])
    descriptors = np.zeros([num_features, descriptor_dim])
    attention = np.zeros([num_features])
    orientations = np.zeros([num_features])
    for i in range(num_features):
        delf_feature = delf_features.feature[i]
        locations[i, 0] = delf_feature.y
        locations[i, 1] = delf_feature.x
        scales[i] = delf_feature.scale
        descriptors[i,] = datum_io.DatumToArray(delf_feature.descriptor)
        attention[i] = delf_feature.strength
        orientations[i] = delf_feature.orientation
    return (locations, scales, descriptors, attention, orientations)

def SerializeToString(locations, scales, descriptors, attention, orientations=None):
    if False:
        return 10
    'Converts numpy arrays to serialized DelfFeatures.\n\n  Args:\n    locations: [N, 2] float array which denotes the selected keypoint locations.\n      N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations. If None, all orientations\n      are set to zero.\n\n  Returns:\n    Serialized DelfFeatures string.\n  '
    delf_features = ArraysToDelfFeatures(locations, scales, descriptors, attention, orientations)
    return delf_features.SerializeToString()

def ParseFromString(string):
    if False:
        return 10
    'Converts serialized DelfFeatures string to numpy arrays.\n\n  Args:\n    string: Serialized DelfFeatures string.\n\n  Returns:\n    locations: [N, 2] float array which denotes the selected keypoint\n      locations. N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations.\n  '
    delf_features = feature_pb2.DelfFeatures()
    delf_features.ParseFromString(string)
    return DelfFeaturesToArrays(delf_features)

def ReadFromFile(file_path):
    if False:
        return 10
    'Helper function to load data from a DelfFeatures format in a file.\n\n  Args:\n    file_path: Path to file containing data.\n\n  Returns:\n    locations: [N, 2] float array which denotes the selected keypoint\n      locations. N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations.\n  '
    with tf.gfile.FastGFile(file_path, 'rb') as f:
        return ParseFromString(f.read())

def WriteToFile(file_path, locations, scales, descriptors, attention, orientations=None):
    if False:
        print('Hello World!')
    'Helper function to write data to a file in DelfFeatures format.\n\n  Args:\n    file_path: Path to file that will be written.\n    locations: [N, 2] float array which denotes the selected keypoint locations.\n      N is the number of features.\n    scales: [N] float array with feature scales.\n    descriptors: [N, depth] float array with DELF descriptors.\n    attention: [N] float array with attention scores.\n    orientations: [N] float array with orientations. If None, all orientations\n      are set to zero.\n  '
    serialized_data = SerializeToString(locations, scales, descriptors, attention, orientations)
    with tf.gfile.FastGFile(file_path, 'w') as f:
        f.write(serialized_data)