"""Python interface for DatumProto.

DatumProto is protocol buffer used to serialize tensor with arbitrary shape.
Please refer to datum.proto for details.

Support read and write of DatumProto from/to NumPy array and file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from delf import datum_pb2

def ArrayToDatum(arr):
    if False:
        return 10
    'Converts NumPy array to DatumProto.\n\n  Supports arrays of types:\n    - float16 (it is converted into a float32 in DatumProto)\n    - float32\n    - float64 (it is converted into a float32 in DatumProto)\n    - uint8 (it is converted into a uint32 in DatumProto)\n    - uint16 (it is converted into a uint32 in DatumProto)\n    - uint32\n    - uint64 (it is converted into a uint32 in DatumProto)\n\n  Args:\n    arr: NumPy array of arbitrary shape.\n\n  Returns:\n    datum: DatumProto object.\n\n  Raises:\n    ValueError: If array type is unsupported.\n  '
    datum = datum_pb2.DatumProto()
    if arr.dtype in ('float16', 'float32', 'float64'):
        datum.float_list.value.extend(arr.astype('float32').flat)
    elif arr.dtype in ('uint8', 'uint16', 'uint32', 'uint64'):
        datum.uint32_list.value.extend(arr.astype('uint32').flat)
    else:
        raise ValueError('Unsupported array type: %s' % arr.dtype)
    datum.shape.dim.extend(arr.shape)
    return datum

def ArraysToDatumPair(arr_1, arr_2):
    if False:
        for i in range(10):
            print('nop')
    'Converts numpy arrays to DatumPairProto.\n\n  Supports same formats as `ArrayToDatum`, see documentation therein.\n\n  Args:\n    arr_1: NumPy array of arbitrary shape.\n    arr_2: NumPy array of arbitrary shape.\n\n  Returns:\n    datum_pair: DatumPairProto object.\n  '
    datum_pair = datum_pb2.DatumPairProto()
    datum_pair.first.CopyFrom(ArrayToDatum(arr_1))
    datum_pair.second.CopyFrom(ArrayToDatum(arr_2))
    return datum_pair

def DatumToArray(datum):
    if False:
        return 10
    'Converts data saved in DatumProto to NumPy array.\n\n  Args:\n    datum: DatumProto object.\n\n  Returns:\n    NumPy array of arbitrary shape.\n  '
    if datum.HasField('float_list'):
        return np.array(datum.float_list.value).astype('float32').reshape(datum.shape.dim)
    elif datum.HasField('uint32_list'):
        return np.array(datum.uint32_list.value).astype('uint32').reshape(datum.shape.dim)
    else:
        raise ValueError('Input DatumProto does not have float_list or uint32_list')

def DatumPairToArrays(datum_pair):
    if False:
        while True:
            i = 10
    'Converts data saved in DatumPairProto to NumPy arrays.\n\n  Args:\n    datum_pair: DatumPairProto object.\n\n  Returns:\n    Two NumPy arrays of arbitrary shape.\n  '
    first_datum = DatumToArray(datum_pair.first)
    second_datum = DatumToArray(datum_pair.second)
    return (first_datum, second_datum)

def SerializeToString(arr):
    if False:
        while True:
            i = 10
    'Converts NumPy array to serialized DatumProto.\n\n  Args:\n    arr: NumPy array of arbitrary shape.\n\n  Returns:\n    Serialized DatumProto string.\n  '
    datum = ArrayToDatum(arr)
    return datum.SerializeToString()

def SerializePairToString(arr_1, arr_2):
    if False:
        while True:
            i = 10
    'Converts pair of NumPy arrays to serialized DatumPairProto.\n\n  Args:\n    arr_1: NumPy array of arbitrary shape.\n    arr_2: NumPy array of arbitrary shape.\n\n  Returns:\n    Serialized DatumPairProto string.\n  '
    datum_pair = ArraysToDatumPair(arr_1, arr_2)
    return datum_pair.SerializeToString()

def ParseFromString(string):
    if False:
        while True:
            i = 10
    'Converts serialized DatumProto string to NumPy array.\n\n  Args:\n    string: Serialized DatumProto string.\n\n  Returns:\n    NumPy array.\n  '
    datum = datum_pb2.DatumProto()
    datum.ParseFromString(string)
    return DatumToArray(datum)

def ParsePairFromString(string):
    if False:
        while True:
            i = 10
    'Converts serialized DatumPairProto string to NumPy arrays.\n\n  Args:\n    string: Serialized DatumProto string.\n\n  Returns:\n    Two NumPy arrays.\n  '
    datum_pair = datum_pb2.DatumPairProto()
    datum_pair.ParseFromString(string)
    return DatumPairToArrays(datum_pair)

def ReadFromFile(file_path):
    if False:
        while True:
            i = 10
    'Helper function to load data from a DatumProto format in a file.\n\n  Args:\n    file_path: Path to file containing data.\n\n  Returns:\n    data: NumPy array.\n  '
    with tf.gfile.GFile(file_path, 'rb') as f:
        return ParseFromString(f.read())

def ReadPairFromFile(file_path):
    if False:
        print('Hello World!')
    'Helper function to load data from a DatumPairProto format in a file.\n\n  Args:\n    file_path: Path to file containing data.\n\n  Returns:\n    Two NumPy arrays.\n  '
    with tf.gfile.GFile(file_path, 'rb') as f:
        return ParsePairFromString(f.read())

def WriteToFile(data, file_path):
    if False:
        i = 10
        return i + 15
    'Helper function to write data to a file in DatumProto format.\n\n  Args:\n    data: NumPy array.\n    file_path: Path to file that will be written.\n  '
    serialized_data = SerializeToString(data)
    with tf.gfile.GFile(file_path, 'w') as f:
        f.write(serialized_data)

def WritePairToFile(arr_1, arr_2, file_path):
    if False:
        print('Hello World!')
    'Helper function to write pair of arrays to a file in DatumPairProto format.\n\n  Args:\n    arr_1: NumPy array of arbitrary shape.\n    arr_2: NumPy array of arbitrary shape.\n    file_path: Path to file that will be written.\n  '
    serialized_data = SerializePairToString(arr_1, arr_2)
    with tf.gfile.GFile(file_path, 'w') as f:
        f.write(serialized_data)