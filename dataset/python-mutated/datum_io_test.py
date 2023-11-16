"""Tests for datum_io, the python interface of DatumProto."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from delf import datum_io

class DatumIoTest(tf.test.TestCase):

    def Conversion2dTestWithType(self, dtype):
        if False:
            print('Hello World!')
        original_data = np.arange(9).reshape(3, 3).astype(dtype)
        serialized = datum_io.SerializeToString(original_data)
        retrieved_data = datum_io.ParseFromString(serialized)
        self.assertTrue(np.array_equal(original_data, retrieved_data))

    def Conversion3dTestWithType(self, dtype):
        if False:
            print('Hello World!')
        original_data = np.arange(24).reshape(2, 3, 4).astype(dtype)
        serialized = datum_io.SerializeToString(original_data)
        retrieved_data = datum_io.ParseFromString(serialized)
        self.assertTrue(np.array_equal(original_data, retrieved_data))

    def testConversion2dWithType(self):
        if False:
            return 10
        self.Conversion2dTestWithType(np.uint16)
        self.Conversion2dTestWithType(np.uint32)
        self.Conversion2dTestWithType(np.uint64)
        self.Conversion2dTestWithType(np.float16)
        self.Conversion2dTestWithType(np.float32)
        self.Conversion2dTestWithType(np.float64)

    def testConversion3dWithType(self):
        if False:
            print('Hello World!')
        self.Conversion3dTestWithType(np.uint16)
        self.Conversion3dTestWithType(np.uint32)
        self.Conversion3dTestWithType(np.uint64)
        self.Conversion3dTestWithType(np.float16)
        self.Conversion3dTestWithType(np.float32)
        self.Conversion3dTestWithType(np.float64)

    def testConversionWithUnsupportedType(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'Unsupported array type'):
            self.Conversion3dTestWithType(int)

    def testWriteAndReadToFile(self):
        if False:
            return 10
        data = np.array([[[-1.0, 125.0, -2.5], [14.5, 3.5, 0.0]], [[20.0, 0.0, 30.0], [25.5, 36.0, 42.0]]])
        tmpdir = tf.test.get_temp_dir()
        filename = os.path.join(tmpdir, 'test.datum')
        datum_io.WriteToFile(data, filename)
        data_read = datum_io.ReadFromFile(filename)
        self.assertAllEqual(data_read, data)

    def testWriteAndReadPairToFile(self):
        if False:
            for i in range(10):
                print('nop')
        data_1 = np.array([[[-1.0, 125.0, -2.5], [14.5, 3.5, 0.0]], [[20.0, 0.0, 30.0], [25.5, 36.0, 42.0]]])
        data_2 = np.array([[[255, 0, 5], [10, 300, 0]], [[20, 1, 100], [255, 360, 420]]], dtype='uint32')
        tmpdir = tf.test.get_temp_dir()
        filename = os.path.join(tmpdir, 'test.datum_pair')
        datum_io.WritePairToFile(data_1, data_2, filename)
        (data_read_1, data_read_2) = datum_io.ReadPairFromFile(filename)
        self.assertAllEqual(data_read_1, data_1)
        self.assertAllEqual(data_read_2, data_2)
if __name__ == '__main__':
    tf.test.main()