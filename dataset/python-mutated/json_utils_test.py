"""Tests for google3.image.understanding.object_detection.utils.json_utils."""
import os
import tensorflow as tf
from object_detection.utils import json_utils

class JsonUtilsTest(tf.test.TestCase):

    def testDumpReasonablePrecision(self):
        if False:
            print('Hello World!')
        output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
        with tf.gfile.GFile(output_path, 'w') as f:
            json_utils.Dump(1.0, f, float_digits=2)
        with tf.gfile.GFile(output_path, 'r') as f:
            self.assertEqual(f.read(), '1.00')

    def testDumpPassExtraParams(self):
        if False:
            i = 10
            return i + 15
        output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
        with tf.gfile.GFile(output_path, 'w') as f:
            json_utils.Dump([1.0], f, float_digits=2, indent=3)
        with tf.gfile.GFile(output_path, 'r') as f:
            self.assertEqual(f.read(), '[\n   1.00\n]')

    def testDumpZeroPrecision(self):
        if False:
            while True:
                i = 10
        output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
        with tf.gfile.GFile(output_path, 'w') as f:
            json_utils.Dump(1.0, f, float_digits=0, indent=3)
        with tf.gfile.GFile(output_path, 'r') as f:
            self.assertEqual(f.read(), '1')

    def testDumpUnspecifiedPrecision(self):
        if False:
            for i in range(10):
                print('nop')
        output_path = os.path.join(tf.test.get_temp_dir(), 'test.json')
        with tf.gfile.GFile(output_path, 'w') as f:
            json_utils.Dump(1.012345, f)
        with tf.gfile.GFile(output_path, 'r') as f:
            self.assertEqual(f.read(), '1.012345')

    def testDumpsReasonablePrecision(self):
        if False:
            return 10
        s = json_utils.Dumps(1.0, float_digits=2)
        self.assertEqual(s, '1.00')

    def testDumpsPassExtraParams(self):
        if False:
            return 10
        s = json_utils.Dumps([1.0], float_digits=2, indent=3)
        self.assertEqual(s, '[\n   1.00\n]')

    def testDumpsZeroPrecision(self):
        if False:
            print('Hello World!')
        s = json_utils.Dumps(1.0, float_digits=0)
        self.assertEqual(s, '1')

    def testDumpsUnspecifiedPrecision(self):
        if False:
            return 10
        s = json_utils.Dumps(1.012345)
        self.assertEqual(s, '1.012345')

    def testPrettyParams(self):
        if False:
            print('Hello World!')
        s = json_utils.Dumps({'v': 1.012345, 'n': 2}, **json_utils.PrettyParams())
        self.assertEqual(s, '{\n  "n": 2,\n  "v": 1.0123\n}')

    def testPrettyParamsExtraParamsInside(self):
        if False:
            i = 10
            return i + 15
        s = json_utils.Dumps({'v': 1.012345, 'n': float('nan')}, **json_utils.PrettyParams(allow_nan=True))
        self.assertEqual(s, '{\n  "n": NaN,\n  "v": 1.0123\n}')
        with self.assertRaises(ValueError):
            s = json_utils.Dumps({'v': 1.012345, 'n': float('nan')}, **json_utils.PrettyParams(allow_nan=False))

    def testPrettyParamsExtraParamsOutside(self):
        if False:
            return 10
        s = json_utils.Dumps({'v': 1.012345, 'n': float('nan')}, allow_nan=True, **json_utils.PrettyParams())
        self.assertEqual(s, '{\n  "n": NaN,\n  "v": 1.0123\n}')
        with self.assertRaises(ValueError):
            s = json_utils.Dumps({'v': 1.012345, 'n': float('nan')}, allow_nan=False, **json_utils.PrettyParams())
if __name__ == '__main__':
    tf.test.main()