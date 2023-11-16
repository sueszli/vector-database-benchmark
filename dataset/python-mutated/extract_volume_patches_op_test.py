"""Functional tests for ExtractVolumePatches op."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ExtractVolumePatches(test.TestCase):
    """Functional tests for ExtractVolumePatches op."""

    def _VerifyValues(self, image, ksizes, strides, padding, patches):
        if False:
            return 10
        'Tests input-output pairs for the ExtractVolumePatches op.\n\n    Args:\n      image: Input tensor with shape:\n             [batch, in_planes, in_rows, in_cols, depth].\n      ksizes: Patch size specified as: [ksize_planes, ksize_rows, ksize_cols].\n      strides: Output strides, specified as:\n               [stride_planes, stride_rows, stride_cols].\n      padding: Padding type.\n      patches: Expected output.\n\n    Note:\n      rates are not supported as of now.\n    '
        ksizes = [1] + ksizes + [1]
        strides = [1] + strides + [1]
        for dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
            out_tensor = array_ops.extract_volume_patches(constant_op.constant(image.astype(dtype)), ksizes=ksizes, strides=strides, padding=padding, name='im2col_3d')
            self.assertAllClose(patches.astype(dtype), self.evaluate(out_tensor))

    def testKsize1x1x1Stride1x1x1(self):
        if False:
            print('Hello World!')
        'Verifies that for 1x1x1 kernel the output equals the input.'
        image = np.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6]) + 1
        patches = image
        for padding in ['VALID', 'SAME']:
            self._VerifyValues(image, ksizes=[1, 1, 1], strides=[1, 1, 1], padding=padding, patches=patches)

    def testKsize1x1x1Stride2x3x4(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for 1x1x1 kernel and strides.'
        image = np.arange(6 * 2 * 4 * 5 * 3).reshape([6, 2, 4, 5, 3]) + 1
        patches = image[:, ::2, ::3, ::4, :]
        for padding in ['VALID', 'SAME']:
            self._VerifyValues(image, ksizes=[1, 1, 1], strides=[2, 3, 4], padding=padding, patches=patches)

    def testKsize1x1x2Stride2x2x3(self):
        if False:
            i = 10
            return i + 15
        'Test for 1x1x2 kernel and strides.'
        image = np.arange(45).reshape([1, 3, 3, 5, 1]) + 1
        patches = np.array([[[[[1, 2], [4, 5]], [[11, 12], [14, 15]]], [[[31, 32], [34, 35]], [[41, 42], [44, 45]]]]])
        for padding in ['VALID', 'SAME']:
            self._VerifyValues(image, ksizes=[1, 1, 2], strides=[2, 2, 3], padding=padding, patches=patches)

    def testKsize2x2x2Stride1x1x1Valid(self):
        if False:
            print('Hello World!')
        'Test for 2x2x2 kernel with VALID padding.'
        image = np.arange(8).reshape([1, 2, 2, 2, 1]) + 1
        patches = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8]]]]])
        self._VerifyValues(image, ksizes=[2, 2, 2], strides=[1, 1, 1], padding='VALID', patches=patches)

    def testKsize2x2x2Stride1x1x1Same(self):
        if False:
            while True:
                i = 10
        'Test for 2x2x2 kernel with SAME padding.'
        image = np.arange(8).reshape([1, 2, 2, 2, 1]) + 1
        patches = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8], [2, 0, 4, 0, 6, 0, 8, 0]], [[3, 4, 0, 0, 7, 8, 0, 0], [4, 0, 0, 0, 8, 0, 0, 0]]], [[[5, 6, 7, 8, 0, 0, 0, 0], [6, 0, 8, 0, 0, 0, 0, 0]], [[7, 8, 0, 0, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0, 0]]]]])
        self._VerifyValues(image, ksizes=[2, 2, 2], strides=[1, 1, 1], padding='SAME', patches=patches)
if __name__ == '__main__':
    test.main()