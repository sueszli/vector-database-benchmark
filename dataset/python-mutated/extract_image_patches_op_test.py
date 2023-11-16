"""Functional tests for ExtractImagePatches op."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ExtractImagePatches(xla_test.XLATestCase):
    """Functional tests for ExtractImagePatches op."""

    def _VerifyValues(self, image, ksizes, strides, rates, padding, patches):
        if False:
            i = 10
            return i + 15
        'Tests input-output pairs for the ExtractImagePatches op.\n\n    Args:\n      image: Input tensor with shape: [batch, in_rows, in_cols, depth].\n      ksizes: Patch size specified as: [ksize_rows, ksize_cols].\n      strides: Output strides, specified as [stride_rows, stride_cols].\n      rates: Atrous rates, specified as [rate_rows, rate_cols].\n      padding: Padding type.\n      patches: Expected output.\n    '
        ksizes = [1] + ksizes + [1]
        strides = [1] + strides + [1]
        rates = [1] + rates + [1]
        with self.session():
            image_placeholder = array_ops.placeholder(dtypes.float32)
            with self.test_scope():
                out_tensor = array_ops.extract_image_patches(image_placeholder, ksizes=ksizes, strides=strides, rates=rates, padding=padding, name='im2col')
            feed_dict = {image_placeholder: image}
            self.assertAllClose(patches, out_tensor.eval(feed_dict=feed_dict))

    def testKsize1x1Stride1x1Rate1x1(self):
        if False:
            while True:
                i = 10
        'Verifies that for 1x1 kernel the output equals the input.'
        image = np.reshape(range(120), [2, 3, 4, 5])
        patches = np.reshape(range(120), [2, 3, 4, 5])
        for padding in ['VALID', 'SAME']:
            self._VerifyValues(image, ksizes=[1, 1], strides=[1, 1], rates=[1, 1], padding=padding, patches=patches)

    def testKsize1x1Stride2x3Rate1x1(self):
        if False:
            print('Hello World!')
        'Test for 1x1 kernel and strides.'
        image = np.reshape(range(120), [2, 4, 5, 3])
        patches = image[:, ::2, ::3, :]
        for padding in ['VALID', 'SAME']:
            self._VerifyValues(image, ksizes=[1, 1], strides=[2, 3], rates=[1, 1], padding=padding, patches=patches)

    def testKsize2x2Stride1x1Rate1x1Valid(self):
        if False:
            print('Hello World!')
        'Test for 2x2 kernel with VALID padding.'
        image = [[[[1], [2]], [[3], [4]]]]
        patches = [[[[1, 2, 3, 4]]]]
        self._VerifyValues(image, ksizes=[2, 2], strides=[1, 1], rates=[1, 1], padding='VALID', patches=patches)

    def testKsize2x2Stride1x1Rate1x1Same(self):
        if False:
            while True:
                i = 10
        'Test for 2x2 kernel with SAME padding.'
        image = [[[[1], [2]], [[3], [4]]]]
        patches = [[[[1, 2, 3, 4], [2, 0, 4, 0]], [[3, 4, 0, 0], [4, 0, 0, 0]]]]
        self._VerifyValues(image, ksizes=[2, 2], strides=[1, 1], rates=[1, 1], padding='SAME', patches=patches)

    def testKsize2x2Stride1x1Rate2x2Valid(self):
        if False:
            i = 10
            return i + 15
        'Test for 2x2 kernel with 2x2 dilation.'
        image = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
        patches = [[[[0, 2, 8, 10], [1, 3, 9, 11]], [[4, 6, 12, 14], [5, 7, 13, 15]]]]
        self._VerifyValues(image, ksizes=[2, 2], strides=[1, 1], rates=[2, 2], padding='VALID', patches=patches)

    def testKsize2x2Stride1x1Rate1x1ValidDepth2(self):
        if False:
            print('Hello World!')
        'Test for 2x2 kernel with VALID padding.'
        image = [[[[1, 5], [2, 6]], [[3, 7], [4, 8]]]]
        patches = [[[[1, 5, 2, 6, 3, 7, 4, 8]]]]
        self._VerifyValues(image, ksizes=[2, 2], strides=[1, 1], rates=[1, 1], padding='VALID', patches=patches)
if __name__ == '__main__':
    test.main()