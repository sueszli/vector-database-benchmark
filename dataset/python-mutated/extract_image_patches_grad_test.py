"""Tests for ExtractImagePatches gradient."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class ExtractImagePatchesGradTest(test.TestCase, parameterized.TestCase):
    """Gradient-checking for ExtractImagePatches op."""
    _TEST_CASES = [{'in_shape': [2, 5, 5, 3], 'ksizes': [1, 1, 1, 1], 'strides': [1, 2, 3, 1], 'rates': [1, 1, 1, 1]}, {'in_shape': [2, 7, 7, 3], 'ksizes': [1, 3, 3, 1], 'strides': [1, 1, 1, 1], 'rates': [1, 1, 1, 1]}, {'in_shape': [2, 8, 7, 3], 'ksizes': [1, 2, 2, 1], 'strides': [1, 1, 1, 1], 'rates': [1, 1, 1, 1]}, {'in_shape': [2, 7, 8, 3], 'ksizes': [1, 3, 2, 1], 'strides': [1, 4, 3, 1], 'rates': [1, 1, 1, 1]}, {'in_shape': [1, 15, 20, 3], 'ksizes': [1, 4, 3, 1], 'strides': [1, 1, 1, 1], 'rates': [1, 2, 4, 1]}, {'in_shape': [2, 7, 8, 1], 'ksizes': [1, 3, 2, 1], 'strides': [1, 3, 2, 1], 'rates': [1, 2, 2, 1]}, {'in_shape': [2, 8, 9, 4], 'ksizes': [1, 2, 2, 1], 'strides': [1, 4, 2, 1], 'rates': [1, 3, 2, 1]}]

    def testGradient(self):
        if False:
            i = 10
            return i + 15
        random_seed = 42
        random_seed_lib.set_random_seed(random_seed)
        with self.cached_session():
            for test_case in self._TEST_CASES:
                np.random.seed(random_seed)
                in_shape = test_case['in_shape']
                in_val = constant_op.constant(np.random.random(in_shape), dtype=dtypes.float32)
                ksizes = tuple(test_case['ksizes'])
                strides = tuple(test_case['strides'])
                rates = tuple(test_case['rates'])
                for padding in ['VALID', 'SAME']:

                    def extract(in_val, ksizes=ksizes, strides=strides, rates=rates, padding=padding):
                        if False:
                            for i in range(10):
                                print('nop')
                        return array_ops.extract_image_patches(in_val, ksizes, strides, rates, padding)
                    err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(extract, [in_val]))
                    self.assertLess(err, 0.0001)

    @parameterized.parameters(set((True, context.executing_eagerly())))
    def testConstructGradientWithLargeImages(self, use_tape):
        if False:
            i = 10
            return i + 15
        with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
            batch_size = 4
            height = 512
            width = 512
            ksize = 5
            shape = (batch_size, height, width, 1)
            images = variables.Variable(np.random.uniform(size=np.prod(shape)).reshape(shape), name='inputs')
            tape.watch(images)
            patches = array_ops.extract_image_patches(images, ksizes=[1, ksize, ksize, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
            gradients = tape.gradient(patches, images)
            self.assertIsNotNone(gradients)

    def _VariableShapeGradient(self, test_shape_pattern):
        if False:
            for i in range(10):
                print('nop')
        'Use test_shape_pattern to infer which dimensions are of\n\n    variable size.\n    '
        with ops.Graph().as_default():
            random_seed = 42
            random_seed_lib.set_random_seed(random_seed)
            with self.test_session():
                for test_case in self._TEST_CASES:
                    np.random.seed(random_seed)
                    in_shape = test_case['in_shape']
                    test_shape = [x if x is None else y for (x, y) in zip(test_shape_pattern, in_shape)]
                    in_val = array_ops.placeholder(shape=test_shape, dtype=dtypes.float32)
                    feed_dict = {in_val: np.random.random(in_shape)}
                    for padding in ['VALID', 'SAME']:
                        out_val = array_ops.extract_image_patches(in_val, test_case['ksizes'], test_case['strides'], test_case['rates'], padding)
                        out_val_tmp = out_val.eval(feed_dict=feed_dict)
                        out_shape = out_val_tmp.shape
                        err = gradient_checker.compute_gradient_error(in_val, in_shape, out_val, out_shape)
                        self.assertLess(err, 0.0001)

    def test_BxxC_Gradient(self):
        if False:
            print('Hello World!')
        self._VariableShapeGradient([-1, None, None, -1])

    def test_xHWx_Gradient(self):
        if False:
            for i in range(10):
                print('nop')
        self._VariableShapeGradient([None, -1, -1, None])

    def test_BHWC_Gradient(self):
        if False:
            i = 10
            return i + 15
        self._VariableShapeGradient([-1, -1, -1, -1])

    def test_AllNone_Gradient(self):
        if False:
            i = 10
            return i + 15
        self._VariableShapeGradient([None, None, None, None])

    def testJitCompile(self):
        if False:
            print('Hello World!')
        with test_util.AbstractGradientTape(use_tape=True) as tape:
            shape = (4, 512, 512, 1)
            ksize = 5
            images = variables.Variable(np.random.uniform(size=shape), name='inputs')
            tape.watch(images)
            extract_image_patches_jit = polymorphic_function.function(array_ops.extract_image_patches, jit_compile=True)
            patches = extract_image_patches_jit(images, ksizes=[1, ksize, ksize, 1], strides=[1] * 4, rates=[1] * 4, padding='SAME')
            gradients = tape.gradient(patches, images)
            self.assertIsNotNone(gradients)
if __name__ == '__main__':
    test.main()