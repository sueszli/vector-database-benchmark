"""Tests for ExtractVolumePatches gradient."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class ExtractVolumePatchesGradTest(test.TestCase, parameterized.TestCase):
    """Gradient-checking for ExtractVolumePatches op."""

    @parameterized.parameters([{'in_shape': [2, 5, 5, 5, 3], 'ksizes': [1, 1, 1, 1, 1], 'strides': [1, 2, 3, 4, 1]}, {'in_shape': [2, 7, 7, 7, 3], 'ksizes': [1, 3, 3, 3, 1], 'strides': [1, 1, 1, 1, 1]}, {'in_shape': [2, 5, 7, 6, 3], 'ksizes': [1, 3, 2, 2, 1], 'strides': [1, 1, 1, 1, 1]}, {'in_shape': [2, 7, 8, 6, 3], 'ksizes': [1, 2, 3, 2, 1], 'strides': [1, 2, 4, 3, 1]}])
    def testGradient(self, in_shape, ksizes, strides):
        if False:
            while True:
                i = 10
        if test_util.is_gpu_available():
            self.skipTest('b/171837334: skip gpu test.')
        random_seed = 42
        random_seed_lib.set_random_seed(random_seed)
        with self.cached_session():
            np.random.seed(random_seed)
            input_val = constant_op.constant(np.random.random(in_shape), dtype=dtypes.float32)
            for padding in ['VALID', 'SAME']:

                def extract(in_val, ksizes=ksizes, strides=strides, padding=padding):
                    if False:
                        for i in range(10):
                            print('nop')
                    return array_ops.extract_volume_patches(in_val, ksizes, strides, padding)
                rtn = gradient_checker_v2.compute_gradient(extract, [input_val])
                err = gradient_checker_v2.max_error(*rtn)
                print('extract_volume_patches gradient err: %.4e' % err)
                self.assertLess(err, 0.0001)

    @parameterized.parameters(set((True, context.executing_eagerly())))
    def testConstructGradientWithLargeVolumes(self, use_tape):
        if False:
            i = 10
            return i + 15
        with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
            batch_size = 4
            planes = 8
            height = 32
            width = 32
            ksize = 5
            shape = (batch_size, planes, height, width, 1)
            volumes = variables.Variable(np.random.uniform(size=np.prod(shape)).reshape(shape), name='inputs')
            tape.watch(volumes)
            patches = array_ops.extract_volume_patches(volumes, ksizes=[1, ksize, ksize, ksize, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
            gradients = tape.gradient(patches, volumes)
            self.assertIsNotNone(gradients)
if __name__ == '__main__':
    test.main()