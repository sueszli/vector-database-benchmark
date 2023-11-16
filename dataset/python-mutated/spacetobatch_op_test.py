"""Functional tests for SpaceToBatch and BatchToSpace ops."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test

def space_to_batch_direct(input_array, block_shape, paddings):
    if False:
        while True:
            i = 10
    'Direct Python implementation of space-to-batch conversion.\n\n  This is used for tests only.\n\n  Args:\n    input_array: N-D array\n    block_shape: 1-D array of shape [num_block_dims].\n    paddings: 2-D array of shape [num_block_dims, 2].\n\n  Returns:\n    Converted tensor.\n  '
    input_array = np.array(input_array)
    block_shape = np.array(block_shape)
    num_block_dims = len(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    padded = np.pad(input_array, pad_width=[[0, 0]] + list(paddings) + [[0, 0]] * (input_array.ndim - 1 - num_block_dims), mode='constant')
    reshaped_padded_shape = [input_array.shape[0]]
    output_shape = [input_array.shape[0] * np.prod(block_shape)]
    for (block_dim, block_shape_value) in enumerate(block_shape):
        reduced_size = padded.shape[block_dim + 1] // block_shape_value
        reshaped_padded_shape.append(reduced_size)
        output_shape.append(reduced_size)
        reshaped_padded_shape.append(block_shape_value)
    reshaped_padded_shape.extend(input_array.shape[num_block_dims + 1:])
    output_shape.extend(input_array.shape[num_block_dims + 1:])
    reshaped_padded = padded.reshape(reshaped_padded_shape)
    permuted_reshaped_padded = np.transpose(reshaped_padded, list(np.arange(num_block_dims) * 2 + 2) + [0] + list(np.arange(num_block_dims) * 2 + 1) + list(np.arange(input_array.ndim - num_block_dims - 1) + 1 + num_block_dims * 2))
    return permuted_reshaped_padded.reshape(output_shape)

class SpaceToBatchTest(xla_test.XLATestCase):
    """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops."""

    def _testPad(self, inputs, paddings, block_size, outputs):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            for dtype in self.float_types:
                placeholder = array_ops.placeholder(dtype)
                x_tf = gen_array_ops.space_to_batch(placeholder, paddings, block_size=block_size)
                self.assertAllEqual(sess.run(x_tf, {placeholder: inputs}), outputs)
                x_tf = gen_array_ops.batch_to_space(placeholder, paddings, block_size=block_size)
                self.assertAllEqual(sess.run(x_tf, {placeholder: outputs}), inputs)

    def _testOne(self, inputs, block_size, outputs):
        if False:
            while True:
                i = 10
        paddings = np.zeros((2, 2), dtype=np.int32)
        self._testPad(inputs, paddings, block_size, outputs)

    def testSmallInput2x2(self):
        if False:
            return 10
        x_np = [[[[1], [2]], [[3], [4]]]]
        block_size = 2
        x_out = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
        self._testOne(x_np, block_size, x_out)

    def testSmallInput2x2Pad1x0(self):
        if False:
            print('Hello World!')
        x_np = [[[[1], [2]], [[3], [4]]]]
        paddings = np.array([[1, 0], [1, 0]], dtype=np.int32)
        block_size = 3
        x_out = [[[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[2]]], [[[0]]], [[[3]]], [[[4]]]]
        self._testPad(x_np, paddings, block_size, x_out)

    def testDepthInput2x2(self):
        if False:
            for i in range(10):
                print('nop')
        x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
        block_size = 2
        x_out = [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
        self._testOne(x_np, block_size, x_out)

    def testLargerInput2x2(self):
        if False:
            i = 10
            return i + 15
        x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
        block_size = 2
        x_out = [[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]], [[[5], [7]], [[13], [15]]], [[[6], [8]], [[14], [16]]]]
        self._testOne(x_np, block_size, x_out)

    def testBatchInput2x2(self):
        if False:
            return 10
        x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]]], [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
        block_size = 2
        x_out = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]], [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
        self._testOne(x_np, block_size, x_out)

    def testLargerInputBatch2x2(self):
        if False:
            while True:
                i = 10
        x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]], [[13], [14], [15], [16]]], [[[17], [18], [19], [20]], [[21], [22], [23], [24]], [[25], [26], [27], [28]], [[29], [30], [31], [32]]]]
        x_out = [[[[1], [3]], [[9], [11]]], [[[17], [19]], [[25], [27]]], [[[2], [4]], [[10], [12]]], [[[18], [20]], [[26], [28]]], [[[5], [7]], [[13], [15]]], [[[21], [23]], [[29], [31]]], [[[6], [8]], [[14], [16]]], [[[22], [24]], [[30], [32]]]]
        block_size = 2
        self._testOne(x_np, block_size, x_out)

class SpaceToBatchNDErrorHandlingTest(xla_test.XLATestCase):

    def testInvalidBlockShape(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'block_shape must be positive'):
            with self.session() as sess, self.test_scope():
                tf_in = constant_op.constant(-3.5e+35, shape=[10, 20, 20], dtype=dtypes.float32)
                block_shape = constant_op.constant(-10, shape=[2], dtype=dtypes.int64)
                paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
                sess.run(array_ops.space_to_batch_nd(tf_in, block_shape, paddings))

    def testOutputSizeOutOfBounds(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'Negative.* dimension size caused by overflow'):
            with self.session() as sess, self.test_scope():
                tf_in = constant_op.constant(-3.5e+35, shape=[10, 19, 22], dtype=dtypes.float32)
                block_shape = constant_op.constant(1879048192, shape=[2], dtype=dtypes.int64)
                paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
                sess.run(array_ops.space_to_batch_nd(tf_in, block_shape, paddings))

class SpaceToBatchNDTest(xla_test.XLATestCase):
    """Tests input-output pairs for the SpaceToBatchND and BatchToSpaceND ops."""

    def _testPad(self, inputs, block_shape, paddings, outputs):
        if False:
            for i in range(10):
                print('nop')
        block_shape = np.array(block_shape)
        paddings = np.array(paddings).reshape((len(block_shape), 2))
        with self.session() as sess, self.test_scope():
            for dtype in self.float_types:
                if dtype == dtypes.bfloat16.as_numpy_dtype:
                    continue
                if dtype == np.float16:
                    actual_inputs = np.array(inputs).astype(dtype)
                    actual_paddings = np.array(paddings).astype(dtype)
                    expected_outputs = np.array(outputs).astype(dtype)
                else:
                    actual_inputs = inputs
                    actual_paddings = paddings
                    expected_outputs = outputs
                placeholder = array_ops.placeholder(dtype)
                x_tf = array_ops.space_to_batch_nd(placeholder, block_shape, actual_paddings)
                self.assertAllEqual(sess.run(x_tf, {placeholder: actual_inputs}), expected_outputs)
                placeholder = array_ops.placeholder(dtype)
                x_tf = array_ops.batch_to_space_nd(placeholder, block_shape, actual_paddings)
                self.assertAllEqual(sess.run(x_tf, {placeholder: expected_outputs}), actual_inputs)

    def _testDirect(self, input_shape, block_shape, paddings):
        if False:
            i = 10
            return i + 15
        inputs = np.arange(np.prod(input_shape), dtype=np.float32)
        inputs = inputs.reshape(input_shape)
        self._testPad(inputs, block_shape, paddings, space_to_batch_direct(inputs, block_shape, paddings))

    def testZeroBlockDimsZeroRemainingDims(self):
        if False:
            print('Hello World!')
        self._testPad(inputs=[1, 2], block_shape=[], paddings=[], outputs=[1, 2])

    def testZeroBlockDimsOneRemainingDim(self):
        if False:
            return 10
        self._testPad(inputs=[[1, 2], [3, 4]], block_shape=[], paddings=[], outputs=[[1, 2], [3, 4]])
        self._testPad(inputs=[[1, 2], [3, 4]], block_shape=[1], paddings=[[0, 0]], outputs=[[1, 2], [3, 4]])

    def testZeroBlockDimsTwoRemainingDims(self):
        if False:
            while True:
                i = 10
        self._testPad(inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], block_shape=[], paddings=[], outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self._testPad(inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], block_shape=[1], paddings=[[0, 0]], outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self._testPad(inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], block_shape=[1, 1], paddings=[[0, 0], [0, 0]], outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def testOneBlockDimZeroRemainingDims(self):
        if False:
            for i in range(10):
                print('nop')
        self._testPad(inputs=[[1, 2, 3], [4, 5, 6]], block_shape=[2], paddings=[1, 0], outputs=[[0, 2], [0, 5], [1, 3], [4, 6]])

    def testOneBlockDimOneRemainingDim(self):
        if False:
            return 10
        self._testPad(inputs=[[[1, 11], [2, 21], [3, 31]], [[4, 41], [5, 51], [6, 61]]], block_shape=[2], paddings=[1, 0], outputs=[[[0, 0], [2, 21]], [[0, 0], [5, 51]], [[1, 11], [3, 31]], [[4, 41], [6, 61]]])

    def testDirect0(self):
        if False:
            i = 10
            return i + 15
        self._testDirect(input_shape=[3, 1, 2, 0], block_shape=[3], paddings=[[0, 2]])

    def testDirect1(self):
        if False:
            for i in range(10):
                print('nop')
        self._testDirect(input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[0, 0]])

    def testDirect2(self):
        if False:
            while True:
                i = 10
        self._testDirect(input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[1, 2]])

    def testDirect3(self):
        if False:
            while True:
                i = 10
        self._testDirect(input_shape=[3, 3, 4, 5, 2], block_shape=[3, 4, 2], paddings=[[1, 2], [0, 0], [3, 0]])

    def testDirect4(self):
        if False:
            while True:
                i = 10
        self._testDirect(input_shape=[3, 3, 4, 5, 2], block_shape=[3, 4, 2, 2], paddings=[[1, 2], [0, 0], [3, 0], [0, 0]])

    def testDirect5(self):
        if False:
            i = 10
            return i + 15
        self._testDirect(input_shape=[3, 2, 2, 3, 4, 5, 2, 5], block_shape=[1, 1, 3, 4, 2, 2], paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0]])

    def testDirect6(self):
        if False:
            return 10
        self._testDirect(input_shape=[3, 2, 2, 3, 4, 5, 2, 5], block_shape=[1, 1, 3, 4, 2, 2, 1], paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0], [0, 0]])
if __name__ == '__main__':
    test.main()