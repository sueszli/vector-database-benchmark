"""Tests for ReduceJoin op from string_ops."""
import itertools
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

def _input_array(num_dims):
    if False:
        for i in range(10):
            print('nop')
    'Creates an ndarray where each element is the binary of its linear index.\n\n  Args:\n    num_dims: The number of dimensions to create.\n\n  Returns:\n    An ndarray of shape [2] * num_dims.\n  '
    formatter = '{:0%db}' % num_dims
    strings = [formatter.format(i) for i in range(2 ** num_dims)]
    return np.array(strings, dtype='S%d' % num_dims).reshape([2] * num_dims)

def _joined_array(num_dims, reduce_dim):
    if False:
        while True:
            i = 10
    'Creates an ndarray with the result from reduce_join on input_array.\n\n  Args:\n    num_dims: The number of dimensions of the original input array.\n    reduce_dim: The dimension to reduce.\n\n  Returns:\n    An ndarray of shape [2] * (num_dims - 1).\n  '
    formatter = '{:0%db}' % (num_dims - 1)
    result = np.zeros(shape=[2] * (num_dims - 1), dtype='S%d' % (2 * num_dims))
    flat = result.ravel()
    for i in range(2 ** (num_dims - 1)):
        dims = formatter.format(i)
        flat[i] = ''.join([(dims[:reduce_dim] + '%d' + dims[reduce_dim:]) % j for j in range(2)])
    return result

class UnicodeTestCase(test.TestCase):
    """Test case with Python3-compatible string comparator."""

    def assertAllEqualUnicode(self, truth, actual):
        if False:
            return 10
        self.assertAllEqual(np.array(truth).astype('U'), np.array(actual).astype('U'))

class ReduceJoinTestHelperTest(UnicodeTestCase):
    """Tests for helper functions."""

    def testInputArray(self):
        if False:
            return 10
        num_dims = 3
        truth = ['{:03b}'.format(i) for i in range(2 ** num_dims)]
        output_array = _input_array(num_dims).reshape([-1])
        self.assertAllEqualUnicode(truth, output_array)

    def testJoinedArray(self):
        if False:
            i = 10
            return i + 15
        num_dims = 3
        truth_dim_zero = [['000100', '001101'], ['010110', '011111']]
        truth_dim_one = [['000010', '001011'], ['100110', '101111']]
        truth_dim_two = [['000001', '010011'], ['100101', '110111']]
        output_array_dim_zero = _joined_array(num_dims, reduce_dim=0)
        output_array_dim_one = _joined_array(num_dims, reduce_dim=1)
        output_array_dim_two = _joined_array(num_dims, reduce_dim=2)
        self.assertAllEqualUnicode(truth_dim_zero, output_array_dim_zero)
        self.assertAllEqualUnicode(truth_dim_one, output_array_dim_one)
        self.assertAllEqualUnicode(truth_dim_two, output_array_dim_two)

class ReduceJoinTest(UnicodeTestCase):

    def _testReduceJoin(self, input_array, truth, truth_shape, axis, keep_dims=False, separator=''):
        if False:
            i = 10
            return i + 15
        'Compares the output of reduce_join to an expected result.\n\n    Args:\n      input_array: The string input to be joined.\n      truth: An array or np.array of the expected result.\n      truth_shape: An array or np.array of the expected shape.\n      axis: The indices to reduce over.\n      keep_dims: Whether or not to retain reduced dimensions.\n      separator: The separator to use for joining.\n    '
        with self.cached_session():
            output = string_ops.reduce_join(inputs=input_array, axis=axis, keep_dims=keep_dims, separator=separator)
            output_array = self.evaluate(output)
        self.assertAllEqualUnicode(truth, output_array)
        self.assertAllEqual(truth_shape, output.get_shape())

    def _testMultipleReduceJoin(self, input_array, axis, separator=' '):
        if False:
            return 10
        'Tests reduce_join for one input and multiple axes.\n\n    Does so by comparing the output to that from nested reduce_string_joins.\n    The correctness of single-dimension reduce_join is verified by other\n    tests below using _testReduceJoin.\n\n    Args:\n      input_array: The input to test.\n      axis: The indices to reduce.\n      separator: The separator to use when joining.\n    '
        with self.cached_session():
            output = string_ops.reduce_join(inputs=input_array, axis=axis, keep_dims=False, separator=separator)
            output_keep_dims = string_ops.reduce_join(inputs=input_array, axis=axis, keep_dims=True, separator=separator)
            truth = input_array
            for index in axis:
                truth = string_ops.reduce_join(inputs=truth, axis=index, keep_dims=True, separator=separator)
            if not axis:
                truth = constant_op.constant(truth)
            truth_squeezed = array_ops.squeeze(truth, axis=axis)
            output_array = self.evaluate(output)
            output_keep_dims_array = self.evaluate(output_keep_dims)
            truth_array = self.evaluate(truth)
            truth_squeezed_array = self.evaluate(truth_squeezed)
        self.assertAllEqualUnicode(truth_array, output_keep_dims_array)
        self.assertAllEqualUnicode(truth_squeezed_array, output_array)
        self.assertAllEqual(truth.get_shape(), output_keep_dims.get_shape())
        self.assertAllEqual(truth_squeezed.get_shape(), output.get_shape())

    def testRankOne(self):
        if False:
            return 10
        input_array = ['this', 'is', 'a', 'test']
        truth = 'thisisatest'
        truth_shape = []
        self._testReduceJoin(input_array, truth, truth_shape, axis=0)

    def testRankTwo(self):
        if False:
            for i in range(10):
                print('nop')
        input_array = [['this', 'is', 'a', 'test'], ['please', 'do', 'not', 'panic']]
        truth_dim_zero = ['thisplease', 'isdo', 'anot', 'testpanic']
        truth_shape_dim_zero = [4]
        truth_dim_one = ['thisisatest', 'pleasedonotpanic']
        truth_shape_dim_one = [2]
        self._testReduceJoin(input_array, truth_dim_zero, truth_shape_dim_zero, axis=0)
        self._testReduceJoin(input_array, truth_dim_one, truth_shape_dim_one, axis=1)
        expected_val = 'thisisatestpleasedonotpanic'
        expected_shape = []
        self._testReduceJoin(input_array, expected_val, expected_shape, axis=None)
        expected_val = input_array
        expected_shape = [2, 4]
        self._testReduceJoin(input_array, expected_val, expected_shape, axis=[])

    def testRankFive(self):
        if False:
            i = 10
            return i + 15
        input_array = _input_array(num_dims=5)
        truths = [_joined_array(num_dims=5, reduce_dim=i) for i in range(5)]
        truth_shape = [2] * 4
        for i in range(5):
            self._testReduceJoin(input_array, truths[i], truth_shape, axis=i)

    def testNegative(self):
        if False:
            while True:
                i = 10
        input_array = _input_array(num_dims=5)
        truths = [_joined_array(num_dims=5, reduce_dim=i) for i in range(5)]
        truth_shape = [2] * 4
        for i in range(5):
            self._testReduceJoin(input_array, truths[i], truth_shape, axis=i - 5)

    def testSingletonDimension(self):
        if False:
            print('Hello World!')
        input_arrays = [_input_array(num_dims=5).reshape([2] * i + [1] + [2] * (5 - i)) for i in range(6)]
        truth = _input_array(num_dims=5)
        truth_shape = [2] * 5
        for i in range(6):
            self._testReduceJoin(input_arrays[i], truth, truth_shape, axis=i)

    def testSeparator(self):
        if False:
            i = 10
            return i + 15
        input_array = [['this', 'is', 'a', 'test'], ['please', 'do', 'not', 'panic']]
        truth_dim_zero = ['this  please', 'is  do', 'a  not', 'test  panic']
        truth_shape_dim_zero = [4]
        truth_dim_one = ['this  is  a  test', 'please  do  not  panic']
        truth_shape_dim_one = [2]
        self._testReduceJoin(input_array, truth_dim_zero, truth_shape_dim_zero, axis=0, separator='  ')
        self._testReduceJoin(input_array, truth_dim_one, truth_shape_dim_one, axis=1, separator='  ')

    @test_util.run_deprecated_v1
    def testUnknownShape(self):
        if False:
            while True:
                i = 10
        input_array = [['a'], ['b']]
        truth = ['ab']
        truth_shape = None
        with self.cached_session():
            placeholder = array_ops.placeholder(dtypes.string, name='placeholder')
            reduced = string_ops.reduce_join(placeholder, axis=0)
            output_array = reduced.eval(feed_dict={placeholder.name: input_array})
            self.assertAllEqualUnicode(truth, output_array)
            self.assertAllEqual(truth_shape, reduced.get_shape())

    @test_util.run_deprecated_v1
    def testUnknownIndices(self):
        if False:
            return 10
        input_array = [['this', 'is', 'a', 'test'], ['please', 'do', 'not', 'panic']]
        truth_dim_zero = ['thisplease', 'isdo', 'anot', 'testpanic']
        truth_dim_one = ['thisisatest', 'pleasedonotpanic']
        truth_shape = None
        with self.cached_session():
            placeholder = array_ops.placeholder(dtypes.int32, name='placeholder')
            reduced = string_ops.reduce_join(input_array, axis=placeholder)
            output_array_dim_zero = reduced.eval(feed_dict={placeholder.name: [0]})
            output_array_dim_one = reduced.eval(feed_dict={placeholder.name: [1]})
            self.assertAllEqualUnicode(truth_dim_zero, output_array_dim_zero)
            self.assertAllEqualUnicode(truth_dim_one, output_array_dim_one)
            self.assertAllEqual(truth_shape, reduced.get_shape())

    def testKeepDims(self):
        if False:
            i = 10
            return i + 15
        input_array = [['this', 'is', 'a', 'test'], ['please', 'do', 'not', 'panic']]
        truth_dim_zero = [['thisplease', 'isdo', 'anot', 'testpanic']]
        truth_shape_dim_zero = [1, 4]
        truth_dim_one = [['thisisatest'], ['pleasedonotpanic']]
        truth_shape_dim_one = [2, 1]
        self._testReduceJoin(input_array, truth_dim_zero, truth_shape_dim_zero, axis=0, keep_dims=True)
        self._testReduceJoin(input_array, truth_dim_one, truth_shape_dim_one, axis=1, keep_dims=True)
        expected_val = [['thisisatestpleasedonotpanic']]
        expected_shape = [1, 1]
        self._testReduceJoin(constant_op.constant(input_array), expected_val, expected_shape, keep_dims=True, axis=None)
        expected_val = input_array
        expected_shape = [2, 4]
        self._testReduceJoin(input_array, expected_val, expected_shape, keep_dims=True, axis=[])

    def testMultiIndex(self):
        if False:
            for i in range(10):
                print('nop')
        num_dims = 3
        input_array = _input_array(num_dims=num_dims)
        for i in range(num_dims + 1):
            for permutation in itertools.permutations(range(num_dims), i):
                self._testMultipleReduceJoin(input_array, axis=permutation)

    @test_util.run_deprecated_v1
    def testInvalidReductionIndices(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with self.assertRaisesRegex(ValueError, 'Invalid reduction dim'):
                string_ops.reduce_join(inputs='', axis=0)
            with self.assertRaisesRegex(ValueError, 'Invalid reduction dimension -3'):
                string_ops.reduce_join(inputs=[['']], axis=-3)
            with self.assertRaisesRegex(ValueError, 'Invalid reduction dimension 2'):
                string_ops.reduce_join(inputs=[['']], axis=2)
            with self.assertRaisesRegex(ValueError, 'Invalid reduction dimension -3'):
                string_ops.reduce_join(inputs=[['']], axis=[0, -3])
            with self.assertRaisesRegex(ValueError, 'Invalid reduction dimension 2'):
                string_ops.reduce_join(inputs=[['']], axis=[0, 2])

    def testZeroDims(self):
        if False:
            return 10
        with self.cached_session():
            inputs = np.zeros([0, 1], dtype=str)
            output = string_ops.reduce_join(inputs=inputs, axis=0)
            self.assertAllEqualUnicode([''], self.evaluate(output))
            output = string_ops.reduce_join(inputs=inputs, axis=1)
            output_shape = self.evaluate(output).shape
            self.assertAllEqual([0], output_shape)

    @test_util.run_deprecated_v1
    def testInvalidArgsUnknownShape(self):
        if False:
            return 10
        with self.cached_session():
            placeholder = array_ops.placeholder(dtypes.string, name='placeholder')
            index_too_high = string_ops.reduce_join(placeholder, axis=1)
            duplicate_index = string_ops.reduce_join(placeholder, axis=[-1, 1])
            with self.assertRaisesOpError('Invalid reduction dimension 1'):
                index_too_high.eval(feed_dict={placeholder.name: ['']})
            with self.assertRaisesOpError('Duplicate reduction dimension 1'):
                duplicate_index.eval(feed_dict={placeholder.name: [['']]})

    @test_util.run_deprecated_v1
    def testInvalidArgsUnknownIndices(self):
        if False:
            return 10
        with self.cached_session():
            placeholder = array_ops.placeholder(dtypes.int32, name='placeholder')
            reduced = string_ops.reduce_join(['test', 'test2'], axis=placeholder)
            with self.assertRaisesOpError('reduction dimension -2'):
                reduced.eval(feed_dict={placeholder.name: -2})
            with self.assertRaisesOpError('reduction dimension 2'):
                reduced.eval(feed_dict={placeholder.name: 2})

    def testDeprecatedArgs(self):
        if False:
            for i in range(10):
                print('nop')
        foobar = constant_op.constant(['foobar'])
        output = string_ops.reduce_join(['foo', 'bar'], reduction_indices=0, keep_dims=True)
        self.assertAllEqual(foobar, output)
        output = string_ops.reduce_join(['foo', 'bar'], axis=0, keepdims=True)
        self.assertAllEqual(foobar, output)
if __name__ == '__main__':
    test.main()