"""Tests for tensorflow.kernels.edit_distance_op."""
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

def ConstantOf(x):
    if False:
        while True:
            i = 10
    x = np.asarray(x)
    if x.dtype.char not in 'SU':
        x = np.asarray(x, dtype=np.int64)
    return constant_op.constant(x)

class EditDistanceTest(test.TestCase):

    def _testEditDistanceST(self, hypothesis_st, truth_st, normalize, expected_output, expected_shape, expected_err_re=None):
        if False:
            for i in range(10):
                print('nop')
        edit_distance = array_ops.edit_distance(hypothesis=hypothesis_st, truth=truth_st, normalize=normalize)
        if expected_err_re is None:
            self.assertEqual(edit_distance.get_shape(), expected_shape)
            output = self.evaluate(edit_distance)
            self.assertAllClose(output, expected_output)
        else:
            with self.assertRaisesOpError(expected_err_re):
                self.evaluate(edit_distance)

    def _testEditDistance(self, hypothesis, truth, normalize, expected_output, expected_err_re=None):
        if False:
            for i in range(10):
                print('nop')
        expected_shape = [max(h, t) for (h, t) in tuple(zip(hypothesis[2], truth[2]))[:-1]]
        with ops.Graph().as_default() as g, self.session(g):
            self._testEditDistanceST(hypothesis_st=sparse_tensor.SparseTensorValue(*[ConstantOf(x) for x in hypothesis]), truth_st=sparse_tensor.SparseTensorValue(*[ConstantOf(x) for x in truth]), normalize=normalize, expected_output=expected_output, expected_shape=expected_shape, expected_err_re=expected_err_re)
        with ops.Graph().as_default() as g, self.session(g):
            self._testEditDistanceST(hypothesis_st=sparse_tensor.SparseTensor(*[ConstantOf(x) for x in hypothesis]), truth_st=sparse_tensor.SparseTensor(*[ConstantOf(x) for x in truth]), normalize=normalize, expected_output=expected_output, expected_shape=expected_shape, expected_err_re=expected_err_re)

    def testEditDistanceNormalized(self):
        if False:
            print('Hello World!')
        hypothesis_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
        hypothesis_values = [0, 1, 1, -1]
        hypothesis_shape = [2, 2]
        truth_indices = [[0, 0], [1, 0], [1, 1]]
        truth_values = [0, 1, 1]
        truth_shape = [2, 2]
        expected_output = [1.0, 0.5]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=True, expected_output=expected_output)

    def testEditDistanceUnnormalized(self):
        if False:
            i = 10
            return i + 15
        hypothesis_indices = [[0, 0], [1, 0], [1, 1]]
        hypothesis_values = [10, 10, 11]
        hypothesis_shape = [2, 2]
        truth_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
        truth_values = [1, 2, 1, -1]
        truth_shape = [2, 3]
        expected_output = [2.0, 2.0]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=False, expected_output=expected_output)

    def testEditDistanceProperDistance(self):
        if False:
            return 10
        hypothesis_indices = [[0, i] for (i, _) in enumerate('algorithm')] + [[1, i] for (i, _) in enumerate('altruistic')]
        hypothesis_values = [x for x in 'algorithm'] + [x for x in 'altruistic']
        hypothesis_shape = [2, 11]
        truth_indices = [[0, i] for (i, _) in enumerate('altruistic')] + [[1, i] for (i, _) in enumerate('algorithm')]
        truth_values = [x for x in 'altruistic'] + [x for x in 'algorithm']
        truth_shape = [2, 11]
        expected_unnormalized = [6.0, 6.0]
        expected_normalized = [6.0 / len('altruistic'), 6.0 / len('algorithm')]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=False, expected_output=expected_unnormalized)
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=True, expected_output=expected_normalized)

    def testEditDistance3D(self):
        if False:
            print('Hello World!')
        hypothesis_indices = [[0, 0, 0], [1, 0, 0]]
        hypothesis_values = [0, 1]
        hypothesis_shape = [2, 1, 1]
        truth_indices = [[0, 1, 0], [1, 0, 0], [1, 1, 0]]
        truth_values = [0, 1, 1]
        truth_shape = [2, 2, 1]
        expected_output = [[np.inf, 1.0], [0.0, 1.0]]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=True, expected_output=expected_output)

    def testEditDistanceZeroLengthHypothesis(self):
        if False:
            i = 10
            return i + 15
        hypothesis_indices = np.empty((0, 2), dtype=np.int64)
        hypothesis_values = []
        hypothesis_shape = [1, 0]
        truth_indices = [[0, 0]]
        truth_values = [0]
        truth_shape = [1, 1]
        expected_output = [1.0]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=True, expected_output=expected_output)

    def testEditDistanceZeroLengthTruth(self):
        if False:
            for i in range(10):
                print('nop')
        hypothesis_indices = [[0, 0]]
        hypothesis_values = [0]
        hypothesis_shape = [1, 1]
        truth_indices = np.empty((0, 2), dtype=np.int64)
        truth_values = []
        truth_shape = [1, 0]
        expected_output = [np.inf]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=True, expected_output=expected_output)

    def testEditDistanceZeroLengthHypothesisAndTruth(self):
        if False:
            i = 10
            return i + 15
        hypothesis_indices = np.empty((0, 2), dtype=np.int64)
        hypothesis_values = []
        hypothesis_shape = [1, 0]
        truth_indices = np.empty((0, 2), dtype=np.int64)
        truth_values = []
        truth_shape = [1, 0]
        expected_output = [0]
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=True, expected_output=expected_output)

    def testEditDistanceBadIndices(self):
        if False:
            for i in range(10):
                print('nop')
        hypothesis_indices = np.full((3, 3), -1250999896764, dtype=np.int64)
        hypothesis_values = np.zeros(3, dtype=np.int64)
        hypothesis_shape = np.zeros(3, dtype=np.int64)
        truth_indices = np.full((3, 3), -1250999896764, dtype=np.int64)
        truth_values = np.full([3], 2, dtype=np.int64)
        truth_shape = np.full([3], 2, dtype=np.int64)
        expected_output = []
        self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=False, expected_output=expected_output, expected_err_re='inner product -\\d+ which would require writing to outside of the buffer for the output tensor|Dimension -\\d+ must be >= 0')

    def testEmptyShapeWithEditDistanceRaisesError(self):
        if False:
            print('Hello World!')
        para = {'hypothesis_indices': [[]], 'hypothesis_values': ['tmp/'], 'hypothesis_shape': [], 'truth_indices': [[]], 'truth_values': [''], 'truth_shape': [], 'normalize': False}
        with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError), 'Input Hypothesis SparseTensors must have rank at least 2, but hypothesis_shape rank is: 0|Input SparseTensors must have rank at least 2, but truth_shape rank is: 0'):
            array_ops.gen_array_ops.EditDistance(**para)

        @def_function.function
        def TestFunction():
            if False:
                i = 10
                return i + 15
            'Wrapper function for edit distance call.'
            array_ops.gen_array_ops.EditDistance(**para)
        with self.assertRaisesRegex(ValueError, 'Input Hypothesis SparseTensors must have rank at least 2, but hypothesis_shape rank is: 0'):
            TestFunction()
        hypothesis_indices = [[]]
        hypothesis_values = [0]
        hypothesis_shape = []
        truth_indices = [[]]
        truth_values = [1]
        truth_shape = []
        expected_output = []
        with self.assertRaisesRegex(ValueError, 'Input Hypothesis SparseTensors must have rank at least 2, but hypothesis_shape rank is: 0'):
            self._testEditDistance(hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape), truth=(truth_indices, truth_values, truth_shape), normalize=False, expected_output=expected_output)
if __name__ == '__main__':
    test.main()