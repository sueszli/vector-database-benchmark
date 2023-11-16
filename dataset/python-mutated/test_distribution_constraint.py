import unittest
import numpy as np
import parameterize as param
import paddle
from paddle.distribution import constraint
np.random.seed(2022)

@param.param_cls((param.TEST_CASE_NAME, 'value'), [('NotImplement', np.random.rand(2, 3))])
class TestConstraint(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._constraint = constraint.Constraint()

    def test_costraint(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(NotImplementedError):
            self._constraint(self.value)

@param.param_cls((param.TEST_CASE_NAME, 'value', 'expect'), [('real', 1.0, True)])
class TestReal(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._constraint = constraint.Real()

    def test_costraint(self):
        if False:
            return 10
        self.assertEqual(self._constraint(self.value), self.expect)

@param.param_cls((param.TEST_CASE_NAME, 'lower', 'upper', 'value', 'expect'), [('in_range', 0, 1, 0.5, True), ('out_range', 0, 1, 2, False)])
class TestRange(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._constraint = constraint.Range(self.lower, self.upper)

    def test_costraint(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self._constraint(self.value), self.expect)

@param.param_cls((param.TEST_CASE_NAME, 'value', 'expect'), [('positive', 1, True), ('negative', -1, False)])
class TestPositive(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._constraint = constraint.Positive()

    def test_costraint(self):
        if False:
            print('Hello World!')
        self.assertEqual(self._constraint(self.value), self.expect)

@param.param_cls((param.TEST_CASE_NAME, 'value', 'expect'), [('simplex', paddle.to_tensor([0.5, 0.5]), True), ('non_simplex', paddle.to_tensor([-0.5, 0.5]), False)])
class TestSimplex(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._constraint = constraint.Simplex()

    def test_costraint(self):
        if False:
            return 10
        self.assertEqual(self._constraint(self.value), self.expect)
if __name__ == '__main__':
    unittest.main()