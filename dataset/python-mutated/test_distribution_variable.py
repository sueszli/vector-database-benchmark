import unittest
import parameterize as param
import paddle
from paddle.distribution import constraint, variable
paddle.seed(2022)

@param.param_cls((param.TEST_CASE_NAME, 'is_discrete', 'event_rank', 'constraint'), [('NotImplement', False, 0, constraint.Constraint())])
class TestVariable(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._var = variable.Variable(self.is_discrete, self.event_rank, self.constraint)

    @param.param_func([(1,)])
    def test_costraint(self, value):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(NotImplementedError):
            self._var.constraint(value)

@param.param_cls((param.TEST_CASE_NAME, 'base', 'rank'), [('real_base', variable.real, 10)])
class TestIndependent(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._var = variable.Independent(self.base, self.rank)

    @param.param_func([(paddle.rand([2, 3, 4]), ValueError)])
    def test_costraint(self, value, expect):
        if False:
            return 10
        with self.assertRaises(expect):
            self._var.constraint(value)

@param.param_cls((param.TEST_CASE_NAME, 'vars', 'axis'), [('real_base', [variable.real], 10)])
class TestStack(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._var = variable.Stack(self.vars, self.axis)

    def test_is_discrete(self):
        if False:
            print('Hello World!')
        self.assertEqual(self._var.is_discrete, False)

    @param.param_func([(paddle.rand([2, 3, 4]), ValueError)])
    def test_costraint(self, value, expect):
        if False:
            print('Hello World!')
        with self.assertRaises(expect):
            self._var.constraint(value)
if __name__ == '__main__':
    unittest.main()