import unittest
import numpy as np
from Orange.classification import SimpleTreeLearner
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table

class SimpleTreeTest(unittest.TestCase):

    def test_nonan_classification(self):
        if False:
            for i in range(10):
                print('nop')
        x = ContinuousVariable('x')
        y = DiscreteVariable('y', values=tuple('ab'))
        d = Domain([x], y)
        t = Table.from_numpy(d, [[0]], [np.nan])
        m = SimpleTreeLearner()(t)
        self.assertFalse(np.isnan(m(t)[0]))

    def test_nonan_regression(self):
        if False:
            for i in range(10):
                print('nop')
        x = ContinuousVariable('x')
        y = ContinuousVariable('y')
        d = Domain([x], y)
        t = Table.from_numpy(d, [[42]], [np.nan])
        m = SimpleTreeLearner()(t)
        self.assertFalse(np.isnan(m(t)[0]))
        self.assertEqual(m(t)[0], 0)
        x2 = ContinuousVariable('x2')
        d = Domain([x, x2], y)
        t = Table.from_numpy(d, [[-1, np.nan], [1, -1], [1, 1]], [-20, 20, np.nan])
        m = SimpleTreeLearner(min_instances=1)(t)
        self.assertFalse(np.isnan(m(t)[0]))
        np.testing.assert_equal(m(t), [-20, 20, 20])

    def test_stub(self):
        if False:
            return 10
        x = ContinuousVariable('x')
        y = ContinuousVariable('y')
        d = Domain([x], y)
        t = Table.from_numpy(d, [[-1], [1]], [-5, 0])
        m = SimpleTreeLearner(min_instances=1)(t)
        np.testing.assert_equal(m(t), [-5, 0])
        m = SimpleTreeLearner()(t)
        np.testing.assert_equal(m(t), [-2.5, -2.5])
if __name__ == '__main__':
    unittest.main()