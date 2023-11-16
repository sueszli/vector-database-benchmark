import unittest
import numpy as np
from Orange.data import Table
from Orange.preprocess import Continuize
from Orange.projection import FreeViz

class TestFreeviz(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.iris = Table('iris')
        cls.housing = Table('housing')
        cls.zoo = Table('zoo')

    def test_basic(self):
        if False:
            while True:
                i = 10
        table = self.iris.copy()
        with table.unlocked():
            table[3, 3] = np.nan
        freeviz = FreeViz()
        model = freeviz(table)
        proj = model(table)
        self.assertEqual(len(proj), len(table))
        self.assertTrue(np.isnan(proj.X).any())
        np.testing.assert_array_equal(proj[:100], model(table[:100]))

    def test_regression(self):
        if False:
            i = 10
            return i + 15
        table = Table('housing')[::10]
        freeviz = FreeViz()
        freeviz(table)
        freeviz = FreeViz(p=2)
        freeviz(table)

    @unittest.skip('Test weights is too slow.')
    def test_weights(self):
        if False:
            return 10
        table = Table('iris')
        weights = np.random.rand(150, 1).flatten()
        freeviz = FreeViz(weights=weights, p=3, scale=False, center=False)
        freeviz(table)
        scale = np.array([0.5, 0.4, 0.6, 0.8])
        freeviz = FreeViz(scale=scale, center=[0.2, 0.6, 0.4, 0.2])
        freeviz(table)

    def test_raising_errors(self):
        if False:
            print('Hello World!')
        table = Table('iris')
        scale = np.array([0.5, 0.4, 0.6])
        freeviz = FreeViz(scale=scale)
        self.assertRaises(ValueError, freeviz, table)
        freeviz = FreeViz(center=[0.6, 0.4, 0.2])
        self.assertRaises(ValueError, freeviz, table)
        weights = np.random.rand(100, 1).flatten()
        freeviz = FreeViz(weights=weights)
        self.assertRaises(ValueError, freeviz, table)
        table = Table('titanic')[::10]
        freeviz = FreeViz()
        self.assertRaises(ValueError, freeviz, table)

    def test_initial(self):
        if False:
            i = 10
            return i + 15
        FreeViz.init_radial(1)
        FreeViz.init_radial(2)
        FreeViz.init_radial(3)
        FreeViz.init_random(2, 4, 5)

    def test_transform_changed_domain(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        1. Open data, apply some preprocessor, splits the data into two parts,\n        use FreeViz on the first part, and then transform the second part.\n\n        2. Open data, split into two parts, apply the same preprocessor and\n        FreeViz only on the first part, and then transform the second part.\n\n        The transformed second part in (1) and (2) has to be the same.\n        '
        data = Table('titanic')[::10]
        normalize = Continuize()
        freeviz = FreeViz(maxiter=40)
        ndata = normalize(data)
        model = freeviz(ndata[:100])
        result_1 = model(ndata[100:])
        ndata = normalize(data[:100])
        model = freeviz(ndata)
        result_2 = model(data[100:])
        np.testing.assert_almost_equal(result_1.X, result_2.X)