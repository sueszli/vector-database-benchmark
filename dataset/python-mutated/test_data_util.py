import unittest
import warnings
from unittest.mock import Mock
import numpy as np
from Orange.data.util import scale, one_hot, SharedComputeValue
import Orange

class TestDataUtil(unittest.TestCase):

    def test_scale(self):
        if False:
            return 10
        np.testing.assert_equal(scale([0, 1, 2], -1, 1), [-1, 0, 1])
        np.testing.assert_equal(scale([3, 3, 3]), [1, 1, 1])
        np.testing.assert_equal(scale([0.1, 0.5, np.nan]), [0, 1, np.nan])
        np.testing.assert_equal(scale(np.array([])), np.array([]))

    def test_one_hot(self):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_equal(one_hot([0, 1, 2, 1], int), [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_equal(one_hot([], int), np.zeros((0, 0), dtype=int))

class DummyPlus(SharedComputeValue):

    def compute(self, data, shared_data):
        if False:
            i = 10
            return i + 15
        return data.X[:, 0] + shared_data

class DummyTable(Orange.data.Table):
    pass

class TestSharedComputeValue(unittest.TestCase):

    def test_compat_compute_value(self):
        if False:
            for i in range(10):
                print('nop')
        data = Orange.data.Table('iris')
        obj = DummyPlus(lambda data: 1.0)
        res = obj(data)
        obj = lambda data: data.X[:, 0] + 1.0
        res2 = obj(data)
        np.testing.assert_equal(res, res2)

    def test_with_row_indices(self):
        if False:
            for i in range(10):
                print('nop')
        obj = DummyPlus(lambda data: 1.0)
        data = Orange.data.Table('iris')
        domain = Orange.data.Domain([Orange.data.ContinuousVariable('cv', compute_value=obj)])
        data1 = Orange.data.Table.from_table(domain, data)[:10]
        data2 = Orange.data.Table.from_table(domain, data, range(10))
        np.testing.assert_equal(data1.X, data2.X)

    def test_single_call(self):
        if False:
            return 10
        obj = DummyPlus(Mock(return_value=1))
        self.assertEqual(obj.compute_shared.call_count, 0)
        data = Orange.data.Table('iris')[45:55]
        domain = Orange.data.Domain([at.copy(compute_value=obj) for at in data.domain.attributes], data.domain.class_vars)
        Orange.data.Table.from_table(domain, data)
        self.assertEqual(obj.compute_shared.call_count, 1)
        ndata = Orange.data.Table.from_table(domain, data)
        self.assertEqual(obj.compute_shared.call_count, 2)
        c = Orange.classification.LogisticRegressionLearner()(ndata)
        self.assertEqual(obj.compute_shared.call_count, 2)
        c(data)
        self.assertEqual(obj.compute_shared.call_count, 3)
        DummyTable.from_table(c.domain, data)
        self.assertEqual(obj.compute_shared.call_count, 4)

    def test_compute_shared_eq_warning(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as warns:
            DummyPlus(compute_shared=lambda *_: 42)

            class Valid:

                def __eq__(self, other):
                    if False:
                        print('Hello World!')
                    pass

                def __hash__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass
            DummyPlus(compute_shared=Valid())
            self.assertEqual(warns, [])

            class Invalid:
                pass
            DummyPlus(compute_shared=Invalid())
            self.assertNotEqual(warns, [])
        with warnings.catch_warnings(record=True) as warns:

            class MissingHash:

                def __eq__(self, other):
                    if False:
                        i = 10
                        return i + 15
                    pass
            DummyPlus(compute_shared=MissingHash())
            self.assertNotEqual(warns, [])
        with warnings.catch_warnings(record=True) as warns:

            class MissingEq:

                def __hash__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass
            DummyPlus(compute_shared=MissingEq())
            self.assertNotEqual(warns, [])
        with warnings.catch_warnings(record=True) as warns:

            class Subclass(Valid):
                pass
            DummyPlus(compute_shared=Subclass())
            self.assertNotEqual(warns, [])

    def test_eq_hash(self):
        if False:
            print('Hello World!')
        x = Orange.data.ContinuousVariable('x')
        y = Orange.data.ContinuousVariable('y')
        x2 = Orange.data.ContinuousVariable('x')
        assert x == x2
        assert hash(x) == hash(x2)
        assert x != y
        assert hash(x) != hash(y)
        c1 = SharedComputeValue(abs, x)
        c2 = SharedComputeValue(abs, x2)
        d = SharedComputeValue(abs, y)
        e = SharedComputeValue(len, x)
        self.assertNotEqual(c1, None)
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))
        self.assertNotEqual(c1, d)
        self.assertNotEqual(hash(c1), hash(d))
        self.assertNotEqual(c1, e)
        self.assertNotEqual(hash(c1), hash(e))