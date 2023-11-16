import unittest
import jittor as jt
import numpy as np

class TestWhereOp(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.where = jt.where

    def test(self):
        if False:
            return 10
        assert (self.where([0, 1, 0, 1])[0].data == [1, 3]).all()
        (a,) = self.where([0, 1, 0, 1])
        assert a.uncertain_shape == [2]
        a.data
        assert a.uncertain_shape == [2]
        (a, b) = self.where([[0, 0, 1], [1, 0, 0]])
        assert (a.data == [0, 1]).all() and (b.data == [2, 0]).all()

    def test_reindex_dep(self):
        if False:
            while True:
                i = 10
        a = jt.random([10])
        (b,) = self.where(a > 1)
        assert len(b.data) == 0
        (b,) = self.where(a > 0.5)
        assert (b.data == np.where(a.data > 0.5)).all()
        b = a.reindex_var(self.where(a > 0.5))
        assert (b.data == a.data[a.data > 0.5]).all()

    def test_binary_dep(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.random([10])
        (b,) = self.where(a > 0.5)
        b = b + 1
        assert (b.data == np.where(a.data > 0.5)[0] + 1).all()
        (b,) = self.where(a > 1)
        b = b + 1
        assert (b.data == np.where(a.data > 1)[0] + 1).all()

    def test_self_dep(self):
        if False:
            while True:
                i = 10
        a = jt.random([100])
        x = a.reindex_var(self.where(a > 0.1))
        x = x.reindex_var(self.where(x < 0.9))
        na = a.data
        assert np.allclose(na[np.logical_and(na > 0.1, na < 0.9)], x.data)

    def test_reduce_dep(self):
        if False:
            return 10
        a = jt.random([100, 100])
        index = self.where(a > 0.5)
        assert isinstance(index, tuple)
        x = a.reindex_var(index)
        xsum = x.sum()
        na = a.data
        assert np.allclose(np.sum(na[na > 0.5]), xsum.data), (x.data, xsum.data, np.sum(na[na > 0.5]))

    def test_doc(self):
        if False:
            i = 10
            return i + 15
        assert 'Where Operator' in jt.where.__doc__

@unittest.skipIf(not jt.has_cuda, 'No Torch found')
class TestWhereOpCuda(TestWhereOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.where = jt.where

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        jt.flags.use_cuda = 1

    @classmethod
    def tearDownClass(self):
        if False:
            while True:
                i = 10
        jt.flags.use_cuda = 0

@unittest.skipIf(not jt.has_cuda, 'No Torch found')
class TestWhereOpCub(TestWhereOpCuda):

    def setUp(self):
        if False:
            print('Hello World!')
        self.where = jt.compile_extern.cub_ops.cub_where

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        jt.flags.use_cuda = 1

    @classmethod
    def tearDownClass(self):
        if False:
            for i in range(10):
                print('nop')
        jt.flags.use_cuda = 0
if __name__ == '__main__':
    unittest.main()