import unittest
import jittor as jt
import numpy as np

class TestIndexOp(unittest.TestCase):

    def test(self):
        if False:
            print('Hello World!')
        assert (jt.index([2, 2], 0).data == [[0, 0], [1, 1]]).all()
        assert (jt.index([2, 2], 1).data == [[0, 1], [0, 1]]).all()
        a = jt.index([2, 2], 0)
        b = jt.index([2, 2], 1)
        c = a + b
        assert (c.data == [[0, 1], [1, 2]]).all(), c.data

    def test_multioutput(self):
        if False:
            return 10
        (a, b) = jt.index([2, 2])
        jt.sync([a, b])
        assert (a.data == [[0, 0], [1, 1]]).all()
        assert (b.data == [[0, 1], [0, 1]]).all(), b.data

    def test_multioutput2(self):
        if False:
            while True:
                i = 10
        (a, b) = jt.index([3, 3])
        assert (a.data == [[0, 0, 0], [1, 1, 1], [2, 2, 2]]).all()
        assert (b.data == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]).all(), b.data
        (a, b) = jt.index([3, 3])
        c = a + b
        assert (c.data == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]).all(), c.data

    def test_multioutput3(self):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = jt.index([3, 3])
        del a
        assert (b.data == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]).all(), b.data

    def test_vary_shape_dep(self):
        if False:
            i = 10
            return i + 15
        (a,) = jt.where([1, 0, 1])
        (b,) = a.index_var()
        assert (b.data == [0, 1]).all()

    def test_vary_shape_dep2(self):
        if False:
            while True:
                i = 10
        a = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        (index0,) = jt.where(a.sum(1) > 7)
        index0 = index0.broadcast([1, 3], dims=[1])
        index1 = index0.index_var(1)
        b = a.reindex_var([index0, index1])
        assert (b.data == [[4, 5, 6], [7, 8, 9]]).all()
        assert (index0.data == [[1, 1, 1], [2, 2, 2]]).all()
        assert (index1.data == [[0, 1, 2], [0, 1, 2]]).all()

    def test_doc(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'Index Operator' in jt.index.__doc__

    def test_wrong_fuse(self):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = jt.index([10, 10])
        c = jt.zeros([10, 10])
        c = c.reindex([b + 1, a])
        x = b.clone()
        jt.sync([c, x])
if __name__ == '__main__':
    unittest.main()