import unittest
import jittor as jt
import numpy as numpy

class TestMergeLoopVarPass(unittest.TestCase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.ones([10, 10, 10, 10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.sum([2, 3])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert 'range01' in src
            assert 'range23' in src

    def test2(self):
        if False:
            while True:
                i = 10
        a = jt.ones([10, 10, 10, 10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a + 1
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert 'range0123' in src

    def test3(self):
        if False:
            while True:
                i = 10
        a = jt.ones([10, 10, 10, 10])
        x = jt.ones([1, 10, 1, 1])
        (a.sync(), x.sync())
        with jt.profile_scope() as rep:
            b = a + x
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert 'range23' in src

    def test4(self):
        if False:
            while True:
                i = 10
        a = jt.ones([10, 10, 10, 10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.reindex_reduce('add', [10, 10], ['i0', 'i1'])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert 'range23' not in src

    def test5(self):
        if False:
            print('Hello World!')
        a = jt.ones([10, 10, 10, 10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.sum([1])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert 'range01' not in src
            assert 'range23' in src

@unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
class TestMergeLoopVarPassCuda(TestMergeLoopVarPass):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        jt.flags.use_cuda = 1

    def tearDown(self):
        if False:
            print('Hello World!')
        jt.flags.use_cuda = 0
if __name__ == '__main__':
    unittest.main()