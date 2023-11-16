import unittest
import os, sys
import jittor as jt
import numpy as np
from jittor.test.test_mpi import run_mpi_test
mpi = jt.compile_extern.mpi
if mpi:
    n = mpi.world_size()

@unittest.skipIf(not jt.in_mpi, 'no inside mpirun')
class TestMpiOps(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        jt.seed(3)

    def test_all_reduce(self):
        if False:
            while True:
                i = 10
        x = jt.random([5, 5])
        y = x.mpi_all_reduce()
        np.testing.assert_allclose(y.data, (x * n).data)
        g = jt.grad(y, x)
        np.testing.assert_allclose(g.data, np.ones([5, 5]) * n)

    def test_all_reduce_mean(self):
        if False:
            i = 10
            return i + 15
        x = jt.random([5, 5])
        y = x.mpi_all_reduce('mean')
        np.testing.assert_allclose(y.data, x.data)
        g = jt.grad(y, x)
        np.testing.assert_allclose(g.data, np.ones([5, 5]))

    def test_broadcast(self):
        if False:
            i = 10
            return i + 15
        data = jt.random([5, 5])
        if mpi.world_rank() == 0:
            x = data
        else:
            x = jt.zeros([5, 5])
        y = x.mpi_broadcast(0)
        np.testing.assert_allclose(y.data, data.data)
        g = jt.grad(y, x)
        if mpi.world_rank() == 0:
            np.testing.assert_allclose(g.data, np.ones([5, 5]) * n)
        else:
            np.testing.assert_allclose(g.data, np.zeros([5, 5]))

    def test_reduce(self):
        if False:
            while True:
                i = 10
        x = jt.random([5, 5])
        y = x.mpi_reduce(root=0)
        y.sync()
        if mpi.world_rank() == 0:
            np.testing.assert_allclose(y.data, (x * n).data)
        else:
            np.testing.assert_allclose(y.data, np.zeros([5, 5]))
        g = jt.grad(y, x)
        print(mpi.world_rank(), g)
        np.testing.assert_allclose(g.data, np.ones([5, 5]))

@unittest.skipIf(not jt.compile_extern.has_mpi, 'no mpi found')
class TestMpiOpsEntry(unittest.TestCase):

    def test(self):
        if False:
            return 10
        run_mpi_test(2, 'test_mpi_op')
if __name__ == '__main__':
    unittest.main()