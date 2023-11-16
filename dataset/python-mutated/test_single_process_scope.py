import unittest
import os, sys
import jittor as jt
import numpy as np
from jittor.test.test_mpi import run_mpi_test
mpi = jt.compile_extern.mpi
from jittor.dataset.mnist import MNIST

def val1():
    if False:
        i = 10
        return i + 15
    dataloader = MNIST(train=False).set_attrs(batch_size=16)
    for (i, (imgs, labels)) in enumerate(dataloader):
        assert imgs.shape[0] == 8
        if i == 5:
            break

@jt.single_process_scope(rank=0)
def val2():
    if False:
        while True:
            i = 10
    dataloader = MNIST(train=False).set_attrs(batch_size=16)
    for (i, (imgs, labels)) in enumerate(dataloader):
        assert imgs.shape[0] == 16
        if i == 5:
            break

@unittest.skipIf(not jt.in_mpi, 'no inside mpirun')
class TestSingleProcessScope(unittest.TestCase):

    def test_single_process_scope(self):
        if False:
            print('Hello World!')
        val1()
        val2()

@unittest.skipIf(not jt.compile_extern.has_mpi, 'no mpi found')
class TestSingleProcessScopeEntry(unittest.TestCase):

    def test_entry(self):
        if False:
            return 10
        run_mpi_test(2, 'test_single_process_scope')
if __name__ == '__main__':
    unittest.main()