import unittest
import numpy as np
from chainermn.datasets import create_empty_dataset
import chainerx as chx

class TestEmptyDataset(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def check_create_empty_dataset(self, original_dataset):
        if False:
            i = 10
            return i + 15
        empty_dataset = create_empty_dataset(original_dataset)
        self.assertEqual(len(original_dataset), len(empty_dataset))
        for i in range(len(original_dataset)):
            self.assertEqual((), empty_dataset[i])

    def test_empty_dataset_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_empty_dataset(np)

    def test_empty_dataset_chx(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_empty_dataset(chx)

    def check_empty_dataset(self, xp):
        if False:
            return 10
        n = 10
        self.check_create_empty_dataset([])
        self.check_create_empty_dataset([0])
        self.check_create_empty_dataset(list(range(n)))
        self.check_create_empty_dataset(list(range(n * 5 - 1)))
        self.check_create_empty_dataset(xp.array([]))
        self.check_create_empty_dataset(xp.array([0]))
        self.check_create_empty_dataset(xp.arange(n))
        self.check_create_empty_dataset(xp.arange(n * 5 - 1))