import unittest
import numpy as np
from paddle.text.datasets import Movielens

class TestMovielensTrain(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        movielens = Movielens(mode='train')
        idx = np.random.randint(0, 900000)
        data = movielens[idx]
        self.assertTrue(len(data) == 8)
        for (i, d) in enumerate(data):
            self.assertTrue(len(d.shape) == 1)
            if i not in [5, 6]:
                self.assertTrue(d.shape[0] == 1)

class TestMovielensTest(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        movielens = Movielens(mode='test')
        idx = np.random.randint(0, 100000)
        data = movielens[idx]
        self.assertTrue(len(data) == 8)
        for (i, d) in enumerate(data):
            self.assertTrue(len(d.shape) == 1)
            if i not in [5, 6]:
                self.assertTrue(d.shape[0] == 1)
if __name__ == '__main__':
    unittest.main()