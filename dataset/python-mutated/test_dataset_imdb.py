import unittest
import numpy as np
from paddle.text.datasets import Imdb

class TestImdbTrain(unittest.TestCase):

    def test_main(self):
        if False:
            while True:
                i = 10
        imdb = Imdb(mode='train')
        self.assertTrue(len(imdb) == 25000)
        idx = np.random.randint(0, 25000)
        (data, label) = imdb[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(int(label) in [0, 1])

class TestImdbTest(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        imdb = Imdb(mode='test')
        self.assertTrue(len(imdb) == 25000)
        idx = np.random.randint(0, 25000)
        (data, label) = imdb[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(int(label) in [0, 1])
if __name__ == '__main__':
    unittest.main()