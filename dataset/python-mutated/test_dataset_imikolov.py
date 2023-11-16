import unittest
import numpy as np
from paddle.text.datasets import Imikolov

class TestImikolovTrain(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        imikolov = Imikolov(mode='train', data_type='NGRAM', window_size=2)
        self.assertTrue(len(imikolov) == 929589)
        idx = np.random.randint(0, 929589)
        data = imikolov[idx]
        self.assertTrue(len(data) == 2)

class TestImikolovTest(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        imikolov = Imikolov(mode='test', data_type='NGRAM', window_size=2)
        self.assertTrue(len(imikolov) == 82430)
        idx = np.random.randint(0, 82430)
        data = imikolov[idx]
        self.assertTrue(len(data) == 2)
if __name__ == '__main__':
    unittest.main()