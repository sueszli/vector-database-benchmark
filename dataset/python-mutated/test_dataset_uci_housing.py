import unittest
import numpy as np
from paddle.text.datasets import WMT14, UCIHousing

class TestUCIHousingTrain(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        uci_housing = UCIHousing(mode='train')
        self.assertTrue(len(uci_housing) == 404)
        idx = np.random.randint(0, 404)
        data = uci_housing[idx]
        self.assertTrue(len(data) == 2)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(data[0].shape[0] == 13)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(data[1].shape[0] == 1)

class TestUCIHousingTest(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        uci_housing = UCIHousing(mode='test')
        self.assertTrue(len(uci_housing) == 102)
        idx = np.random.randint(0, 102)
        data = uci_housing[idx]
        self.assertTrue(len(data) == 2)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(data[0].shape[0] == 13)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(data[1].shape[0] == 1)

class TestWMT14Train(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        wmt14 = WMT14(mode='train', dict_size=50)
        self.assertTrue(len(wmt14) == 191155)
        idx = np.random.randint(0, 191155)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)

class TestWMT14Test(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        wmt14 = WMT14(mode='test', dict_size=50)
        self.assertTrue(len(wmt14) == 5957)
        idx = np.random.randint(0, 5957)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)

class TestWMT14Gen(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        wmt14 = WMT14(mode='gen', dict_size=50)
        self.assertTrue(len(wmt14) == 3001)
        idx = np.random.randint(0, 3001)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)
if __name__ == '__main__':
    unittest.main()