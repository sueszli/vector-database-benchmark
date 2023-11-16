import unittest
import numpy as np
from paddle.text.datasets import WMT14, WMT16

class TestWMT14Train(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
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
            i = 10
            return i + 15
        wmt14 = WMT14(mode='gen', dict_size=50)
        self.assertTrue(len(wmt14) == 3001)
        idx = np.random.randint(0, 3001)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)

class TestWMT16Train(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 29000)
        idx = np.random.randint(0, 29000)
        data = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)

class TestWMT16Test(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        wmt16 = WMT16(mode='test', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 1000)
        idx = np.random.randint(0, 1000)
        data = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)

class TestWMT16Val(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        wmt16 = WMT16(mode='val', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 1014)
        idx = np.random.randint(0, 1014)
        data = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)
if __name__ == '__main__':
    unittest.main()