import unittest
import numpy as np
from paddle.vision.datasets import Cifar10, Cifar100

class TestCifar10Train(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        cifar = Cifar10(mode='train')
        self.assertTrue(len(cifar) == 50000)
        idx = np.random.randint(0, 50000)
        (data, label) = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(0 <= int(label) <= 9)

class TestCifar10Test(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        cifar = Cifar10(mode='test')
        self.assertTrue(len(cifar) == 10000)
        idx = np.random.randint(0, 10000)
        (data, label) = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(0 <= int(label) <= 9)
        cifar = Cifar10(mode='test', backend='cv2')
        self.assertTrue(len(cifar) == 10000)
        idx = np.random.randint(0, 10000)
        (data, label) = cifar[idx]
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(0 <= int(label) <= 99)
        with self.assertRaises(ValueError):
            cifar = Cifar10(mode='test', backend=1)

class TestCifar100Train(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        cifar = Cifar100(mode='train')
        self.assertTrue(len(cifar) == 50000)
        idx = np.random.randint(0, 50000)
        (data, label) = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(0 <= int(label) <= 99)

class TestCifar100Test(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        cifar = Cifar100(mode='test')
        self.assertTrue(len(cifar) == 10000)
        idx = np.random.randint(0, 10000)
        (data, label) = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(0 <= int(label) <= 99)
        cifar = Cifar100(mode='test', backend='cv2')
        self.assertTrue(len(cifar) == 10000)
        idx = np.random.randint(0, 10000)
        (data, label) = cifar[idx]
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(0 <= int(label) <= 99)
        with self.assertRaises(ValueError):
            cifar = Cifar100(mode='test', backend=1)
if __name__ == '__main__':
    unittest.main()