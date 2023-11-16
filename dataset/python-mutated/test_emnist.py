import unittest
import jittor as jt
from jittor.dataset.mnist import EMNIST, MNIST
import jittor.transform as transform

@unittest.skipIf(True, f'skip emnist test')
class TestEMNIST(unittest.TestCase):

    def test_emnist(self):
        if False:
            i = 10
            return i + 15
        import pylab as pl
        emnist_dataset = EMNIST()
        for (imgs, labels) in emnist_dataset:
            print(imgs.shape, labels.shape)
            print(labels.max(), labels.min())
            imgs = imgs.transpose(1, 2, 0, 3)[0].reshape(28, -1)
            print(labels)
            (pl.imshow(imgs), pl.show())
            break
if __name__ == '__main__':
    unittest.main()