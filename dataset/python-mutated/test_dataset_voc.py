import unittest
import numpy as np
from paddle.vision.datasets import VOC2012, voc2012
voc2012.VOC_URL = 'https://paddlemodels.bj.bcebos.com/voc2012_stub/VOCtrainval_11-May-2012.tar'
voc2012.VOC_MD5 = '34cb1fe5bdc139a5454b25b16118fff8'

class TestVOC2012Train(unittest.TestCase):

    def test_main(self):
        if False:
            while True:
                i = 10
        voc2012 = VOC2012(mode='train')
        self.assertTrue(len(voc2012) == 3)
        idx = np.random.randint(0, 3)
        (image, label) = voc2012[idx]
        image = np.array(image)
        label = np.array(label)
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)

class TestVOC2012Valid(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        voc2012 = VOC2012(mode='valid')
        self.assertTrue(len(voc2012) == 1)
        idx = np.random.randint(0, 1)
        (image, label) = voc2012[idx]
        image = np.array(image)
        label = np.array(label)
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)

class TestVOC2012Test(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        voc2012 = VOC2012(mode='test')
        self.assertTrue(len(voc2012) == 2)
        idx = np.random.randint(0, 1)
        (image, label) = voc2012[idx]
        image = np.array(image)
        label = np.array(label)
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)
        voc2012 = VOC2012(mode='test', backend='cv2')
        self.assertTrue(len(voc2012) == 2)
        idx = np.random.randint(0, 1)
        (image, label) = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)
        with self.assertRaises(ValueError):
            voc2012 = VOC2012(mode='test', backend=1)
if __name__ == '__main__':
    unittest.main()