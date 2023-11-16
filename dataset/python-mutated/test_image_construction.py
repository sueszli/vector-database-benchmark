import pytest
from tests.base import TestBase
from tests.utils import is_package_installed
from aim.sdk import Image
import numpy as np

class TestImageConstruction(TestBase):
    img1 = np.array([[[0, 1], [1, 0]]], np.uint8)
    img2 = np.array([[0, 1], [1, 0]], np.uint8)
    img3 = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 1], [1, 0]]], np.uint8)

    @pytest.mark.skipif(not is_package_installed('torch'), reason="'torch' is not installed. skipping.")
    def test_image_from_torch_tensor(self):
        if False:
            while True:
                i = 10
        import torch
        self.assertEqual(Image(torch.tensor(self.img1)), Image(np.transpose(self.img1, (1, 2, 0))))
        self.assertEqual(Image(torch.tensor(self.img2)), Image(self.img2))
        self.assertEqual(Image(torch.tensor(self.img3)), Image(np.transpose(self.img3, (1, 2, 0))))
        self.assertEqual(Image(torch.tensor(self.img1.astype(np.float32))), Image(255 * np.transpose(self.img1, (1, 2, 0))))
        self.assertEqual(Image(torch.tensor(self.img2.astype(np.float32))), Image(255 * self.img2))
        self.assertEqual(Image(torch.tensor(self.img3.astype(np.float32))), Image(255 * np.transpose(self.img3, (1, 2, 0))))

    @pytest.mark.skipif(not is_package_installed('tensorflow'), reason="'tensorflow' is not installed. skipping.")
    def test_image_from_tf_tensor(self):
        if False:
            while True:
                i = 10
        import tensorflow as tf
        self.assertEqual(Image(tf.convert_to_tensor(self.img1, tf.uint8)), Image(self.img1))
        self.assertEqual(Image(tf.convert_to_tensor(self.img2, tf.uint8)), Image(self.img2))
        self.assertEqual(Image(tf.convert_to_tensor(self.img3, tf.uint8)), Image(self.img3))
        self.assertEqual(Image(tf.convert_to_tensor(self.img1, tf.float32)), Image(255 * self.img1))
        self.assertEqual(Image(tf.convert_to_tensor(self.img2, tf.float32)), Image(255 * self.img2))
        self.assertEqual(Image(tf.convert_to_tensor(self.img3, tf.float32)), Image(255 * self.img3))
if __name__ == '__main__':
    TestImageConstruction().test_image_from_torch_tensor()
    TestImageConstruction().test_image_from_tf_tensor()