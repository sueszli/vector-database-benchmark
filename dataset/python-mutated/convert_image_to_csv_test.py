"""Tests image file conversion utilities."""
import os
import numpy as np
from tensorflow.lite.tools import convert_image_to_csv
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
PREFIX_PATH = resource_loader.get_path_to_datafile('../../core/lib/')

class ConvertImageToCsvTest(test_util.TensorFlowTestCase):

    def testGetImageRaisesMissingFile(self):
        if False:
            print('Hello World!')
        image_path = os.path.join(PREFIX_PATH, 'jpeg', 'testdata', 'no_such.jpg')
        with self.assertRaises(NotFoundError):
            _ = convert_image_to_csv.get_image(64, 96, False, image_path)

    def testGetImageSizeIsCorrect(self):
        if False:
            return 10
        image_path = os.path.join(PREFIX_PATH, 'jpeg', 'testdata', 'small.jpg')
        image_data = convert_image_to_csv.get_image(64, 96, False, image_path)
        self.assertEqual((96, 64, 3), image_data.shape)

    def testGetImageConvertsToGrayscale(self):
        if False:
            i = 10
            return i + 15
        image_path = os.path.join(PREFIX_PATH, 'jpeg', 'testdata', 'medium.jpg')
        image_data = convert_image_to_csv.get_image(40, 20, True, image_path)
        self.assertEqual((20, 40, 1), image_data.shape)

    def testGetImageCanLoadPng(self):
        if False:
            i = 10
            return i + 15
        image_path = os.path.join(PREFIX_PATH, 'png', 'testdata', 'lena_rgba.png')
        image_data = convert_image_to_csv.get_image(10, 10, False, image_path)
        self.assertEqual((10, 10, 3), image_data.shape)

    def testGetImageConvertsGrayscaleToColor(self):
        if False:
            for i in range(10):
                print('nop')
        image_path = os.path.join(PREFIX_PATH, 'png', 'testdata', 'lena_gray.png')
        image_data = convert_image_to_csv.get_image(23, 19, False, image_path)
        self.assertEqual((19, 23, 3), image_data.shape)

    def testGetImageColorValuesInRange(self):
        if False:
            for i in range(10):
                print('nop')
        image_path = os.path.join(PREFIX_PATH, 'jpeg', 'testdata', 'small.jpg')
        image_data = convert_image_to_csv.get_image(47, 31, False, image_path)
        self.assertLessEqual(0, np.min(image_data))
        self.assertGreaterEqual(255, np.max(image_data))

    def testGetImageGrayscaleValuesInRange(self):
        if False:
            return 10
        image_path = os.path.join(PREFIX_PATH, 'jpeg', 'testdata', 'small.jpg')
        image_data = convert_image_to_csv.get_image(27, 33, True, image_path)
        self.assertLessEqual(0, np.min(image_data))
        self.assertGreaterEqual(255, np.max(image_data))

    def testArrayToIntCsv(self):
        if False:
            for i in range(10):
                print('nop')
        csv_string = convert_image_to_csv.array_to_int_csv(np.array([[1, 2], [3, 4]]))
        self.assertEqual('1,2,3,4', csv_string)

    def testArrayToIntCsvRounding(self):
        if False:
            while True:
                i = 10
        csv_string = convert_image_to_csv.array_to_int_csv(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual('1,2,3,4', csv_string)
if __name__ == '__main__':
    test.main()