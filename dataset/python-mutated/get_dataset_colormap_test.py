"""Tests for get_dataset_colormap.py."""
import numpy as np
import tensorflow as tf
from deeplab.utils import get_dataset_colormap

class VisualizationUtilTest(tf.test.TestCase):

    def testBitGet(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that if the returned bit value is correct.'
        self.assertEqual(1, get_dataset_colormap.bit_get(9, 0))
        self.assertEqual(0, get_dataset_colormap.bit_get(9, 1))
        self.assertEqual(0, get_dataset_colormap.bit_get(9, 2))
        self.assertEqual(1, get_dataset_colormap.bit_get(9, 3))

    def testPASCALLabelColorMapValue(self):
        if False:
            while True:
                i = 10
        'Test the getd color map value.'
        colormap = get_dataset_colormap.create_pascal_label_colormap()
        self.assertTrue(np.array_equal([128.0, 0.0, 128.0], colormap[5, :]))
        self.assertTrue(np.array_equal([128.0, 192.0, 128.0], colormap[23, :]))
        self.assertTrue(np.array_equal([128.0, 0.0, 192.0], colormap[37, :]))
        self.assertTrue(np.array_equal([224.0, 192.0, 192.0], colormap[127, :]))
        self.assertTrue(np.array_equal([192.0, 160.0, 192.0], colormap[175, :]))

    def testLabelToPASCALColorImage(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the value of the converted label value.'
        label = np.array([[0, 16, 16], [52, 7, 52]])
        expected_result = np.array([[[0, 0, 0], [0, 64, 0], [0, 64, 0]], [[0, 64, 192], [128, 128, 128], [0, 64, 192]]])
        colored_label = get_dataset_colormap.label_to_color_image(label, get_dataset_colormap.get_pascal_name())
        self.assertTrue(np.array_equal(expected_result, colored_label))

    def testUnExpectedLabelValueForLabelToPASCALColorImage(self):
        if False:
            print('Hello World!')
        'Raise ValueError when input value exceeds range.'
        label = np.array([[120], [600]])
        with self.assertRaises(ValueError):
            get_dataset_colormap.label_to_color_image(label, get_dataset_colormap.get_pascal_name())

    def testUnExpectedLabelDimensionForLabelToPASCALColorImage(self):
        if False:
            i = 10
            return i + 15
        'Raise ValueError if input dimension is not correct.'
        label = np.array([120])
        with self.assertRaises(ValueError):
            get_dataset_colormap.label_to_color_image(label, get_dataset_colormap.get_pascal_name())

    def testGetColormapForUnsupportedDataset(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            get_dataset_colormap.create_label_colormap('unsupported_dataset')

    def testUnExpectedLabelDimensionForLabelToADE20KColorImage(self):
        if False:
            return 10
        label = np.array([250])
        with self.assertRaises(ValueError):
            get_dataset_colormap.label_to_color_image(label, get_dataset_colormap.get_ade20k_name())

    def testFirstColorInADE20KColorMap(self):
        if False:
            print('Hello World!')
        label = np.array([[1, 3], [10, 20]])
        expected_result = np.array([[[120, 120, 120], [6, 230, 230]], [[4, 250, 7], [204, 70, 3]]])
        colored_label = get_dataset_colormap.label_to_color_image(label, get_dataset_colormap.get_ade20k_name())
        self.assertTrue(np.array_equal(colored_label, expected_result))

    def testMapillaryVistasColorMapValue(self):
        if False:
            return 10
        colormap = get_dataset_colormap.create_mapillary_vistas_label_colormap()
        self.assertTrue(np.array_equal([190, 153, 153], colormap[3, :]))
        self.assertTrue(np.array_equal([102, 102, 156], colormap[6, :]))
if __name__ == '__main__':
    tf.test.main()