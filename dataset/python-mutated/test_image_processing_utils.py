import unittest
from transformers.image_processing_utils import get_size_dict

class ImageProcessingUtilsTester(unittest.TestCase):

    def test_get_size_dict(self):
        if False:
            while True:
                i = 10
        inputs = {'wrong_key': 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)
        inputs = {'height': 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)
        inputs = {'width': 224, 'shortest_edge': 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)
        inputs = {'height': 224, 'width': 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, inputs)
        inputs = {'shortest_edge': 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, {'shortest_edge': 224})
        inputs = {'longest_edge': 224, 'shortest_edge': 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, {'longest_edge': 224, 'shortest_edge': 224})
        outputs = get_size_dict(224)
        self.assertEqual(outputs, {'height': 224, 'width': 224})
        outputs = get_size_dict(224, default_to_square=False)
        self.assertEqual(outputs, {'shortest_edge': 224})
        outputs = get_size_dict((150, 200))
        self.assertEqual(outputs, {'height': 150, 'width': 200})
        outputs = get_size_dict((150, 200), height_width_order=False)
        self.assertEqual(outputs, {'height': 200, 'width': 150})
        outputs = get_size_dict(224, max_size=256, default_to_square=False)
        self.assertEqual(outputs, {'shortest_edge': 224, 'longest_edge': 256})
        with self.assertRaises(ValueError):
            get_size_dict(224, max_size=256, default_to_square=True)