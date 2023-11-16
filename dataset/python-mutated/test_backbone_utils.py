import unittest
from transformers.utils.backbone_utils import BackboneMixin, get_aligned_output_features_output_indices, verify_out_features_out_indices

class BackboneUtilsTester(unittest.TestCase):

    def test_get_aligned_output_features_output_indices(self):
        if False:
            i = 10
            return i + 15
        stage_names = ['a', 'b', 'c']
        (out_features, out_indices) = get_aligned_output_features_output_indices(None, None, stage_names)
        self.assertEqual(out_features, ['c'])
        self.assertEqual(out_indices, [2])
        (out_features, out_indices) = get_aligned_output_features_output_indices(['a', 'c'], None, stage_names)
        self.assertEqual(out_features, ['a', 'c'])
        self.assertEqual(out_indices, [0, 2])
        (out_features, out_indices) = get_aligned_output_features_output_indices(None, [0, 2], stage_names)
        self.assertEqual(out_features, ['a', 'c'])
        self.assertEqual(out_indices, [0, 2])
        (out_features, out_indices) = get_aligned_output_features_output_indices(None, [-3, -1], stage_names)
        self.assertEqual(out_features, ['a', 'c'])
        self.assertEqual(out_indices, [-3, -1])

    def test_verify_out_features_out_indices(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(['a', 'b'], (0, 1), None)
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(('a', 'b'), (0, 1), ['a', 'b'])
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(['a', 'b'], (0, 1), ['a'])
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(None, 0, ['a', 'b'])
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(None, (0, 1), ['a'])
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(['a', 'b'], (0,), ['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(['a', 'b'], (0, 2), ['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(['b', 'a'], (0, 1), ['a', 'b'])
        verify_out_features_out_indices(['a', 'b', 'd'], (0, 1, -1), ['a', 'b', 'c', 'd'])

    def test_backbone_mixin(self):
        if False:
            for i in range(10):
                print('nop')
        backbone = BackboneMixin()
        backbone.stage_names = ['a', 'b', 'c']
        backbone._out_features = ['a', 'c']
        backbone._out_indices = [0, 2]
        self.assertEqual(backbone.out_features, ['a', 'c'])
        self.assertEqual(backbone.out_indices, [0, 2])
        backbone.out_features = ['a', 'b']
        self.assertEqual(backbone.out_features, ['a', 'b'])
        self.assertEqual(backbone.out_indices, [0, 1])
        backbone.out_indices = [-3, -1]
        self.assertEqual(backbone.out_features, ['a', 'c'])
        self.assertEqual(backbone.out_indices, [-3, -1])