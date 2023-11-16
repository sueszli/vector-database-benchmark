"""ModeKey Tests."""
from tensorflow.python.platform import test
from tensorflow.python.saved_model.model_utils import mode_keys

class ModeKeyMapTest(test.TestCase):

    def test_map(self):
        if False:
            print('Hello World!')
        mode_map = mode_keys.ModeKeyMap(**{mode_keys.KerasModeKeys.PREDICT: 3, mode_keys.KerasModeKeys.TEST: 1})
        self.assertEqual(3, mode_map[mode_keys.KerasModeKeys.PREDICT])
        self.assertEqual(3, mode_map[mode_keys.EstimatorModeKeys.PREDICT])
        self.assertEqual(1, mode_map[mode_keys.KerasModeKeys.TEST])
        self.assertEqual(1, mode_map[mode_keys.EstimatorModeKeys.EVAL])
        with self.assertRaises(KeyError):
            _ = mode_map[mode_keys.KerasModeKeys.TRAIN]
        with self.assertRaises(KeyError):
            _ = mode_map[mode_keys.EstimatorModeKeys.TRAIN]
        with self.assertRaisesRegex(ValueError, 'Invalid mode'):
            _ = mode_map['serve']
        self.assertLen(mode_map, 2)
        self.assertEqual({1, 3}, set(mode_map.values()))
        self.assertEqual({mode_keys.KerasModeKeys.TEST, mode_keys.KerasModeKeys.PREDICT}, set(mode_map.keys()))
        with self.assertRaises(TypeError):
            mode_map[mode_keys.KerasModeKeys.TEST] = 1

    def test_invalid_init(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'Multiple keys/values found'):
            _ = mode_keys.ModeKeyMap(**{mode_keys.KerasModeKeys.PREDICT: 3, mode_keys.EstimatorModeKeys.PREDICT: 1})
if __name__ == '__main__':
    test.main()