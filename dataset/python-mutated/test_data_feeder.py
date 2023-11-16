import unittest
import paddle
from paddle import base
paddle.enable_static()

class TestDataFeeder(unittest.TestCase):

    def test_lod_level_0_converter(self):
        if False:
            while True:
                i = 10
        img = paddle.static.data(name='image', shape=[-1, 1, 28, 28])
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        feeder = base.DataFeeder([img, label], base.CPUPlace())
        result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])
        self.assertEqual(result['image'].shape(), [2, 1, 28, 28])
        self.assertEqual(result['label'].shape(), [2, 1])
        self.assertEqual(result['image'].recursive_sequence_lengths(), [])
        self.assertEqual(result['label'].recursive_sequence_lengths(), [])
        try:
            result = feeder.feed([([0] * 783, [9]), ([1] * 783, [1])])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_lod_level_1_converter(self):
        if False:
            i = 10
            return i + 15
        sentences = paddle.static.data(name='sentences', shape=[-1, 1], dtype='int64', lod_level=1)
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        feeder = base.DataFeeder([sentences, label], base.CPUPlace())
        result = feeder.feed([([1, 2, 3], [1]), ([4, 5], [1]), ([6, 7, 8, 9], [1])])
        self.assertEqual(result['sentences'].shape(), [9, 1])
        self.assertEqual(result['label'].shape(), [3, 1])
        self.assertEqual(result['sentences'].recursive_sequence_lengths(), [[3, 2, 4]])
        self.assertEqual(result['label'].recursive_sequence_lengths(), [])

    def test_lod_level_2_converter(self):
        if False:
            print('Hello World!')
        paragraphs = paddle.static.data(name='paragraphs', shape=[-1, 1], dtype='int64', lod_level=2)
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        feeder = base.DataFeeder([paragraphs, label], base.CPUPlace())
        result = feeder.feed([([[1, 2, 3], [4, 5]], [1]), ([[6, 7, 8, 9]], [1])])
        self.assertEqual(result['paragraphs'].shape(), [9, 1])
        self.assertEqual(result['label'].shape(), [2, 1])
        self.assertEqual(result['paragraphs'].recursive_sequence_lengths(), [[2, 1], [3, 2, 4]])
        self.assertEqual(result['label'].recursive_sequence_lengths(), [])
if __name__ == '__main__':
    unittest.main()