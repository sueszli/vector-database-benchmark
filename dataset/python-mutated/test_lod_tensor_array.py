import unittest
import numpy as np
import paddle
from paddle.base import core

class TestLoDTensorArray(unittest.TestCase):

    def test_get_set(self):
        if False:
            while True:
                i = 10
        scope = core.Scope()
        arr = scope.var('tmp_lod_tensor_array')
        tensor_array = arr.get_lod_tensor_array()
        self.assertEqual(0, len(tensor_array))
        cpu = core.CPUPlace()
        for i in range(10):
            t = core.LoDTensor()
            t.set(np.array([i], dtype='float32'), cpu)
            t.set_recursive_sequence_lengths([[1]])
            tensor_array.append(t)
        self.assertEqual(10, len(tensor_array))
        for i in range(10):
            t = tensor_array[i]
            self.assertEqual(np.array(t), np.array([i], dtype='float32'))
            self.assertEqual([[1]], t.recursive_sequence_lengths())
            t = core.LoDTensor()
            t.set(np.array([i + 10], dtype='float32'), cpu)
            t.set_recursive_sequence_lengths([[1]])
            tensor_array[i] = t
            t = tensor_array[i]
            self.assertEqual(np.array(t), np.array([i + 10], dtype='float32'))
            self.assertEqual([[1]], t.recursive_sequence_lengths())

class TestCreateArray(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.place = paddle.CPUPlace()
        self.shapes = [[10, 4], [8, 12], [1]]

    def test_initialized_list_and_error(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        init_data = [np.random.random(shape).astype('float32') for shape in self.shapes]
        array = paddle.tensor.create_array('float32', [paddle.to_tensor(x) for x in init_data])
        for (res, gt) in zip(array, init_data):
            np.testing.assert_array_equal(res, gt)
        array = paddle.tensor.create_array('float32')
        self.assertTrue(isinstance(array, list))
        self.assertEqual(len(array), 0)
        with self.assertRaises(TypeError):
            paddle.tensor.create_array('float32', 'str')

    def test_static(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        init_data = [paddle.ones(shape, dtype='int32') for shape in self.shapes]
        array = paddle.tensor.create_array('float32', init_data)
        for (res, gt) in zip(array, init_data):
            self.assertTrue(res.shape, gt.shape)
        with self.assertRaises(TypeError):
            paddle.tensor.create_array('float32', [init_data[0], [init_data[1]]])
        with self.assertRaises(TypeError):
            paddle.tensor.create_array('float32', 'str')
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()