import unittest
import paddle

class StaticShapeInferrenceTest(unittest.TestCase):

    def test_static_graph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        data = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        shape = paddle.shape(data)
        x = paddle.uniform(shape)
        paddle.utils.try_set_static_shape_tensor(x, shape)
        self.assertEqual(x.shape, data.shape)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()