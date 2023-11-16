import unittest
import numpy as np
import paddle
paddle.disable_static()

class EmbeddingDygraph(unittest.TestCase):

    def test_1(self):
        if False:
            while True:
                i = 10
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)
        embedding = paddle.nn.Embedding(10, 3, sparse=True, padding_idx=9)
        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding.weight.set_value(w0)
        adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
        adam.clear_grad()
        out = embedding(x)
        out.backward()
        adam.step()

    def test_2(self):
        if False:
            return 10
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)
        y = paddle.to_tensor(y_data, stop_gradient=False)
        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(10, 3, padding_idx=11, sparse=True)
        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(-1, 3, sparse=True)
        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(10, -3, sparse=True)
if __name__ == '__main__':
    unittest.main()