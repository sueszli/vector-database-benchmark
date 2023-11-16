import unittest
import paddle

class TestSparseEmbeddingAPIError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with paddle.base.dygraph.guard():

            def test_0_size():
                if False:
                    return 10
                input = paddle.to_tensor([], dtype='int64')
                paddle.static.nn.sparse_embedding(input, [2097152, 2097152, 2097152, 2097152], padding_idx=2097152)
            self.assertRaises(ValueError, test_0_size)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()