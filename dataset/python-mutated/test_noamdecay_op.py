import unittest
import paddle

class TestSparseEmbeddingAPIError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with paddle.base.dygraph.guard():

            def test_0_d_model():
                if False:
                    while True:
                        i = 10
                schedular = paddle.optimizer.lr.NoamDecay(d_model=0, warmup_steps=0)
            self.assertRaises(ValueError, test_0_d_model)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()