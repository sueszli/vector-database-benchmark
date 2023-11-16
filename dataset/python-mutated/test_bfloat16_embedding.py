import unittest
import numpy as np
from test_sparse_attention_op import get_cuda_version
import paddle
import paddle.nn.functional as F

class BF16EmbeddingTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = 30
        self.vocab_size = 1024
        self.hidden_size = 512
        self.seed = 10

    def run_main(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        (ids, weight, dout) = self.gen_random()
        origin_dtype = weight.dtype
        weight_cast = weight.astype(dtype)
        out = F.embedding(ids, weight_cast)
        dout = dout.astype(out.dtype)
        dweight = paddle.autograd.grad(out, weight, dout)
        return (out.astype(origin_dtype).numpy(), dweight[0].astype(origin_dtype).numpy())

    def gen_random(self):
        if False:
            return 10
        np.random.seed(self.seed)
        weight = np.random.random([self.vocab_size, self.hidden_size]).astype('float32')
        ids = np.random.randint(low=0, high=self.vocab_size, size=[self.batch_size])
        dout = np.random.random([self.batch_size, self.hidden_size]).astype('float32')
        weight = paddle.to_tensor(weight)
        weight.stop_gradient = False
        ids = paddle.to_tensor(ids)
        dout = paddle.to_tensor(dout)
        return (ids, weight, dout)

    def test_main(self):
        if False:
            print('Hello World!')
        if not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000:
            return
        ret1 = self.run_main('float32')
        ret2 = self.run_main('bfloat16')
        self.assertEqual(len(ret1), len(ret2))
        for (i, (r1, r2)) in enumerate(zip(ret1, ret2)):
            np.testing.assert_allclose(r1, r2, atol=0.001, rtol=0.01)

class BF16EmbeddingTestOddHiddenSize(BF16EmbeddingTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.batch_size = 30
        self.vocab_size = 511
        self.hidden_size = 512
        self.seed = 20
if __name__ == '__main__':
    unittest.main()