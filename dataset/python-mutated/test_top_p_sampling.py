import unittest
import numpy as np
import paddle
from paddle.base import core

def TopPProcess(probs, top_p):
    if False:
        return 10
    sorted_probs = paddle.sort(probs, descending=True)
    sorted_indices = paddle.argsort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype='int64')
    sorted_indices_to_remove = paddle.static.setitem(sorted_indices_to_remove, (slice(None), slice(1, None)), sorted_indices_to_remove[:, :-1].clone())
    sorted_indices_to_remove = paddle.static.setitem(sorted_indices_to_remove, (slice(None), 0), 0)
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
    condition = paddle.scatter(sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten())
    condition = paddle.cast(condition, 'bool').reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    next_tokens = paddle.multinomial(probs)
    next_scores = paddle.index_sample(probs, next_tokens)
    return (next_scores, next_tokens)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA ')
class TestTopPAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.topp = 0.0
        self.seed = 6688
        self.batch_size = 3
        self.vocab_size = 10000
        self.dtype = 'float32'
        self.input_data = np.random.rand(self.batch_size, self.vocab_size)

    def run_dygraph(self, place):
        if False:
            while True:
                i = 10
        with paddle.base.dygraph.guard(place):
            input_tensor = paddle.to_tensor(self.input_data, self.dtype)
            topp_tensor = paddle.to_tensor([self.topp] * self.batch_size, self.dtype).reshape((-1, 1))
            paddle_result = paddle.tensor.top_p_sampling(input_tensor, topp_tensor, seed=self.seed)
            ref_res = TopPProcess(input_tensor, self.topp)
            np.testing.assert_allclose(paddle_result[0].numpy(), ref_res[0].numpy(), rtol=1e-05)
            np.testing.assert_allclose(paddle_result[1].numpy().flatten(), ref_res[1].numpy().flatten(), rtol=0)

    def run_static(self, place):
        if False:
            return 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            input_tensor = paddle.static.data(name='x', shape=[6, 1030], dtype=self.dtype)
            topp_tensor = paddle.static.data(name='topp', shape=[6, 1], dtype=self.dtype)
            result = paddle.tensor.top_p_sampling(input_tensor, topp_tensor, seed=self.seed)
            ref_res = TopPProcess(input_tensor, self.topp)
            exe = paddle.static.Executor(place)
            input_data = np.random.rand(6, 1030).astype(self.dtype)
            paddle_result = exe.run(feed={'x': input_data, 'topp': np.array([self.topp] * 6).astype(self.dtype)}, fetch_list=[result[0], result[1], ref_res[0], ref_res[1]])
            np.testing.assert_allclose(paddle_result[0], paddle_result[2], rtol=1e-05)
            np.testing.assert_allclose(paddle_result[1], paddle_result[3], rtol=1e-05)

    def test_cases(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            places = [core.CUDAPlace(0)]
            for place in places:
                self.run_dygraph(place)
                self.run_static(place)
if __name__ == '__main__':
    unittest.main()