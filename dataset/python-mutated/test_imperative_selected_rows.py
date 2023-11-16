import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import to_variable

class SimpleNet(paddle.nn.Layer):

    def __init__(self, vocab_size, hidden_size, dtype):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.emb = paddle.nn.Embedding(vocab_size, hidden_size, weight_attr='emb.w', sparse=True)

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        input_emb = self.emb(input)
        return (input_emb, self.emb)

class TestSimpleNet(unittest.TestCase):

    def test_selectedrows_gradient1(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            for dtype in ['float32', 'float64']:
                for sort_sum_gradient in [True, False]:
                    paddle.disable_static(place)
                    base.set_flags({'FLAGS_sort_sum_gradient': sort_sum_gradient})
                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = paddle.to_tensor(input_word)
                    simplenet = SimpleNet(20, 32, dtype)
                    adam = paddle.optimizer.SGD(learning_rate=0.001, parameters=simplenet.parameters())
                    (input_emb, emb) = simplenet(input)
                    input_emb.retain_grads()
                    self.assertIsNone(emb.weight.gradient())
                    self.assertIsNone(input_emb.gradient())
                    input_emb.backward()
                    adam.minimize(input_emb)
                    self.assertIsNotNone(emb.weight.gradient())
                    emb.clear_gradients()
                    self.assertIsNone(emb.weight.gradient())
                    input_emb.clear_gradient()
                    self.assertIsNotNone(input_emb.gradient())
                    paddle.enable_static()

    def test_selectedrows_gradient2(self):
        if False:
            print('Hello World!')
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            for sort_sum_gradient in [True, False]:
                with base.dygraph.guard(place):
                    base.set_flags({'FLAGS_sort_sum_gradient': sort_sum_gradient})
                    grad_clip = paddle.nn.ClipGradByGlobalNorm(5.0)
                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = to_variable(input_word)
                    simplenet = SimpleNet(20, 32, 'float32')
                    adam = paddle.optimizer.SGD(learning_rate=0.001, parameters=simplenet.parameters(), grad_clip=grad_clip)
                    (input_emb, emb) = simplenet(input)
                    input_emb.retain_grads()
                    self.assertIsNone(emb.weight.gradient())
                    self.assertIsNone(input_emb.gradient())
                    input_emb.backward()
                    adam.minimize(input_emb)
                    self.assertIsNotNone(emb.weight.gradient())
                    emb.clear_gradients()
                    self.assertIsNone(emb.weight.gradient())
                    input_emb.clear_gradient()
                    self.assertIsNotNone(input_emb.gradient())
if __name__ == '__main__':
    unittest.main()