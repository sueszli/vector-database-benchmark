import unittest
import paddle

class TestAdagradOpV2(unittest.TestCase):

    def test_v20_coverage(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        inp = paddle.rand(shape=[10, 10])
        linear = paddle.nn.Linear(10, 10)
        out = linear(inp)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(learning_rate=0.1, parameters=linear.parameters())
        out.backward()
        adagrad.step()
        adagrad.clear_grad()

class TestAdagradOpV2Group(TestAdagradOpV2):

    def test_v20_coverage(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        inp = paddle.rand(shape=[10, 10])
        linear_1 = paddle.nn.Linear(10, 10)
        linear_2 = paddle.nn.Linear(10, 10)
        out = linear_1(inp)
        out = linear_2(out)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=[{'params': linear_1.parameters()}, {'params': linear_2.parameters(), 'weight_decay': 0.001}], weight_decay=0.1)
        out.backward()
        adagrad.step()
        adagrad.clear_grad()
if __name__ == '__main__':
    unittest.main()