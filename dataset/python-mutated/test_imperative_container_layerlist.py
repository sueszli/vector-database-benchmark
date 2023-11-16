import unittest
import numpy as np
import paddle
from paddle import base

class MyLayer(paddle.nn.Layer):

    def __init__(self, layerlist):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layerlist = layerlist

    def forward(self, x):
        if False:
            while True:
                i = 10
        for l in self.layerlist:
            x = l(x)
        return x

class TestImperativeContainer(unittest.TestCase):

    def paddle_imperative_list(self):
        if False:
            i = 10
            return i + 15
        return paddle.nn.LayerList([paddle.nn.Linear(2 ** i, 2 ** (i + 1)) for i in range(6)])

    def layer_list(self, use_base_api):
        if False:
            print('Hello World!')
        data_np = np.random.uniform(-1, 1, [5, 1]).astype('float32')
        with base.dygraph.guard():
            x = base.dygraph.to_variable(data_np)
            layerlist = self.paddle_imperative_list()
            size = len(layerlist)
            model = MyLayer(layerlist)
            res1 = model(x)
            self.assertListEqual(res1.shape, [5, 2 ** size])
            model.layerlist[size - 1] = paddle.nn.Linear(2 ** (size - 1), 5)
            res2 = model(x)
            self.assertListEqual(res2.shape, [5, 5])
            del model.layerlist[size - 1]
            res3 = model(x)
            self.assertListEqual(res3.shape, [5, 2 ** (size - 1)])
            model.layerlist.append(paddle.nn.Linear(2 ** (size - 1), 3))
            res4 = model(x)
            self.assertListEqual(res4.shape, [5, 3])
            res4.backward()
            model2 = MyLayer(layerlist[:-1])
            res5 = model2(x)
            self.assertListEqual(res5.shape, [5, 2 ** (size - 1)])
            del model2.layerlist[1:]
            res6 = model2(x)
            self.assertListEqual(res6.shape, [5, 2 ** (0 + 1)])
            res6.backward()
            model3 = MyLayer(layerlist[:-2])
            model3.layerlist.append(paddle.nn.Linear(3, 1))
            model3.layerlist.insert(size - 2, paddle.nn.Linear(2 ** (size - 2), 3))
            res7 = model3(x)
            self.assertListEqual(res7.shape, [5, 1])
            to_be_extended = [paddle.nn.Linear(3 ** i, 3 ** (i + 1)) for i in range(3)]
            model3.layerlist.extend(to_be_extended)
            res8 = model3(x)
            self.assertListEqual(res8.shape, [5, 3 ** 3])
            res8.backward()
            model4 = MyLayer(layerlist[:3])
            model4.layerlist[-1] = paddle.nn.Linear(4, 5)
            res9 = model4(x)
            self.assertListEqual(res9.shape, [5, 5])
            del model4.layerlist[-1]
            res10 = model4(x)
            self.assertListEqual(res10.shape, [5, 4])
            model4.layerlist.insert(-1, paddle.nn.Linear(2, 2))
            res11 = model4(x)
            self.assertListEqual(res11.shape, [5, 4])
            res11.backward()

    def test_test_layer_list(self):
        if False:
            while True:
                i = 10
        self.layer_list(True)
        self.layer_list(False)
if __name__ == '__main__':
    unittest.main()